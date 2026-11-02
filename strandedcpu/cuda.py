"""Batched ISA execution with one CUDA thread per program.

Words use unsigned 64-bit arithmetic, wrapped to N bits; carries, division
and runtime checks use 128-bit intermediates. N ranges from 2 to 64.
Binary fix checks require N <= 32 because their products need 4N bits.

run_batch returns each program's memory, status and fault. Fault details are
reconstructed with the interpreter's check code on the stopped memory.
The shipped kernel needs a CUDA GPU but no compiler. Runtime compilation
is available when the prebuilt image cannot be loaded.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from strandedcpu.compiler import ExecutionFault
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.transformer import logadd_table, vrand_constants

CHECK_KINDS = ('add', 'sub', 'mul', 'div', 'mod', 'shl', 'shr', 'abs', 'bounds',
               'fadd', 'fsub', 'fmul', 'fdiv', 'fdivi', 'fneg', 'rcast', 'fcast')
CHECK_CODE = {k: i + 1 for i, k in enumerate(CHECK_KINDS)}

# The kernel: plain CUDA C++ with no headers, so NVRTC (the runtime compiler)
# and nvcc build the same text.
# The private loom_kernel symbol is the ABI of the shipped binary. Keep this
# source byte-for-byte stable unless rebuilding kernel.fatbin and its metadata.
_DEVICE = r"""
typedef long long i64;
typedef unsigned long long u64;
typedef __int128 i128;
typedef unsigned __int128 u128;

struct Cfg { int N, s, m, T, max_lv; i64 max_steps; };

// Word arithmetic is done on u64 (two's complement bits, wrapping mod 2^64)
// and then wrapped to N bits, so N = 64 has no signed overflow anywhere.
__device__ __forceinline__ u64 maskN(int N) { return N >= 64 ? ~0ULL : (1ULL << N) - 1; }
__device__ __forceinline__ i64 wrapN(u64 v, int N) {
    u64 u = v & maskN(N);
    if (N < 64 && (u >> (N - 1))) u |= ~maskN(N);
    return (i64)u;
}
__device__ __forceinline__ u64 uns(i64 v, int N) { return (u64)v & maskN(N); }
__device__ __forceinline__ int bitlen(u64 u) { return u ? 64 - __clzll((long long)u) : 0; }

__device__ __forceinline__ bool is_page2(int a) {
    return (a >= 64 && a <= 79) || (a >= 84 && a <= 88);
}

__device__ i64 shift_op(i64 b, i64 c, int N) {
    if (c >= 0) return c >= N ? 0 : wrapN((u64)b << c, N);
    i64 k = -c;
    return k >= 63 ? (b < 0 ? -1 : 0) : (b >> k);
}

__device__ i64 logadd_op(i64 b, i64 c, const i64* table, const Cfg& g) {
    u64 d = b > c ? (u64)b - (u64)c : (u64)c - (u64)b;
    return wrapN((u64)(b > c ? b : c) + (u64)(d < (u64)g.T ? table[d] : 0), g.N);
}

// prog: per instruction 8 words (op, b, c, check kind, 4 check slots) -- one
// 64-byte line per fetch. res: per program 8 words (status, calls, executed,
// fault instruction, int_div_remainder, fix_rounding, carry, unused).
extern "C" __global__ void loom_kernel(const i64* __restrict__ prog, const i64* __restrict__ lens, i64* mem,
                                       i64* scratch, i64* res, const i64* __restrict__ consts, int N_, int s_, int m_,
                                       int T_, i64 max_steps, int B, int L) {
    const Cfg g{N_, s_, m_, T_, 20, max_steps};
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= B) return;
    const int N = g.N, s = g.s, m = g.m;
    const i64 lo = (i64)(~0ULL << (N - 1)), hi = ~lo;     // -2^(N-1), 2^(N-1) - 1
    const i64* table = consts;                  // LOGADD corrections, then VRAND constants
    const i64* vr = consts + g.T;
    i64* M = mem + (i64)p * m;
    i64* VC = scratch + (i64)p * 3 * m;
    i64* OLD = VC + m;
    i64* OLDC = OLD + m;
    for (int j = 0; j < m; j++) VC[j] = 0;
    const i64* P = prog + (i64)p * L * 8;
    const i64 len = lens[p];

    auto get = [&](i64 addr) -> i64 { i64 i = addr - s; return (i >= 0 && i < m) ? M[i] : 0; };
    auto set = [&](i64 addr, u64 v) { i64 i = addr - s; if (i >= 0 && i < m) M[i] = wrapN(v, N); };
    auto word = [&](i64 slot) -> i64 { return (slot >= 0 && slot < m) ? M[slot] : 0; };
    auto pointer = [&](i64 v) -> i64 { return s + (i64)(uns(v, N) % (u64)s); };
    int kbits = bitlen((u64)s) - 1;
    auto xlat_addr = [&](i64 v, i64 col) -> i64 {
        int k = kbits < N ? kbits : N;
        return col | (i64)(uns(v, N) & maskN(k));
    };

    i64 pc = s + m, calls = 0, executed = 0, i_div = 0, i_round = 0;
    int carry = 0, st = 1;          // 0 halted, 1 step limit, 2 overflow fault, 3 runtime fault
    i64 vl_log = 0, at = -1;
    while (calls < g.max_steps) {
        calls++;
        i64 i = pc - s - m;
        if (pc == 0 || i < 0 || i >= len) { pc = 0; st = 0; break; }
        const i64* ins = P + i * 8;
        int kind = (int)ins[3];
        if (kind) {
            const i64* sl = ins + 4;
            int fault = 0;          // 2 overflow, 3 runtime error
            if (kind <= 3) {                                   // add sub mul
                i64 x = word(sl[0]), y = word(sl[1]);
                i128 v = kind == 1 ? (i128)x + y : kind == 2 ? (i128)x - y : (i128)x * y;
                if (v < lo || v > hi) fault = 2;
            } else if (kind == 4 || kind == 5) {               // div mod
                i64 x = word(sl[0]), y = word(sl[1]);
                if (y == 0) fault = 3;
                else if (x == lo && y == -1) fault = 2;
                else if (kind == 4 && x % y) i_div++;
            } else if (kind == 6 || kind == 7) {               // shl shr
                i64 x = word(sl[0]), k = word(sl[1]);
                if (k < 0 || k >= N) fault = 3;
                else if (kind == 6) { i128 v = (i128)x * ((i128)1 << k); if (v < lo || v > hi) fault = 2; }
            } else if (kind == 8) {                            // abs
                if (word(sl[0]) == lo) fault = 2;
            } else if (kind == 9) {                            // bounds
                i64 ptr = word(sl[0]);
                if (!(sl[1] <= ptr && ptr < sl[1] + sl[2])) fault = 3;
            } else if (kind <= 15) {                           // binary fix: values R / 2^N
                i128 Ra = (i128)word(sl[0]) * ((i128)1 << N) + (i128)uns(word(sl[1]), N);
                i128 top = (i128)1 << (2 * N - 1);             // 2^(N-1) in raw units
                if (kind == 15) {                              // fneg
                    i128 v = -Ra;
                    if (v < -top || v >= top) fault = 2;
                } else if (kind == 14) {                       // fdivi: v = Ra / (2^N k)
                    i64 k = word(sl[2]);
                    if (k == 0) fault = 3;
                    else {
                        i128 bound = top * (i128)k;            // compare Ra / k against +-top
                        bool ok = k > 0 ? (Ra >= -bound && Ra < bound) : (Ra <= -bound && Ra > bound);
                        if (!ok) fault = 2;
                        else if (Ra % (i128)k) i_round++;
                    }
                } else {
                    i128 Rb = (i128)word(sl[2]) * ((i128)1 << N) + (i128)uns(word(sl[3]), N);
                    if (kind == 10 || kind == 11) {            // fadd fsub
                        i128 v = kind == 10 ? Ra + Rb : Ra - Rb;
                        if (v < -top || v >= top) fault = 2;
                    } else if (kind == 12) {                   // fmul: v = Ra Rb / 2^(2N)
                        i128 prod = Ra * Rb;
                        i128 lim = top << N;                   // 2^(3N-1)
                        if (prod < -lim || prod >= lim) fault = 2;
                        else if (prod & (((i128)1 << N) - 1)) i_round++;
                    } else {                                   // fdiv: v = Ra / Rb
                        if (Rb == 0) fault = 3;
                        else {
                            i128 bound = (top >> N) * Rb;      // 2^(N-1) Rb
                            bool ok = Rb > 0 ? (Ra >= -bound && Ra < bound) : (Ra <= -bound && Ra > bound);
                            if (!ok) fault = 2;
                            else if ((Ra * ((i128)1 << N)) % Rb) i_round++;
                        }
                    }
                }
            } else if (kind == 16) {                           // rcast
                i64 d = word(sl[1]);
                if (d && d != -1 && word(sl[0]) % d) i_round++;
            } else if (kind == 17) {                           // fcast
                if (uns(word(sl[1]), N)) i_round++;
            }
            if (fault) { st = fault; at = i; break; }
        }
        executed++;
        pc++;
        const int a = (int)ins[0];
        const i64 b = ins[1], c = ins[2];
        const i64 vb = get(b), vc = get(c);
        if (a >= 32 && !is_page2(a)) {                         // SUBLEQ
            i64 r = wrapN((u64)vb - (u64)get(a), N);
            set(b, r);
            if (r <= 0) pc = c;
            continue;
        }
        switch (a) {
        case 0: pc = 0; st = 0; break;
        case 1: set(b, vc); break;
        case 2: carry = (int)(((u128)uns(vb, N) + uns(vc, N)) >> N); set(b, (u64)vb + (u64)vc); break;
        case 3: pc = c; break;
        case 4: if (vb == 0) pc = c; break;
        case 5: if (vb != 0) pc = c; break;
        case 6: set(b, (u64)vb + 1); break;
        case 7: set(b, (u64)vb - 1); break;
        case 8: set(b, (u64)vb << 1); break;
        case 9: set(b, vb >> 1); break;
        case 10: if (vb < 0) pc = c; break;
        case 11: set(b, get(pointer(vc))); break;
        case 12: set(b, vb & vc); break;
        case 13: set(b, vb | vc); break;
        case 14: set(b, vb ^ vc); break;
        case 15: carry = uns(vb, N) >= uns(vc, N); set(b, (u64)vb - (u64)vc); break;
        case 16: { i64 f = 0; for (int j = 0; j < m; j++) if (M[j] == vc) { f = j; break; } set(b, f); break; }
        case 17: set(b, vc); set(c, vb); break;
        case 18: if (vb < 0) set(b, vc); break;
        case 19: set(b, ((u64)vb << 1) + (vb < 0 ? (u64)vc : 0)); break;
        case 20: set(pointer(vc), vb); break;
        case 21: set(b, vb > vc ? vb : vc); break;
        case 22: set(b, logadd_op(vb, vc, table, g)); break;
        case 23: set(b, get(xlat_addr(vb, c))); break;
        case 24: set(b, shift_op(vb, vc, N)); break;
        case 25: { u128 t = (u128)uns(vb, N) + uns(vc, N) + (u64)carry; carry = (int)(t >> N); set(b, (u64)t); break; }
        case 26: {                                             // MULSTEP on (b, b^1)
            u64 lou = uns(get(b ^ 1), N);
            u128 t = (u128)uns(vb, N) + ((lou & 1) ? uns(vc, N) : 0);
            set(b, (u64)(t >> 1));
            set(b ^ 1, ((u64)(t & 1) << (N - 1)) | (lou >> 1));
            break;
        }
        case 27: {                                             // DIVSTEP on (b, b^1)
            u64 qu = uns(get(b ^ 1), N);
            u128 r = 2 * (u128)uns(vb, N) + (qu >> (N - 1));
            u64 q = (qu << 1) & maskN(N);
            if (r >= uns(vc, N)) { r -= uns(vc, N); q |= 1; }
            set(b, (u64)r); set(b ^ 1, q);
            break;
        }
        case 28: { u64 u = uns(vb, N); set(b, (u >> 1) ^ ((u & 1) ? uns(vc, N) : 0)); break; }
        case 29: set(b, N - bitlen(uns(vc, N))); break;
        case 30: set(b, vc < 0 ? 0 - (u64)vc : (u64)vc); break;
        case 31: {                                             // ARGMAX over 2^vb words at c
            i64 count = (vb >= 0 && vb < 31) ? (1LL << vb) : 1, best = 0;
            for (i64 j = 1; j < count; j++) if (get(c + j) > get(c + best)) best = j;
            set(b, best);
            break;
        }
        case 64: pc = s + m + (i64)uns(vc, N); break;          // JMPI
        case 65: {                                             // FALIGN
            i128 d = (i128)vc - vb;
            if (d > 0) {
                i64 mant = get(b ^ 1);
                i64 k = d < N - 1 ? (i64)d : N - 1;
                set(b, vc); set(b ^ 1, mant >> k);
            }
            break;
        }
        case 66: {                                             // FNORM
            i64 mant = get(b ^ 1);
            if (mant == 0) { set(b, lo); set(b ^ 1, 0); break; }
            u64 u = uns(mant, N);
            u64 sign = u >> (N - 1);
            int run = 0;
            while (run < N && ((u >> (N - 1 - run)) & 1) == sign) run++;
            int k = run - 1;
            set(b, (u64)vb - (u64)k); set(b ^ 1, (u64)mant << k);
            break;
        }
        case 67: vl_log = vb; break;                           // VSETVL
        default: {                                             // vector unit
            if (vl_log < 0 || vl_log > g.max_lv) { st = 3; at = i; break; }
            const i64 vl = 1LL << vl_log, t = vl_log;
            const i64 bb = b & ~(vl - 1), cb = c & ~(vl - 1);
            for (int j = 0; j < m; j++) { OLD[j] = M[j]; OLDC[j] = VC[j]; }
            auto oget = [&](i64 col) -> i64 { i64 k = col - s; return (k >= 0 && k < m) ? OLD[k] : 0; };
            auto ocar = [&](i64 col) -> i64 { i64 k = col - s; return (k >= 0 && k < m) ? OLDC[k] : 0; };
            const i64 msk = (i64)(uns(vc, N) & (u64)(vl - 1));
            for (i64 e = 0; e < vl; e++) {
                i64 col = bb + e, j = col - s;
                if (j < 0 || j >= m) continue;
                i64 x = OLD[j], y = 0;
                u64 r = 0;
                if (a >= 68 && a <= 72) y = oget(cb + e);
                else if (a >= 84 && a <= 88) y = oget(bb + (e ^ msk));
                switch (a) {
                case 68: case 84: r = y; break;
                case 69: case 85: r = (u64)x + (u64)y; VC[j] = (i64)(((u128)uns(x, N) + uns(y, N)) >> N); break;
                case 70: r = (u64)x - (u64)y; VC[j] = uns(x, N) >= uns(y, N); break;
                case 71: case 87: r = x > y ? x : y; break;
                case 72: case 88: r = logadd_op(x, y, table, g); break;
                case 73: { i64 k = ocar(cb + e); r = (u64)x + (u64)k; VC[j] = (i64)(((u128)uns(x, N) + (u64)k) >> N); break; }
                case 86: r = (col & (i64)uns(vc, N)) ? (u64)y - (u64)x : (u64)x + (u64)y; break;
                case 74: r = vc; break;
                case 75: r = shift_op(x, vc, N); break;
                case 76: r = oget(xlat_addr(x, c)); break;
                case 77: { i64 rv = 0; for (i64 k = 0; k < t; k++) rv = (rv << 1) | ((e >> k) & 1); r = oget(cb + rv); break; }
                case 78: { i64 ti = e; if (t % 2 == 0) { i64 h = t / 2; ti = ((e & ((1LL << h) - 1)) << h) | (e >> h); }
                           r = oget(cb + ti); break; }
                case 79: {                                     // VRAND
                    u64 mask = maskN(N), u = uns(vc, N);
                    for (int q = 0; q < N; q++) if ((j >> q) & 1) u = (u + (u64)vr[q]) & mask;
                    u64 v = u ^ (u >> vr[N + 1]);
                    u64 w = ((u64)vr[N] * v) & mask;
                    r = w ^ (w >> vr[N + 2]);
                    break;
                }
                default: r = x; break;
                }
                M[j] = wrapN(r, N);
            }
            break;
        }
        }
        if (st == 0 || st == 3) break;
    }
    i64* R = res + (i64)p * 8;
    R[0] = st; R[1] = calls; R[2] = executed; R[3] = at; R[4] = i_div; R[5] = i_round; R[6] = carry; R[7] = 0;
}

"""

# nvcc build: the kernel plus a PyTorch launcher on the current stream
_TORCH_LAUNCHER = r"""
void run(torch::Tensor prog, torch::Tensor lens, torch::Tensor mem, torch::Tensor scratch,
              torch::Tensor res, torch::Tensor consts, int64_t T, int64_t N, int64_t s, int64_t m,
              int64_t max_steps) {
    int B = prog.size(0), L = prog.size(1);
    int threads = 64, blocks = (B + threads - 1) / threads;
    auto p = [](torch::Tensor t) { return reinterpret_cast<i64*>(t.data_ptr<int64_t>()); };
    loom_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        p(prog), p(lens), p(mem), p(scratch), p(res), p(consts), (int)N, (int)s, (int)m, (int)T,
        (i64)max_steps, B, L);
}
"""
_SOURCE = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + _DEVICE + _TORCH_LAUNCHER

_CPP = ("void run(torch::Tensor prog, torch::Tensor lens, torch::Tensor mem, torch::Tensor scratch, "
        "torch::Tensor res, torch::Tensor consts, int64_t T, int64_t N, int64_t s, int64_t m, int64_t max_steps);")

_ext = None


BUILD: Dict[str, object] = {'mode': None, 'errors': []}


def _find_nvrtc() -> str:
    """Find libnvrtc in the PyTorch installation, NVIDIA wheels or system libraries."""
    import ctypes.util
    import glob
    import os
    import re
    site = os.path.dirname(os.path.dirname(torch.__file__))
    found = (glob.glob(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libnvrtc.so*'))
             + glob.glob(os.path.join(site, 'nvidia', '*', 'lib', 'libnvrtc.so*')))
    found = [f for f in found if re.search(r'libnvrtc\.so(\.\d+)*$', f)]
    if found:
        return sorted(found)[-1]
    name = ctypes.util.find_library('nvrtc')
    if name:
        return name
    raise RuntimeError("libnvrtc not found")


class _DriverKernel:
    """The kernel loaded from a compiled image (cubin, PTX or a fatbin of
    both) through the CUDA driver API and launched on PyTorch's current
    stream. Same run signature as the nvcc extension."""

    def __init__(self, image: bytes):
        import ctypes
        torch.zeros(1, device='cuda')
        cuda = ctypes.CDLL('libcuda.so.1')
        self.image = ctypes.create_string_buffer(image, len(image))
        self.module, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        for call, args in (('cuModuleLoadData', (ctypes.byref(self.module), self.image)),
                           ('cuModuleGetFunction', (ctypes.byref(self.fn), self.module, b'loom_kernel'))):
            rc = getattr(cuda, call)(*args)
            if rc:
                raise RuntimeError(f"{call} failed with CUDA error {rc}")
        cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p,
                                                                                ctypes.POINTER(ctypes.c_void_p),
                                                                                ctypes.c_void_p]
        self.ctypes, self.cuda = ctypes, cuda

    def run(self, prog, lens, mem, scratch, res, consts, T, N, s, m, max_steps):
        c = self.ctypes
        B, L = prog.shape[0], prog.shape[1]
        args = ([c.c_uint64(t.data_ptr()) for t in (prog, lens, mem, scratch, res, consts)]
                + [c.c_int32(N), c.c_int32(s), c.c_int32(m), c.c_int32(T), c.c_int64(max_steps), c.c_int32(B),
                   c.c_int32(L)])
        params = (c.c_void_p * len(args))(*[c.cast(c.byref(a), c.c_void_p) for a in args])
        threads = 64
        rc = self.cuda.cuLaunchKernel(self.fn, (B + threads - 1) // threads, 1, 1, threads, 1, 1, 0,
                                      c.c_void_p(torch.cuda.current_stream().cuda_stream), params, None)
        if rc:
            raise RuntimeError(f"cuLaunchKernel failed with CUDA error {rc}")


def _nvrtc_image(ptx: bool) -> bytes:
    """Compile a device cubin with NVRTC, or compute_80 PTX for driver JIT."""
    import ctypes
    nvrtc = ctypes.CDLL(_find_nvrtc())
    major, minor = torch.cuda.get_device_capability()
    arch = 'compute_80' if ptx else f'sm_{major}{minor}'
    prog = ctypes.c_void_p()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(prog), _DEVICE.encode(), b'strandedcpu.cu', 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    opts = [f'-arch={arch}'.encode(), b'-std=c++17', b'--device-int128']
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    if rc:
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC ({arch}): {log.value.decode(errors='replace')[-500:]}")
    kind = 'PTX' if ptx else 'CUBIN'
    size = ctypes.c_size_t()
    getattr(nvrtc, f'nvrtcGet{kind}Size')(prog, ctypes.byref(size))
    image = ctypes.create_string_buffer(size.value)
    getattr(nvrtc, f'nvrtcGet{kind}')(prog, image)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return image.raw


# Prebuilt kernel: one fatbin with machine code for each GPU generation and
# PTX for newer ones; the driver picks. build_fatbin() makes it; the JSON
# beside it records the source hash, which the loader checks before use.
FATBIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel.fatbin')
FATBIN_ARCHS = ('75', '80', '86', '89', '90', '100', '120')     # Turing .. Blackwell (RTX PRO 6000 = 120)
FATBIN_PTX = '75'


def source_sha256() -> str:
    import hashlib
    return hashlib.sha256(_DEVICE.encode()).hexdigest()


def _prebuilt_image() -> bytes:
    import json
    import os
    if not os.path.exists(FATBIN):
        raise RuntimeError(f"no prebuilt kernel at {FATBIN}")
    with open(FATBIN + '.json') as f:
        meta = json.load(f)
    if meta.get('source_sha256') != source_sha256():
        raise RuntimeError("prebuilt kernel is stale (kernel source changed; run python -m strandedcpu.cuda build-fatbin)")
    with open(FATBIN, 'rb') as f:
        return f.read()


def build_fatbin(nvcc: str = 'nvcc') -> Dict[str, object]:
    """Compile the kernel with nvcc for FATBIN_ARCHS plus PTX, into FATBIN
    and its metadata. Needs nvcc (12.8 or newer reaches sm_120), not a GPU."""
    import json
    import os
    import subprocess
    import tempfile
    flags = [f'-gencode=arch=compute_{a},code=sm_{a}' for a in FATBIN_ARCHS]
    flags.append(f'-gencode=arch=compute_{FATBIN_PTX},code=compute_{FATBIN_PTX}')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'strandedcpu_kernel.cu')
        with open(src, 'w') as f:
            f.write(_DEVICE)
        subprocess.run([nvcc, '-fatbin', '-O3', '-std=c++17', *flags, src, '-o', FATBIN], check=True)
    version = subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    meta = {'source_sha256': source_sha256(), 'archs': ['sm_' + a for a in FATBIN_ARCHS],
            'ptx': 'compute_' + FATBIN_PTX, 'nvcc': version, 'bytes': os.path.getsize(FATBIN)}
    with open(FATBIN + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
        f.write('\n')
    return meta


def _module():
    """Load the kernel once per process, recording attempts in BUILD.

    Try the prebuilt image, device NVRTC, NVRTC PTX, device nvcc, then nvcc PTX.
    STRANDEDCPU_CUDA_BUILD selects one of prebuilt, nvrtc, nvrtc-ptx, nvcc or
    nvcc-ptx instead. Each path uses the same device source.
    """
    global _ext
    if _ext is None:
        import os

        def nvcc(ptx: bool):
            from torch.utils.cpp_extension import load_inline
            return load_inline('strandedcpu_cuda_ext' + ('_ptx' if ptx else ''), cpp_sources=_CPP, cuda_sources=_SOURCE,
                               functions=['run'], verbose=False,
                               extra_cuda_cflags=['-O3'] + (['-gencode=arch=compute_80,code=compute_80'] if ptx else []))
        builds = {'prebuilt': lambda: _DriverKernel(_prebuilt_image()),
                  'nvrtc': lambda: _DriverKernel(_nvrtc_image(False)),
                  'nvrtc-ptx': lambda: _DriverKernel(_nvrtc_image(True)),
                  'nvcc': lambda: nvcc(False), 'nvcc-ptx': lambda: nvcc(True)}
        order = [os.environ['STRANDEDCPU_CUDA_BUILD']] if os.environ.get('STRANDEDCPU_CUDA_BUILD') else list(builds)
        for mode in order:
            try:
                _ext = builds[mode]()
                BUILD['mode'] = mode
                break
            except Exception as exc:
                BUILD['errors'].append(f"{mode}: {type(exc).__name__}: {str(exc)[-300:]}")
        else:
            raise RuntimeError("no CUDA build worked: " + " | ".join(BUILD['errors']))
    return _ext


def available() -> bool:
    return torch.cuda.is_available()


class Program:
    """Commands and checks packed once (reusable across runs): per
    instruction (op, b, c, check kind, 4 check slots)."""
    __slots__ = ('code', 'checks')

    def __init__(self, commands: Sequence[Sequence[int]], checks: Optional[Dict] = None):
        cmds = np.asarray(commands, dtype=np.int64).reshape(-1, 3)
        self.code = np.zeros((len(cmds), 8), np.int64)
        self.code[:, :3] = cmds
        self.checks = checks or {}
        if self.checks:
            idx = np.fromiter(self.checks.keys(), np.int64, len(self.checks))
            self.code[idx, 3] = [CHECK_CODE[v[0]] for v in self.checks.values()]
            self.code[idx, 4:] = [(list(v[1:]) + [0, 0, 0, 0])[:4] for v in self.checks.values()]


def _signed(v, N: int) -> int:
    """v wrapped to an N-bit two's complement word (as the interpreter stores it)."""
    v = int(v) & ((1 << N) - 1)
    return v - (1 << N) if v >> (N - 1) else v


def run_batch(s: int, m: int, N: int, programs: Sequence[Tuple], max_steps: int = 1_000_000,
              log_frac_bits: int = 2, device: str = 'cuda') -> List[Dict[str, object]]:
    """Execute a batch and return per-program memory, status, steps, fault and carry.

    An item is (memory, commands, checks) or (memory, Program). Final memory is a
    NumPy int64 array. status is halted, step_limit or fault; fault is an
    ExecutionFault object. steps includes HALT, but excludes an instruction
    stopped by a check. inexact records integer division remainders and fix rounding.
    """
    if not 2 <= N <= 64:
        raise ValueError("strandedcpu.cuda runs words of 2..64 bits")
    progs = [item[1] if isinstance(item[1], Program) else Program(item[1], item[2] if len(item) > 2 else None)
             for item in programs]
    if N > 32 and any(((p.code[:, 3] >= CHECK_CODE['fadd']) & (p.code[:, 3] <= CHECK_CODE['fneg'])).any() for p in progs):
        raise ValueError("strandedcpu.cuda checks binary fix arithmetic only up to N=32 (its products need 4N bits)")
    B = len(progs)
    L = max(1, max(len(p.code) for p in progs))
    code = np.zeros((B, L, 8), np.int64)
    lens = np.zeros(B, np.int64)
    mem = np.zeros((B, m), np.int64)
    for p, (prog, item) in enumerate(zip(progs, programs)):
        code[p, :len(prog.code)] = prog.code
        lens[p] = len(prog.code)
        if len(item[0]):
            try:
                mem[p, :len(item[0])] = np.asarray(item[0], dtype=np.int64)
            except OverflowError:
                mem[p, :len(item[0])] = [_signed(v, 64) for v in item[0]]
    if N < 63:
        half = 1 << (N - 1)
        mem = (mem + half) % (1 << N) - half
    elif N == 63:
        mem = np.asarray([[_signed(v, N) for v in row] for row in mem.tolist()], np.int64)
    table = logadd_table(log_frac_bits) or [0]
    c, K, h1, h2 = vrand_constants(N)
    consts = np.asarray([_signed(v, 64) for v in list(table) + list(c) + [K, h1, h2]], np.int64)
    dev = torch.device(device)
    to = lambda a: torch.from_numpy(a).to(dev, non_blocking=True)
    t_mem = to(mem)
    res = torch.empty(B, 8, dtype=torch.int64, device=dev)
    _module().run(to(code), to(lens), t_mem, torch.empty(B, 3 * m, dtype=torch.int64, device=dev),
                       res, to(consts), len(table), N, s, m, max_steps)
    mem_out = t_mem.cpu().numpy()
    res = res.cpu().tolist()
    status = [r[0] for r in res]
    calls, executed, fault_at = [r[1] for r in res], [r[2] for r in res], [r[3] for r in res]
    inexact, carry = [(r[4], r[5]) for r in res], [r[6] for r in res]
    results = []
    for p, prog in enumerate(progs):
        r = {'memory': mem_out[p], 'steps': calls[p], 'executed': executed[p], 'carry': carry[p],
             'inexact': {'int_div_remainder': inexact[p][0], 'fix_rounding': inexact[p][1]}, 'fault': None}
        if status[p] == 0:
            r['status'] = 'halted'
        elif status[p] == 1:
            r['status'] = 'step_limit'
        else:
            r['status'] = 'fault'
            r['steps'] = executed[p]
            if fault_at[p] in prog.checks:
                # the interpreter's own check, on the memory the kernel stopped with
                vm = ISAInterpreter(s, m, N, mem_out[p].tolist(), [], log_frac_bits)
                try:
                    vm._check(prog.checks[fault_at[p]])
                    r['fault'] = ExecutionFault('runtime_error', 'cuda and interpreter checks disagree')
                except ExecutionFault as exc:
                    r['fault'] = exc
            else:
                r['fault'] = ExecutionFault('runtime_error', 'vector length out of range')
        results.append(r)
    return results


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['build-fatbin']:
        print(build_fatbin())
    else:
        print("usage: python -m strandedcpu.cuda build-fatbin")
