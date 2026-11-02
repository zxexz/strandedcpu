"""Eight-layer analytical transformer for the StrandedCPU ISA.

Derived from LOOM, with scalar arithmetic, a second opcode page and vector
execution; see NOTICE for attribution. Each forward pass fetches and executes
one instruction on the bipolar state tensor.

Named opcodes 0..31 and the page-two patterns use explicit FFN routing.
The SUBLEQ operand path is gated by bit 5 (value 32) of the a field.
"""

import math
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from itertools import product
from typing import List, Tuple
from dataclasses import dataclass

from strandedcpu.subleq import (
    SUBLEQConfig, NeuralSUBLEQ, TransformerLayer,
    to_bipolar, from_bipolar, signed_to_bipolar, signed_from_bipolar
)

# Named scalar opcodes occupy 0..31; SUBLEQ reads are gated by a & 32.
OP_HALT = 0
OP_MOV = 1
OP_ADD = 2
OP_JMP = 3
OP_JZ = 4
OP_JNZ = 5
OP_INC = 6
OP_DEC = 7
OP_SHL = 8
OP_SHR = 9
OP_CMP = 10
OP_LOAD = 11
OP_AND = 12
OP_OR = 13
OP_XOR = 14
OP_SUB = 15
OP_FIND = 16
OP_SWAP = 17
OP_CMOV = 18
OP_MULACC = 19
OP_STORE = 20
# Opcodes 21-31: fixed-point / log-domain extension (see _bayes_decode_rows)
OP_MAX = 21      # mem[b] = max(mem[b], mem[c])
OP_LOGADD = 22   # mem[b] = max(mem[b], mem[c]) + T[|mem[b] - mem[c]|]
OP_XLAT = 23     # mem[b] = mem[c | mem[b]]
OP_SHIFT = 24    # mem[b] <<= mem[c] (arithmetic >> when mem[c] < 0)
OP_ADC = 25      # mem[b] += mem[c] + carry
OP_MULSTEP = 26  # one shift-add step of (mem[b]:mem[b^1]) * mem[c]
OP_DIVSTEP = 27  # one restoring-division step of (mem[b]:mem[b^1]) / mem[c]
OP_LFSR = 28     # mem[b] = (mem[b] >>> 1) ^ (mem[c] if LSB(mem[b]) else 0)
OP_CLZ = 29      # mem[b] = leading zeros of mem[c]
OP_ABS = 30      # mem[b] = |mem[c]|
OP_ARGMAX = 31   # mem[b] = argmax over mem[c : c + 2**mem[b]]

BAYES_OPCODES = (OP_MAX, OP_LOGADD, OP_XLAT, OP_SHIFT, OP_ADC, OP_MULSTEP,
                 OP_DIVSTEP, OP_LFSR, OP_CLZ, OP_ABS, OP_ARGMAX)

# Page 2: opcodes 64-95, present when n >= 128 (logn >= 7). An a field in
# 64..95 was never a working SUBLEQ address: SUBLEQ mode is detected from the
# 32 bit alone, so these patterns were dead. See _page2_decode_rows.
OP_JMPI = 64     # PC = s + m + mem[c] (instruction index; CALL/RET)
OP_FALIGN = 65   # float (mem[b]: exponent, mem[b^1]: mantissa): align to exponent mem[c]
OP_FNORM = 66    # float: normalize the mantissa, adjust the exponent
OP_VSETVL = 67   # vector length = 2**mem[b]
# Vector ops act on aligned blocks of VL words at b (and c); see _vector_* methods.
OP_VMOV = 68     # B[i] = C[i]
OP_VADD = 69     # B[i] += C[i]          (per-column carry out)
OP_VSUB = 70     # B[i] -= C[i]          (per-column carry = no borrow)
OP_VMAX = 71     # B[i] = max(B[i], C[i])
OP_VLOGADD = 72  # B[i] = logadd(B[i], C[i])
OP_VCADD = 73    # B[i] += carry of column C[i]   (multiword vectors)
OP_VFILL = 74    # B[i] = mem[c]
OP_VSHIFT = 75   # B[i] <<= mem[c]
OP_VXLAT = 76    # B[i] = mem[c | B[i]]  (per-column table lookup)
OP_VREV = 77     # B[i] = C[bitreverse(i)]
OP_VTRANS = 78   # B = transpose(C), a 2^(t/2) x 2^(t/2) block
OP_VRAND = 79    # B[i] = hash(mem[c], column)  (counter-based PRNG)
# Bit 16 selects the XOR partner B[i ^ mem[c]] as the second operand:
OP_VPERMX = 84   # B[i] = B[i ^ m]
OP_VADDX = 85    # B[i] += B[i ^ m]      (all-reduce stage)
OP_VBFLY = 86    # B[i], B[i^m] = B[i] + B[i^m], B[i] - B[i^m]  (m = 2^s; Walsh-Hadamard)
OP_VMAXX = 87    # B[i] = max(B[i], B[i ^ m])
OP_VLOGADDX = 88 # B[i] = logadd(B[i], B[i ^ m])

VECTOR_OPCODES = (OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX, OP_VLOGADD, OP_VCADD, OP_VFILL,
                  OP_VSHIFT, OP_VXLAT, OP_VREV, OP_VTRANS, OP_VRAND,
                  OP_VPERMX, OP_VADDX, OP_VBFLY, OP_VMAXX, OP_VLOGADDX)
PAGE2_OPCODES = (OP_JMPI, OP_FALIGN, OP_FNORM, OP_VSETVL) + VECTOR_OPCODES


def vrand_constants(N: int):
    """Deterministic constants for VRAND's per-column hash.

    u=seed+sum(c[p]*tag[p]), v=u^(u>>h1), w=K*v, out=w^(w>>h2), modulo 2**N.
    L4 through L7 perform the four stages. tests/test_vector.py checks correlations
    across neighboring columns and successive seeds.
    """
    import random as _random
    rng = _random.Random(0)
    mask = (1 << N) - 1
    c = [rng.randrange(1, 1 << N) | 1 for _ in range(N)]
    K = (round((1 << N) * 0.8242) | 1) & mask
    return c, K, max(1, (3 * N) // 8), N // 2


def logadd_table(frac_bits: int) -> List[int]:
    """Rounded LOGADD correction for log words with F fractional bits.

    T[d] = floor(2^F * log2(1 + 2^(-d / 2^F)) + 0.5).
    For encoded words a and b, the unwrapped result is max(a,b) + T[|a-b|].
    The table ends at the first d whose correction rounds to zero.
    """
    scale = 2 ** frac_bits
    table = []
    while True:
        v = math.floor(scale * math.log2(1.0 + 2.0 ** (-len(table) / scale)) + 0.5)
        if v == 0:
            return table
        table.append(v)


@dataclass
class Config(SUBLEQConfig):
    """State layout for scalar operations, indirect addressing and optional vectors."""

    log_frac_bits: int = 2  # LOGADD scale: a word v encodes log2(p) = v / 2**log_frac_bits
    vector_unit: bool = True  # build the page-2 vector unit (needs n >= 128)

    def __post_init__(self):
        super().__post_init__()
        # Buffer: buf_a (N), buf_b (N), buf_c (N), find_temp (N), load_temp (logn)
        # FIND and LOAD use separate temp regions because both attention heads fire
        # simultaneously in L3: shared temp causes cross-head interference.
        self.nrows_buffer = 3 * self.N + self.N + self.logn
        # Memory-relative tags supply FIND/ARGMAX results and VRAND column inputs.
        self.nrows_tag = self.N
        self.idx_tag = (self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad +
                       self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer)
        # ADD, SUB and ADC update scalar carry (0/1) at column zero.
        # Vector carry and the VL mask below also persist between instructions.
        self.nrows_flags = 1
        self.idx_carry = self.idx_tag + self.nrows_tag
        # Page 2 (opcodes 64-95) needs 7 address bits; the vector unit adds:
        #   vcarry  per-memory-column carry (VADD/VSUB/VADDX/VCADD), persistent
        #   memcol  static 1 at memory columns (the broadcast's query)
        #   vgate   column 0: 1 while a vector instruction broadcasts
        #   vmask   column 0, logn rows in pos_enc order: 1 for address bits
        #           inside a vector block (VSETVL), persistent
        self.page2 = self.logn >= 7
        self.vector = self.page2 and self.vector_unit
        if self.vector:
            self.idx_vcarry = self.idx_carry + 1
            self.idx_memcol = self.idx_carry + 2
            self.idx_vgate = self.idx_carry + 3
            self.idx_vmask = self.idx_carry + 4
            self.nrows_flags += 3 + self.logn
        self.d_model = (self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad +
                       self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer +
                       self.nrows_tag + self.nrows_flags + self.nrows_indicator)


def opcode_pattern(opcode: int, logn: int) -> torch.Tensor:
    """Convert opcode to bipolar pattern (MSB first)."""
    pattern = torch.zeros(logn)
    for i in range(logn):
        bit = (opcode >> (logn - 1 - i)) & 1
        pattern[i] = 1.0 if bit else -1.0
    return pattern


def create_bipolar_value(val: int, N: int) -> torch.Tensor:
    """Create bipolar representation of signed integer."""
    result = torch.zeros(N)
    if val < 0:
        val = val + (1 << N)
    for i in range(N):
        bit = (val >> (N - 1 - i)) & 1
        result[i] = 1.0 if bit else -1.0
    return result


# Sparse gated FFN rows (used by opcodes 21-31)
# A literal (row, w, c) contributes w*X[row] + c at column 0: +1 when it holds,
# <= -1 when it does not. A linear form (weights, const) is an integer-valued
# function of the column-0 state, e.g. the signed value of a buffered word.

def _half(row, bit):
    """Buffer bit read by a 50/50 attention tie: +/-0.5."""
    return (row, 2.0 if bit else -2.0, 0.0)


def _full(row, bit):
    """scratch_cmd bit: +/-1."""
    return (row, 1.0 if bit else -1.0, 0.0)


def _flag(row, bit):
    """0/1 flag row."""
    return (row, 2.0, -1.0) if bit else (row, -2.0, 1.0)


def _opcode_lits(addr_a, opcode, logn, ignore=0):
    """Literals matching opcode in the logn rows at addr_a; bits set in
    ignore are left out, so one row set serves an opcode family."""
    return [_full(addr_a + j, p > 0)
            for j, p in enumerate(opcode_pattern(opcode, logn).tolist())
            if not (ignore >> (logn - 1 - j)) & 1]


def _word(row, N, signed=False, upto=None, full=False):
    """Value of the word at rows row..row+N-1 (MSB first): half-scale bits
    (+/-0.5) by default, full-scale (+/-1) with full=True.

    upto=j keeps only LSB-index bits 0..j (a partial sum for bit extraction).
    """
    hi = N - 1 if upto is None else upto
    w = {row + N - 1 - p: float(2 ** p) for p in range(hi + 1)}
    if signed and hi == N - 1:
        w[row] = -float(2 ** (N - 1))
    const = 0.5 * sum(w.values())
    if full:
        w = {k: 0.5 * v for k, v in w.items()}
    return w, const


def _lin(*terms):
    """Linear combination of (coef, form) terms."""
    w, c = {}, 0.0
    for a, (fw, fc) in terms:
        for k, v in fw.items():
            w[k] = w.get(k, 0.0) + a * v
        c += a * fc
    return w, c


def _bits(value, N):
    """MSB-first bits of value mod 2^N."""
    return [(value >> (N - 1 - i)) & 1 for i in range(N)]


class _GatedRows:
    """Collect sparse FFN rows for scalar or memory-column execution.

    append_to(col0=True) targets column zero; col0=False targets memory columns.
    Biases suppress rows outside that region. Literal gates select opcodes and
    operand conditions within it.
    """

    def __init__(self):
        self.rows = []

    @staticmethod
    def _gate(lits, scale):
        w1, c = {}, 0.0
        for row, w, k in lits:
            w1[row] = w1.get(row, 0.0) + scale * w
            c += scale * k
        return w1, c - scale * len(lits)

    def gate(self, lits, out):
        """out (row -> delta) is added when every literal holds."""
        w1, c = self._gate(lits, 1.0)
        self.rows.append((w1, c + 1.0, dict(out)))

    def copy(self, src, lits, out):
        """X[src] (a 0/1 flag) times out is added when every literal holds."""
        w1, c = self._gate(lits, 2.0)
        w1[src] = w1.get(src, 0.0) + 1.0
        self.rows.append((w1, c, dict(out)))

    def step(self, x, t, lits, out):
        """Add out when integer x >= integer t and every literal holds.

        ReLU(x-t+1+G) - ReLU(x-t+G) is [x >= t] for integer x and t when the
        literal gate G is 0; a failed literal gives G <= -2L, and L exceeds
        |x - t| so both rows are off.
        """
        xw, xc = x
        L = sum(abs(v) for v in xw.values()) + abs(xc - t) + 2.0
        w1, c = self._gate(lits, L)
        for k, v in xw.items():
            w1[k] = w1.get(k, 0.0) + v
        base = xc - t + c
        self.rows.append((w1, base + 1.0, dict(out)))
        self.rows.append((dict(w1), base, {k: -v for k, v in out.items()}))

    def window(self, x, lo, hi, lits, out):
        """out is added when lo <= x <= hi."""
        self.step(x, lo, lits, out)
        self.step(x, hi + 1, lits, {k: -v for k, v in out.items()})

    def clear(self, row):
        """Subtract X[row] at column 0, whatever its value."""
        self.rows.append(({row: 1.0}, 0.0, {row: -1.0}))
        self.rows.append(({row: -1.0}, 0.0, {row: 1.0}))

    def append_to(self, layer, col0=True):
        """col0: rows act at column 0 only; otherwise at memory columns only.
        Every other column gets a bias far below any reachable
        pre-activation: literals alone cannot exclude a column whose rows hold
        values beyond +/-1 (column 0's branch flag, a memory column's
        mismatch count)."""
        cfg = layer.cfg
        k = len(self.rows)
        W1 = torch.zeros(k, cfg.d_model)
        b1 = torch.zeros(k, cfg.n)
        W2 = torch.zeros(cfg.d_model, k)
        for r, (w1, bias, out) in enumerate(self.rows):
            for col, v in w1.items():
                W1[r, col] = v
            for col, v in out.items():
                W2[col, r] = v
            # Bias other columns below the reachable pre-activation range.
            kill = bias - 64.0 * (W1[r].abs().sum().item() + abs(bias) + 1.0)
            if col0:
                b1[r, :] = kill
                b1[r, 0] = bias
            else:
                b1[r, :] = kill
                b1[r, cfg.s:cfg.s + cfg.m] = bias
        layer.W1 = torch.cat([layer.W1, W1])
        layer.b1 = torch.cat([layer.b1, b1])
        layer.W2 = torch.cat([layer.W2, W2], dim=1)


class Computer(NeuralSUBLEQ):
    """Build the eight analytical layers from Config."""
    
    def __init__(self, cfg: SUBLEQConfig):
        if not isinstance(cfg, Config):
            self.cfg = Config(s=cfg.s, m=cfg.m, n=cfg.n, N=cfg.N, lam=cfg.lam)
        else:
            self.cfg = cfg
        self.layers = self._build_layers()
    
    def _build_layers(self) -> List[TransformerLayer]:
        layers = []
        layers.append(self._init_read_inst())    # L1: Fetch instruction
        layers.append(self._init_read_mem())     # L2: Read memory + decode
        layers.append(self._init_indirect_read()) # L3: Indirect read + scratchpad correction
        layers.append(self._init_direct_subtract())  # L4: Direct subtract (single layer)
        layers.append(self._init_write_mem())    # L5: Write result (2-head for SWAP)
        layers.extend(self._init_cond_branch())  # L6-7: Branch (flag+PC+1 merged, select)
        layers.append(self._init_error_correction()) # L8: Error correction
        return layers
    
    def _init_read_inst(self) -> TransformerLayer:
        """Fetch an instruction through symmetric address attention.

        Q=K adds PC and position encoding into the same projection. Column zero
        has PC but zero position encoding; instruction columns have the reverse.
        The matching instruction and column zero tie, giving half-scale command
        bits in the buffer. The FFN restores full-scale scratch_cmd bits.
        """
        cfg = self.cfg
        size_inst = cfg.nrows_cmds
        size_pos_enc = cfg.logn
        
        logn = cfg.logn
        pair_ops = (OP_MULSTEP, OP_DIVSTEP) + ((OP_FALIGN, OP_FNORM) if cfg.page2 else ())
        layer = TransformerLayer(cfg, cfg.logn, 6 * size_inst + len(pair_ops) * logn, num_heads=1)

        # Match PC against position encoding; column zero has no position term.
        layer.Q[:, cfg.idx_pc:cfg.idx_pc + size_pos_enc] = torch.eye(size_pos_enc)
        layer.Q[:, cfg.idx_pos_enc:cfg.idx_pos_enc + size_pos_enc] = torch.eye(size_pos_enc)
        layer.K = layer.Q.clone()
        
        # Read command rows only; scratch_cmd would feed the previous instruction back.
        layer.V[cfg.idx_buffer:cfg.idx_buffer+size_inst, :cfg.nrows_cmds] = torch.eye(size_inst)

        # Restore command bits from the half-scale buffer, then clear the old scratch
        # command and buffer through residual cancellation.
        
        large_const = 100.0
        
        for i in range(size_inst):
            # Row 2i: buffer[i] > 0 case
            r = 2 * i
            layer.W1[r, cfg.idx_buffer + i] = 4.0  # 4*0.5 = 2 when buffer=0.5
            layer.W1[r, -1] = large_const  # indicator column
            layer.b1[r, :] = -large_const  # only fires in scratchpad columns
            
            # Row 2i+1: buffer[i] < 0 case  
            r = 2 * i + 1
            layer.W1[r, cfg.idx_buffer + i] = -4.0  # 4*0.5 = 2 when buffer=-0.5
            layer.W1[r, -1] = large_const
            layer.b1[r, :] = -large_const
        
        # Rows for clearing scratch_cmd (only in scratchpad column)
        base = 2 * size_inst
        for i in range(size_inst):
            # Row base+2i: fires when scratch_cmd[i] = +1 AND indicator=1
            layer.W1[base + 2*i, cfg.idx_scratch_cmd + i] = 1.0
            layer.W1[base + 2*i, -1] = large_const
            layer.b1[base + 2*i, :] = -large_const
            # Row base+2i+1: fires when scratch_cmd[i] = -1 AND indicator=1
            layer.W1[base + 2*i + 1, cfg.idx_scratch_cmd + i] = -1.0
            layer.W1[base + 2*i + 1, -1] = large_const
            layer.b1[base + 2*i + 1, :] = -large_const
        
        # Rows for clearing buffer (only in scratchpad column)
        buf_base = base + 2 * size_inst
        for i in range(size_inst):
            layer.W1[buf_base + 2*i, cfg.idx_buffer + i] = 1.0
            layer.W1[buf_base + 2*i, -1] = large_const
            layer.b1[buf_base + 2*i, :] = -large_const
            layer.W1[buf_base + 2*i + 1, cfg.idx_buffer + i] = -1.0
            layer.W1[buf_base + 2*i + 1, -1] = large_const
            layer.b1[buf_base + 2*i + 1, :] = -large_const
        
        # W2: set scratch_cmd from buffer sign
        # Half-scale command bits produce FFN output 2; weights of ±0.5 restore ±1.
        for i in range(size_inst):
            # scratch_cmd[i] += 1 when buffer positive (ff1 output = 2, so 0.5*2 = 1)
            layer.W2[cfg.idx_scratch_cmd + i, 2*i] = 0.5
            layer.W2[cfg.idx_scratch_cmd + i, 2*i + 1] = -0.5
            layer.W2[cfg.idx_scratch_cmd + i, base + 2*i] = -1.0
            layer.W2[cfg.idx_scratch_cmd + i, base + 2*i + 1] = 1.0
        
        for i in range(size_inst):
            layer.W2[cfg.idx_buffer + i, buf_base + 2*i] = -1.0
            layer.W2[cfg.idx_buffer + i, buf_base + 2*i + 1] = 1.0

        # Pair read for MULSTEP/DIVSTEP/FALIGN/FNORM: L2 head 1 queries addr_a + load_temp.
        # Write load_temp = pos(b XOR 1) - opcode so that head 1 reads the
        # partner word mem[b^1] instead of mem[opcode]. For each bit where the
        # partner address differs from the opcode bit, aux = 2 * partner bit.
        # Buffer holds 0.5 * instruction, so 2*buffer is +/-1 per bit.
        load_temp = cfg.idx_buffer + 4 * cfg.N
        r = 6 * size_inst
        for opcode in pair_ops:
            pat = opcode_pattern(opcode, logn)
            for i in range(logn):
                want = -pat[i].item()             # partner bit != opcode bit
                b_bit = -want if i == logn - 1 else want  # LSB of b is flipped
                for j in range(logn):
                    layer.W1[r, cfg.idx_buffer + j] = 2.0 * pat[j].item()
                layer.W1[r, cfg.idx_buffer + logn + i] = 2.0 * b_bit
                layer.W1[r, -1] = large_const
                layer.b1[r, :] = -large_const - logn
                layer.W2[load_temp + i, r] = 2.0 * want
                r += 1

        return layer
    
    def _init_read_mem(self) -> TransformerLayer:
        """Read three half-scale operand buffers and route them by opcode.

        For SUBLEQ, buf_a supplies scr_sub and buf_b supplies scr_min. Named
        opcodes replace or correct those scratch operands before L4 subtraction.
        Pair operations retarget the a read to b^1 through load_temp.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        
        buf_a = cfg.idx_buffer
        buf_b = cfg.idx_buffer + N
        buf_c = cfg.idx_buffer + 2 * N
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N
        addr_a = cfg.idx_scratch_cmd
        bit32 = logn - 6
        bit16 = logn - 5
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        bit32 = logn - 6
        bit16 = logn - 5
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        # Opcode bit positions are relative to logn, not fixed row offsets.
        bit32 = logn - 6  # 2^5 = 32
        bit16 = logn - 5  # 2^4 = 16
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        
        # Reserve rows for the hand-built opcode gates; sparse extensions append more.
        num_ffn = 80 * N
        layer = TransformerLayer(cfg, logn, num_ffn, num_heads=3)
        
        # ATTENTION HEADS
        
        # Head 1: Read mem[a] -> buf_a
        # load_temp retargets paired operations to mem[b^1]; see L1.
        layer.Q1[:, cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+logn] = torch.eye(logn)
        layer.Q1[:, cfg.idx_pos_enc:cfg.idx_pos_enc+logn] = torch.eye(logn)
        layer.Q1[:, cfg.idx_buffer+4*N:cfg.idx_buffer+4*N+logn] = torch.eye(logn)
        layer.K1 = layer.Q1.clone()
        layer.V1[buf_a:buf_a+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        layer.V1[buf_a:buf_a+N, scr_sub:scr_sub+N] = torch.eye(N)
        
        # Head 2: Read mem[b] -> buf_b
        layer.Q2[:, cfg.idx_scratch_cmd+logn:cfg.idx_scratch_cmd+2*logn] = torch.eye(logn)
        layer.Q2[:, cfg.idx_pos_enc:cfg.idx_pos_enc+logn] = torch.eye(logn)
        layer.K2 = layer.Q2.clone()
        layer.V2[buf_b:buf_b+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        layer.V2[buf_b:buf_b+N, scr_min:scr_min+N] = torch.eye(N)
        
        # Head 3: Read mem[c] -> buf_c
        layer.Q3[:, cfg.idx_scratch_cmd+2*logn:cfg.idx_scratch_cmd+3*logn] = torch.eye(logn)
        layer.Q3[:, cfg.idx_pos_enc:cfg.idx_pos_enc+logn] = torch.eye(logn)
        layer.K3 = layer.Q3.clone()
        layer.V3[buf_c:buf_c+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        
        # FFN: Gated copy and opcode routing
        
        large = 100.0
        row = 0
        
        # SUBLEQ copies buf_a into scr_sub only when opcode bit 32 is set.
        # Clear the read buffer afterward; buf_b supplies scr_min for every opcode.

        for i in range(N):
            r = row + 4*i
            # Gate on opcode bit 32 and the positive half-scale operand bit.
            layer.W1[r, mode_bit0] = 1.0      # Full weight for mode (bit 32)
            layer.W1[r, buf_a + i] = 1.0
            layer.W1[r, -1] = large
            layer.b1[r, :] = -large           # Cancel indicator for non-scratchpad
            layer.W2[scr_sub + i, r] = 2.0

            r = row + 4*i + 1
            layer.W1[r, mode_bit0] = 1.0
            layer.W1[r, buf_a + i] = -1.0
            layer.W1[r, -1] = large
            layer.b1[r, :] = -large
            layer.W2[scr_sub + i, r] = -2.0
            
            r = row + 4*i + 2
            layer.W1[r, buf_a + i] = 1.0
            layer.W2[buf_a + i, r] = -1.0
            
            r = row + 4*i + 3
            layer.W1[r, buf_a + i] = -1.0
            layer.W2[buf_a + i, r] = 1.0
        row += 4*N
        
        # Always copy buf_b -> scr_min (valid for all modes)
        for i in range(N):
            r = row + 4*i
            layer.W1[r, scr_min + i] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.W1[r, -1] = large
            layer.b1[r, :] = -large
            layer.W2[scr_min + i, r] = 1.0
            
            r = row + 4*i + 1
            layer.W1[r, scr_min + i] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.W1[r, -1] = large
            layer.b1[r, :] = -large
            layer.W2[scr_min + i, r] = -1.0
            
            r = row + 4*i + 2
            layer.W1[r, buf_b + i] = 1.0
            layer.W2[buf_b + i, r] = -1.0
            
            r = row + 4*i + 3
            layer.W1[r, buf_b + i] = -1.0
            layer.W2[buf_b + i, r] = 1.0
        row += 4*N

        # Clear buf_c at MEMORY columns only
        # Keep buf_c at scratchpad col 0 (= 0.5*mem[c] from attention) for L5 Head 2.
        # At memory cols (indicator=0): ReLU(±buf_c - large*0) fires → clears buf_c.
        # At scratchpad cols (indicator=1): ReLU(±buf_c - large) ≈ 0 → no effect.
        for i in range(N):
            r = row + 2*i
            layer.W1[r, buf_c + i] = 1.0
            layer.W1[r, -1] = -large
            layer.W2[buf_c + i, r] = -1.0
            r = row + 2*i + 1
            layer.W1[r, buf_c + i] = -1.0
            layer.W1[r, -1] = -large
            layer.W2[buf_c + i, r] = 1.0
        row += 2*N
        
        # EXTENDED MODE: Set values directly based on opcode
        # Named opcode rows route operands directly into the shared subtraction path.
        # INC/DEC subtract -1/+1; branches subtract zero; MOV replaces scr_min.
        # ADD negates c into scr_sub, while SUB copies c there.

        # INC (opcode 6 = 000110): scr_sub = -1 (all +1s in bipolar)
        # Gate fires with value 0.5 when opcode matches, so scale W2 by 2
        inc_pat = opcode_pattern(OP_INC, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = inc_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = 2.0  # 0.5 * 2 = +1
        row += 1
        
        # DEC (opcode 7 = 000111): scr_sub = +1 ([-1]*7 + [+1])
        dec_pat = opcode_pattern(OP_DEC, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = dec_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N - 1):
            layer.W2[scr_sub + i, r] = -2.0
        layer.W2[scr_sub + N - 1, r] = 2.0
        row += 1
        
        # JZ (opcode 4): scr_sub = 0 (all -1s)
        jz_pat = opcode_pattern(OP_JZ, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = jz_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # JNZ (opcode 5): scr_sub = 0 (all -1s)
        jnz_pat = opcode_pattern(OP_JNZ, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = jnz_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # JMP (opcode 3): scr_sub = 0 (all -1s)
        jmp_pat = opcode_pattern(OP_JMP, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = jmp_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # HALT (opcode 0): scr_sub = 0 (all -1s)
        halt_pat = opcode_pattern(OP_HALT, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = halt_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # MOV (opcode 1): scr_sub = 0 (all -1s), scr_min = buf_c
        mov_pat = opcode_pattern(OP_MOV, logn)
        
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = mov_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        for i in range(N):
            # Case: buf_c = +, buf_b = - -> add +2
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = mov_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0   # Scaled for ±0.5
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
            
            # Case: buf_c = -, buf_b = + -> add -2
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = mov_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
        row += 2*N
        
        # ADD (opcode 2): scr_sub = -buf_c (TWO'S COMPLEMENT)
        add_pat = opcode_pattern(OP_ADD, logn)
        
        # ADD uses L4 subtraction, so prepare -c as ~c + 1.
        # The carry reaches bit i exactly when every lower bit of c is zero.
        # Separate gated rows flip those bits after the complement is written.
        
        # First pass: set scr_sub to flipped buf_c (one's complement)
        for i in range(N):
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = -1.0
            
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = 1.0
        row += 2*N
        
        # Second pass: add 1 using carry propagation.
        # Carry-in for bit i occurs when all lower bits of buf_c are -1 (i.e., 0 in binary),
        # because f = ~buf_c has trailing +1s in those positions.
        #   buf_c[i] = +1 (f_i = -1) -> add +2 (flip to +1)
        #   buf_c[i] = -1 (f_i = +1) -> add -2 (flip to -1)
        for i in range(N - 1, -1, -1):
            lower_bits = N - 1 - i
            # Case 1: buf_c[i] = +1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # Require lower bits = -1
            layer.W1[r, buf_c + i] = 2.0       # Require buf_c[i] = +1
            layer.b1[r, :] = -(logn + lower_bits + 1) + 1.0
            layer.W2[scr_sub + i, r] = 2.0
            row += 1
            
            # Case 2: buf_c[i] = -1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # Require lower bits = -1
            layer.W1[r, buf_c + i] = -2.0      # Require buf_c[i] = -1
            layer.b1[r, :] = -(logn + lower_bits + 1) + 1.0
            layer.W2[scr_sub + i, r] = -2.0
            row += 1
        
        # SUB (opcode 15): scr_sub = buf_c
        sub_pat = opcode_pattern(OP_SUB, logn)
        
        for i in range(N):
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = sub_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = 1.0
            
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = sub_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = -1.0
        row += 2*N

        # SHL (opcode 8): Shift Left by 1
        # mem[b] <<= 1. Bit ordering: index 0 = MSB, index N-1 = LSB.
        # SHL: result[i] = input[i+1] for i=0..N-2, result[N-1] = -1 (zero bit)
        # Route zero into scr_sub and the shifted b word into scr_min.
        # delta[i] = buf_b[i+1] - buf_b[i] for i < N-1
        # delta[N-1] = -1 - buf_b[N-1]
        shl_pat = opcode_pattern(OP_SHL, logn)

        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = shl_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Correction: for i < N-1, add (buf_b[i+1] - buf_b[i]) to scr_min[i]
        for i in range(N - 1):
            # Case: buf_b[i+1]=+1, buf_b[i]=-1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shl_pat[j].item()
            layer.W1[r, buf_b + i + 1] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
            row += 1

            # Case: buf_b[i+1]=-1, buf_b[i]=+1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shl_pat[j].item()
            layer.W1[r, buf_b + i + 1] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
            row += 1

        # For LSB (i=N-1): set to -1 (zero). Correction = -1 - buf_b[N-1].
        # If buf_b[N-1] = +1: subtract 2. If buf_b[N-1] = -1: no change.
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = shl_pat[j].item()
        layer.W1[r, buf_b + N - 1] = 2.0
        layer.b1[r, 0] = -logn
        layer.W2[scr_min + N - 1, r] = -2.0
        row += 1

        # SHR (opcode 9): Arithmetic Shift Right by 1
        # mem[b] >>= 1 (arithmetic, preserves sign).
        # SHR: result[i] = input[i-1] for i=1..N-1, result[0] = input[0] (sign preserved)
        # scr_min has buf_b. Correct: scr_min[i] should become buf_b[i-1] for i>0.
        # Correction: delta[i] = buf_b[i-1] - buf_b[i] for i > 0
        # delta[0] = 0 (sign bit stays)
        shr_pat = opcode_pattern(OP_SHR, logn)

        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = shr_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Correction: for i > 0, add (buf_b[i-1] - buf_b[i]) to scr_min[i]
        for i in range(1, N):
            # Case: buf_b[i-1]=+1, buf_b[i]=-1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shr_pat[j].item()
            layer.W1[r, buf_b + i - 1] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
            row += 1

            # Case: buf_b[i-1]=-1, buf_b[i]=+1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shr_pat[j].item()
            layer.W1[r, buf_b + i - 1] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
            row += 1

        # CMP (opcode 10 = BLTZ): scr_sub = 0 (all -1s)
        # CMP(b, c): if mem[b] < 0, PC = c; else PC = PC+1.  mem[b] unchanged.
        # scr_sub = 0 means subtract phase is no-op (result = scr_min = buf_b = original mem[b]).
        cmp_pat = opcode_pattern(OP_CMP, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = cmp_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # LOAD (opcode 11): mem[b] = mem[mem[c]]
        # Instruction: (OP_LOAD, target_addr, pointer_addr)
        # Head 2 reads mem[b] -> buf_b (target, overwritten)
        # Head 3 reads mem[c] -> buf_c (pointer value)
        # Indirect read layer uses pointer to read mem[pointer] -> scr_min
        # LOAD FFN rows need indicator gating to restrict
        # firing to scratchpad columns only (indicator=1). Without this,
        # memory columns' cmd rows can accidentally match the opcode pattern
        # (from L1/L2 attention spillover), causing temp to be set in memory
        # columns and corrupting L3's indirect read attention.
        load_pat = opcode_pattern(OP_LOAD, logn)
        indicator = cfg.d_model - 1  # indicator row: 1 for scratchpad, 0 for memory

        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = load_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Zero scr_min for LOAD (undo "always copy" which sets scr_min = 2*buf_b).
        # buf_b = ±0.5 (from 50/50 attention), always-copy gives scr_min = ±1.0.
        # Need ∓1.0 correction to zero it.
        # ReLU output is 0.5 when gated row fires, so W2 = ±2.0 gives ±1.0.
        find_temp = cfg.idx_buffer + 3 * N
        load_temp = cfg.idx_buffer + 3 * N + N
        for i in range(N):
            # Row A: LOAD AND buf_b[i] = +1 → subtract 1 from scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = -2.0
            row += 1

            # Row B: LOAD AND buf_b[i] = -1 → add 1 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = 2.0
            row += 1

        # Convert buf_c (N-bit pointer value) → pointer_pos_enc (logn bits) in load_temp.
        # L3 uses Q=K symmetric attention reading load_temp + pos_enc to select
        # column s+pointer_value. The conversion maps:
        #   s = 2^k, column s+p has pos_enc bits:
        #     bit_position > k: constant 0 → bipolar -1
        #     bit_position == k: constant 1 → bipolar +1
        #     bit_position < k: bit of p → copy from buf_c[N-1-bit_position]
        # Indicator gating ensures only scratchpad columns fire.
        k = int(np.log2(cfg.s))  # s = 2^k

        # Row 1: Set constant bits in load_temp (all constants in one row).
        # For pos_enc index i: bit_position = logn-1-i
        #   bit_position > k → load_temp[i] = -1 (W2 = -2.0)
        #   bit_position == k → load_temp[i] = +1 (W2 = +2.0)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = load_pat[j].item()
        layer.W1[r, indicator] = float(logn)
        layer.b1[r, :] = -2.0 * logn + 0.5
        # ReLU output when LOAD+indicator: logn + logn - 2*logn + 0.5 = 0.5
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position > k:
                layer.W2[load_temp + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            elif bit_position == k:
                layer.W2[load_temp + i, r] = 2.0   # 0.5 * 2.0 = +1.0
        row += 1

        # Copy bits: for bit_position < k, copy buf_c[N-1-bit_position] → load_temp[i]
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position >= k:
                continue  # handled by constant row above
            src_idx = N - 1 - bit_position  # buf_c index for this bit of p

            # Row A: LOAD AND indicator AND buf_c[src] > 0 → load_temp[i] = +1
            # buf_c = ±0.5 from 50/50 attention; ReLU output = 0.5 when fires
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_c + src_idx] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[load_temp + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

            # Row B: LOAD AND indicator AND buf_c[src] < 0 → load_temp[i] = -1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_c + src_idx] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[load_temp + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            row += 1

        # FIND (opcode 16): Content-addressable memory search
        # FIND(dest, value_addr): mem[dest] = index i where mem[i] == mem[value_addr]
        # Head 2 reads mem[b] -> buf_b (destination, overwritten)
        # Head 3 reads mem[c] -> buf_c (search value)
        # L3 head 2 does content-addressable search using temp as query.
        find_pat = opcode_pattern(OP_FIND, logn)

        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = find_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Zero scr_min for FIND (undo "always copy" buf_b → scr_min)
        for i in range(N):
            # Row A: FIND AND buf_b[i] = +1 → subtract 1 from scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = -2.0
            row += 1

            # Row B: FIND AND buf_b[i] = -1 → add 1 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = 2.0
            row += 1

        # Copy buf_c (search value, N bits) → find_temp[0:N] with indicator gating.
        # Scale half-width buffered bits to full-width FIND query bits.
        for i in range(N):
            # Row A: FIND AND indicator AND buf_c[i] > 0 → find_temp[i] = +1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[find_temp + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

            # Row B: FIND AND indicator AND buf_c[i] < 0 → find_temp[i] = -1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[find_temp + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            row += 1

        # CMOV (opcode 18): Conditional Move
        # CMOV(b, c): if mem[b] < 0, mem[b] = mem[c]; else mem[b] unchanged.
        # scr_min already contains b. A positive bipolar sign bit (buf_b[0]=+0.5)
        # enables corrections that replace it with c.
        cmov_pat = opcode_pattern(OP_CMOV, logn)

        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = cmov_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Conditional correction: replace scr_min bits when buf_b[0] = +1 (negative)
        # Special handling for i=0: buf_b+0 == buf_b (sign bit position), so the sign
        # gate weight and destination weight occupy the same tensor position. A
        # combined weight of 4.0 accounts for both conditions in Case B at i=0.
        # Case A for i=0 is impossible (would need sign=+ AND dest=-, contradiction).

        # i=0 (sign bit): only Case B needed (buf_c[0]=-, buf_b[0]=+)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = cmov_pat[j].item()
        layer.W1[r, buf_b] = 4.0       # combined: 2.0(sign) + 2.0(dest i=0)
        layer.W1[r, buf_c] = -2.0      # source bit 0 (want negative)
        layer.b1[r, 0] = -logn - 2.5
        layer.W2[scr_min, r] = -4.0    # 0.5 * -4.0 = -2.0
        row += 1

        # The other sign-bit case already has the desired value; no row is needed.

        # i>0: normal 3-condition gating (sign + buf_c + buf_b at separate positions)
        for i in range(1, N):
            # Case A: sign=neg AND buf_c[i]=+ AND buf_b[i]=- → add +2 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = cmov_pat[j].item()
            layer.W1[r, buf_b] = 2.0       # sign bit (buf_b[0])
            layer.W1[r, buf_c + i] = 2.0   # source bit
            layer.W1[r, buf_b + i] = -2.0  # dest bit (want negative)
            layer.b1[r, 0] = -logn - 2.5
            layer.W2[scr_min + i, r] = 4.0  # 0.5 * 4.0 = +2.0
            row += 1

            # Case B: sign=neg AND buf_c[i]=- AND buf_b[i]=+ → add -2 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = cmov_pat[j].item()
            layer.W1[r, buf_b] = 2.0       # sign bit (buf_b[0])
            layer.W1[r, buf_c + i] = -2.0  # source bit (want negative)
            layer.W1[r, buf_b + i] = 2.0   # dest bit
            layer.b1[r, 0] = -logn - 2.5
            layer.W2[scr_min + i, r] = -4.0  # 0.5 * -4.0 = -2.0
            row += 1

        # SWAP (opcode 17): Swap mem[b] and mem[c]
        # Put mem[c] in scr_min for head one and half-scale mem[b] in buf_c
        # for head two. Other scalar opcodes leave head two as an identity write.
        swap_pat = opcode_pattern(OP_SWAP, logn)

        # Use b1[:] to gate off ALL non-col-0 columns (addr_a=0 at memory cols).
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = swap_pat[j].item()
        layer.b1[r, :] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # MOV correction: scr_min = buf_c (same pattern as MOV opcode)
        # b1[:] ensures this only fires at col 0 (where opcode pattern lives).
        for i in range(N):
            # Case: buf_c = +, buf_b = - → add +2
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0

            # Case: buf_c = -, buf_b = + → add -2
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
        row += 2*N

        # MOV correction: buf_c ← buf_b (SWAP-gated).
        # After L2 attention, buf_c[col0] = 0.5*mem[c]. For SWAP, replace with
        # 0.5*mem[b] so L5 Head 2 writes mem[b] to addr_c.
        # b1[:] prevents spurious firing at memory columns.
        for i in range(N):
            # Case: buf_b > buf_c → add +1 to buf_c
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.W1[r, buf_c + i] = -2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[buf_c + i, r] = 1.0

            # Case: buf_b < buf_c → add -1 to buf_c
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[buf_c + i, r] = -1.0
        row += 2*N

        # MULACC (opcode 19): Multiply-Accumulate Step
        # MULACC(b, c): mem[b] = (mem[b] << 1) + (mem[c] if MSB(mem[b]) else 0)
        # Used for shift-and-add multiplication.
        # Implementation:
        #   scr_min = shifted(buf_b)          (same SHL correction as opcode 8)
        #   scr_sub = -buf_c when buf_b[0]=+1 (MSB set), else 0
        #   L4 computes: scr_min - scr_sub
        #     MSB set:   shifted(buf_b) - (-buf_c) = shifted(buf_b) + buf_c
        #     MSB clear: shifted(buf_b) - 0        = shifted(buf_b)
        mulacc_pat = opcode_pattern(OP_MULACC, logn)

        # Part 1: Clear scr_sub to 0 (all -1s)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = mulacc_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Part 2: SHL correction for scr_min (same pattern as SHL opcode)
        # For i < N-1: delta = buf_b[i+1] - buf_b[i]
        for i in range(N - 1):
            # Case: buf_b[i+1]=+1, buf_b[i]=-1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b + i + 1] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
            row += 1

            # Case: buf_b[i+1]=-1, buf_b[i]=+1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b + i + 1] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
            row += 1

        # LSB (i=N-1): set to -1 (zero). If buf_b[N-1]=+1, subtract 2.
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = mulacc_pat[j].item()
        layer.W1[r, buf_b + N - 1] = 2.0
        layer.b1[r, 0] = -logn
        layer.W2[scr_min + N - 1, r] = -2.0
        row += 1

        # Part 3: Conditional two's complement negation of buf_c into scr_sub
        # Only fires when buf_b[0] = +1 (MSB of original value was set).
        # Same structure as ADD's negation but with extra buf_b[0] gate.

        # Pass 1: One's complement (flip buf_c bits) into scr_sub
        # scr_sub was cleared to -1 (0). ~buf_c[i]:
        #   buf_c[i]=+1 -> ~=-1, already -1, no change
        #   buf_c[i]=-1 -> ~=+1, need +2 correction
        for i in range(N):
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b] = 2.0        # gate on MSB
            layer.W1[r, buf_c + i] = -2.0   # fire when buf_c[i] = -1
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_sub + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

        # Pass 2: Add 1 carry chain (same logic as ADD's carry chain)
        # For each bit i: flip if all lower bits of buf_c are -1 (trailing zeros
        # in original, trailing ones in ~buf_c, carry propagates through them).
        for i in range(N - 1, -1, -1):
            lower_bits = N - 1 - i
            # Case 1: buf_c[i] = +1 (f[i]=-1, carry flips to +1) -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b] = 2.0        # gate on MSB
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # require lower bits = -1
            layer.W1[r, buf_c + i] = 2.0       # buf_c[i] = +1
            layer.b1[r, :] = -(logn + lower_bits + 2) + 1.0
            layer.W2[scr_sub + i, r] = 2.0
            row += 1

            # Case 2: buf_c[i] = -1 (f[i]=+1, carry flips to -1) -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b] = 2.0        # gate on MSB
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # require lower bits = -1
            layer.W1[r, buf_c + i] = -2.0      # buf_c[i] = -1
            layer.b1[r, :] = -(logn + lower_bits + 2) + 1.0
            layer.W2[scr_sub + i, r] = -2.0
            row += 1

        # AND (opcode 12): Bitwise AND
        # Starting from b, AND clears a set bit only where c is clear.
        and_pat = opcode_pattern(OP_AND, logn)
        
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = and_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # Correction: subtract 2 when buf_b=+1 and buf_c=-1
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = and_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0   # +1 when buf_b = +0.5
            layer.W1[r, buf_c + i] = -2.0  # +1 when buf_c = -0.5
            layer.b1[r, 0] = -logn - 1.0   # Need opcode + buf_b>0 + buf_c<0
            layer.W2[scr_min + i, r] = -2.0  # Flip +1 to -1
        row += N
        
        # OR (opcode 13): Bitwise OR
        # Starting from b, OR sets a clear bit only where c is set.
        or_pat = opcode_pattern(OP_OR, logn)
        
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = or_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # Correction: add 2 when buf_b=-1 and buf_c=+1
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = or_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0  # +1 when buf_b = -0.5
            layer.W1[r, buf_c + i] = 2.0   # +1 when buf_c = +0.5
            layer.b1[r, 0] = -logn - 1.0   # Need opcode + buf_b<0 + buf_c>0
            layer.W2[scr_min + i, r] = 2.0  # Flip -1 to +1
        row += N
        
        # XOR (opcode 14): Bitwise XOR
        # Starting from b, XOR flips a bit exactly where c is set.
        xor_pat = opcode_pattern(OP_XOR, logn)
        
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = xor_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # Case 1: buf_b=+1, buf_c=+1 -> subtract 2 (XOR gives -1)
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = xor_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
        row += N
        
        # Case 2: buf_b=-1, buf_c=+1 -> add 2 (XOR gives +1)
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = xor_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
        row += N
        
        # Note: b2 bias NOT used - it would affect SUBLEQ mode too
        # Extended opcodes set values from 0 baseline using ±1 contributions

        # STORE (opcode 20): mem[mem[c]] = mem[b]
        # Preserve mem[b] as the scalar result and replace addr_b with the
        # pointer column. L5 then writes through the pointer using its usual head.
        store_pat = opcode_pattern(OP_STORE, logn)
        addr_b_start = cfg.idx_scratch_cmd + logn  # addr_b region in scratch_cmd

        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = store_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Clear old addr_b bits from scratch_cmd (at scratchpad cols only).
        # addr_b bits are FULL scale ±1 (from L1 instruction decode), not ±0.5.
        # With W1 weight 2.0, contribution = 2.0*1.0 = 2.0 (not 1.0 like buf_c).
        # Bias must be -(2*logn + 3.5) to get ReLU output = 0.5 when match.
        for i in range(logn):
            # Row A: STORE AND indicator AND addr_b[i] = +1 → subtract
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, addr_b_start + i] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 3.5
            layer.W2[addr_b_start + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            row += 1

            # Row B: STORE AND indicator AND addr_b[i] = -1 → add
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, addr_b_start + i] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 3.5
            layer.W2[addr_b_start + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

        # Write new addr_b = position encoding of column (s + pointer_value).
        # Same bit layout as LOAD's load_temp conversion, but targeting addr_b.
        k = int(np.log2(cfg.s))  # s = 2^k

        # Constants row: set bits for positions > k to -1, position k to +1
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = store_pat[j].item()
        layer.W1[r, indicator] = float(logn)
        layer.b1[r, :] = -2.0 * logn + 0.5
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position > k:
                layer.W2[addr_b_start + i, r] = -2.0
            elif bit_position == k:
                layer.W2[addr_b_start + i, r] = 2.0
        row += 1

        # Copy variable bits from buf_c to addr_b
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position >= k:
                continue
            src_idx = N - 1 - bit_position

            # Row A: STORE AND indicator AND buf_c[src] > 0 → addr_b[i] = +1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, buf_c + src_idx] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[addr_b_start + i, r] = 2.0
            row += 1

            # Row B: STORE AND indicator AND buf_c[src] < 0 → addr_b[i] = -1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, buf_c + src_idx] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[addr_b_start + i, r] = -2.0
            row += 1

        # Opcodes 21-31 and the carry flag
        self._bayes_decode_rows().append_to(layer)
        if cfg.page2:
            self._page2_decode_rows().append_to(layer)

        return layer

    def _init_write_mem(self) -> TransformerLayer:
        """Write scalar results through two address-matching heads.

        Head one writes scr_min at addr_b. Head two uses buf_c for SWAP's other
        value or a paired operation's second word at addr_c. Otherwise its matched
        value reproduces the existing memory word, so its FFN delta is zero.
        Vector instructions redirect these scalar writes and use a separate gather head.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        buf_a = cfg.idx_buffer
        find_temp = cfg.idx_buffer + 3 * N

        # 4N for Head 1 write + 4N for Head 2 write + 2N for buf_c clear = 10N
        num_ffn = 10 * N
        layer = TransformerLayer(cfg, logn, num_ffn, num_heads=3 if cfg.vector else 2)
        large = 100.0

        # Head 1: Match addr_b (same as base _init_write_mem)
        layer.Q1[:, cfg.idx_scratch_cmd + logn:cfg.idx_scratch_cmd + 2*logn] = torch.eye(logn)
        layer.Q1[:, cfg.idx_pos_enc:cfg.idx_pos_enc + logn] = torch.eye(logn)
        layer.K1 = layer.Q1.clone()

        # V1: memory + scr_min → buf_a
        layer.V1[buf_a:buf_a+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        layer.V1[buf_a:buf_a+N, cfg.idx_scratchpad+N:cfg.idx_scratchpad+2*N] = torch.eye(N)

        # Head 2: Match addr_c (SWAP second write)
        layer.Q2[:, cfg.idx_scratch_cmd + 2*logn:cfg.idx_scratch_cmd + 3*logn] = torch.eye(logn)
        layer.Q2[:, cfg.idx_pos_enc:cfg.idx_pos_enc + logn] = torch.eye(logn)
        layer.K2 = layer.Q2.clone()

        # V2: 2*buf_c + memory → find_temp
        # buf_c at col 0 = 0.5*mem[c] (non-SWAP) or 0.5*mem[b] (SWAP).
        # buf_c at memory cols = 0 (cleared by L2 memory-only clear).
        # At matched addr_c: 50/50 gives find_temp = 0.5*(2*buf_c[0]+0) + 0.5*(0+mem[c])
        #   non-SWAP: = 0.5*mem[c] + 0.5*mem[c] = mem[c] → FFN: 0 (no-op)
        #   SWAP:     = 0.5*mem[b] + 0.5*mem[c] → FFN: mem[b]-mem[c] (writes mem[b])
        buf_c = cfg.idx_buffer + 2 * N
        layer.V2[find_temp:find_temp+N, buf_c:buf_c+N] = 2.0 * torch.eye(N)
        layer.V2[find_temp:find_temp+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)

        # FFN Section 1: Head 1 write (buf_a → memory at addr_b)
        # Same logic as base _init_write_mem
        layer.W1[:N, cfg.idx_memory:cfg.idx_memory+N] = -2 * torch.eye(N)
        layer.W1[:N, buf_a:buf_a+N] = 2 * torch.eye(N)
        layer.W1[:N, -1] = -large
        layer.W1[N:2*N, cfg.idx_memory:cfg.idx_memory+N] = 2 * torch.eye(N)
        layer.W1[N:2*N, buf_a:buf_a+N] = -2 * torch.eye(N)
        layer.W1[N:2*N, -1] = -large
        # buf_a clear
        layer.W1[2*N:3*N, buf_a:buf_a+N] = torch.eye(N)
        layer.W1[3*N:4*N, buf_a:buf_a+N] = -torch.eye(N)

        layer.W2[cfg.idx_memory:cfg.idx_memory+N, :N] = torch.eye(N)
        layer.W2[cfg.idx_memory:cfg.idx_memory+N, N:2*N] = -torch.eye(N)
        layer.W2[buf_a:buf_a+N, 2*N:3*N] = -torch.eye(N)
        layer.W2[buf_a:buf_a+N, 3*N:4*N] = torch.eye(N)

        # FFN Section 2: Head 2 write (find_temp → memory at addr_c)
        off = 4 * N
        layer.W1[off:off+N, cfg.idx_memory:cfg.idx_memory+N] = -2 * torch.eye(N)
        layer.W1[off:off+N, find_temp:find_temp+N] = 2 * torch.eye(N)
        layer.W1[off:off+N, -1] = -large
        layer.W1[off+N:off+2*N, cfg.idx_memory:cfg.idx_memory+N] = 2 * torch.eye(N)
        layer.W1[off+N:off+2*N, find_temp:find_temp+N] = -2 * torch.eye(N)
        layer.W1[off+N:off+2*N, -1] = -large
        # find_temp clear
        layer.W1[off+2*N:off+3*N, find_temp:find_temp+N] = torch.eye(N)
        layer.W1[off+3*N:off+4*N, find_temp:find_temp+N] = -torch.eye(N)

        layer.W2[cfg.idx_memory:cfg.idx_memory+N, off:off+N] = torch.eye(N)
        layer.W2[cfg.idx_memory:cfg.idx_memory+N, off+N:off+2*N] = -torch.eye(N)
        layer.W2[find_temp:find_temp+N, off+2*N:off+3*N] = -torch.eye(N)
        layer.W2[find_temp:find_temp+N, off+3*N:off+4*N] = torch.eye(N)

        # FFN Section 3: Clear buf_c (prevent accumulation across cycles)
        off2 = 8 * N
        layer.W1[off2:off2+N, buf_c:buf_c+N] = torch.eye(N)
        layer.W1[off2+N:off2+2*N, buf_c:buf_c+N] = -torch.eye(N)

        layer.W2[buf_c:buf_c+N, off2:off2+N] = -torch.eye(N)
        layer.W2[buf_c:buf_c+N, off2+N:off2+2*N] = torch.eye(N)

        # Vector unit: gather head (Head 3) and the elementwise ops
        if cfg.vector:
            self._vector_gather_head(layer, 3)
            self._vector_l5_rows().append_to(layer, col0=False)

        return layer

    def _init_indirect_read(self) -> TransformerLayer:
        """
        Layer for indirect memory read (LOAD), content-addressable search (FIND),
        and scratchpad correction.

        Head 1 (LOAD): Q=K symmetric on load_temp[0:logn] + pos_enc[0:logn]
        Head 2 (FIND): Q=K symmetric on find_temp[0:N] + idx_memory[0:N]

        FFN: Clears temp buffers + snaps scr_sub/scr_min back to bipolar.
        The correction runs after attention, so it fixes errors from both
        L2's memory read and L3's own LOAD/FIND attention.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        find_temp = cfg.idx_buffer + 3 * N
        load_temp = cfg.idx_buffer + 3 * N + N
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N

        large = 100.0
        num_rows_Q = N + logn   # head 3 (ARGMAX) uses value + address dims
        num_clear = N + logn    # clear find_temp (N) + load_temp (logn)
        num_correct = 12 * N    # 6 rows per bit × 2 targets (scr_sub, scr_min)
        num_argmax = 2 * logn + 2 * N  # clear block key + winner's tag -> find_temp
        layer = TransformerLayer(cfg, num_rows_Q, 2 * num_clear + num_correct + num_argmax,
                                 num_heads=3)

        # Head 1 (LOAD): Q=K on load_temp[0:logn] + pos_enc[0:logn]
        for i in range(logn):
            layer.Q1[i, load_temp + i] = 1.0
            layer.Q1[i, cfg.idx_pos_enc + i] = 1.0
        layer.K1 = layer.Q1.clone()

        # V1: copy memory → scr_min with weight 2.0
        for i in range(N):
            layer.V1[scr_min + i, cfg.idx_memory + i] = 2.0

        # Head 2 (FIND): Q=K on find_temp[0:N] + idx_memory[0:N]
        for i in range(N):
            layer.Q2[i, find_temp + i] = 1.0
            layer.Q2[i, cfg.idx_memory + i] = 1.0
        layer.K2 = layer.Q2.clone()

        # V2: copy tag → scr_min with weight 2.0
        for i in range(N):
            layer.V2[scr_min + i, cfg.idx_tag + i] = 2.0

        # Head 3 (ARGMAX): column 0 is the only source with a nonzero key
        # (L2 sets it for ARGMAX only), and its score against column j is
        #   A * (2*mem[j] + 1) - 2*j + sum_i key_i * pos_enc_i(j).
        # The key is +/-M on the address bits that select the block, so the
        # top column is the block's largest word, lowest address on ties.
        # Distinct candidates differ by >= 2, so there is never a 50/50 tie.
        # V3 moves the gate (buf_a[0] = 1) to the winner as its marker.
        A, _ = self._argmax_scales()
        gate = cfg.idx_buffer
        for i in range(N):
            layer.Q3[i, cfg.idx_memory + i] = 1.0
            layer.K3[i, gate] = A * (-(2 ** (N - 1)) if i == 0 else 2 ** (N - 1 - i))
        for i in range(logn):
            layer.Q3[N + i, cfg.idx_pos_enc + i] = 1.0
            layer.K3[N + i, gate] = -float(2 ** (logn - 1 - i))
            layer.K3[N + i, gate + 1 + i] = 1.0
        layer.V3[gate, gate] = 1.0

        # FFN Section 1: Clear find_temp (N rows) and load_temp (logn rows).
        # Indicator-gated to scratchpad columns only.
        r = 0
        for temp_start, temp_size in [(find_temp, N), (load_temp, logn)]:
            for i in range(temp_size):
                layer.W1[r, temp_start + i] = 1.0
                layer.W1[r, -1] = large
                layer.b1[r, :] = -large
                layer.W2[temp_start + i, r] = -1.0
                r += 1

                layer.W1[r, temp_start + i] = -1.0
                layer.W1[r, -1] = large
                layer.b1[r, :] = -large
                layer.W2[temp_start + i, r] = 1.0
                r += 1

        # FFN Section 2: Scratchpad correction: snap scr_sub/scr_min to bipolar.
        # eps = 0.25 makes every constant here exact in float32 (thresholds
        # 0.25/0.75 next to the ±100 indicator offset, scale 2), so exact ±1
        # inputs leave as exact ±1. With 0.1 they came out ~1e-6 off, which
        # L4's 2^(N-1) weights amplified to ~0.1 at N=16. Inputs are 0, ±1 or
        # ±2, all outside the 0.25..0.75 ramp.
        eps = 0.25
        scale = 1.0 / (1.0 - 2.0 * eps)

        for target in (scr_sub, scr_min):
            for i in range(N):
                layer.W1[r, target + i] = 1.0
                layer.W1[r + 1, target + i] = 1.0
                layer.W1[r + 2, target + i] = 1.0
                layer.W1[r + 3, target + i] = 1.0
                layer.W1[r + 4, target + i] = -1.0
                layer.W1[r + 5, target + i] = 1.0

                for k in range(6):
                    layer.W1[r + k, -1] = large

                layer.b1[r, :] = 1.0 - eps - large
                layer.b1[r + 1, :] = eps - large
                layer.b1[r + 2, :] = -eps - large
                layer.b1[r + 3, :] = -1.0 + eps - large
                layer.b1[r + 4, :] = -large
                layer.b1[r + 5, :] = -large

                layer.W2[target + i, r] = scale
                layer.W2[target + i, r + 1] = -scale
                layer.W2[target + i, r + 2] = scale
                layer.W2[target + i, r + 3] = -scale
                layer.W2[target + i, r + 4] = 1.0
                layer.W2[target + i, r + 5] = -1.0

                r += 6

            layer.b2[target:target + N, :cfg.s] = -1.0

        # FFN Section 3 (ARGMAX): clear the block key (buf_a[1:1+logn]) at
        # column 0; the marker buf_a[0] stays for L4.
        for i in range(logn):
            layer.W1[r, gate + 1 + i] = 1.0
            layer.W2[gate + 1 + i, r] = -1.0
            layer.W1[r + 1, gate + 1 + i] = -1.0
            layer.W2[gate + 1 + i, r + 1] = 1.0
            r += 2

        # The marked column copies its tag into find_temp:
        # ReLU(±tag + 2*marker - 2) is ±tag where marker = 1, else 0.
        # Column 0 is also marked but has no tag, so only the winner writes.
        for i in range(N):
            layer.W1[r, cfg.idx_tag + i] = 1.0
            layer.W1[r, gate] = 2.0
            layer.b1[r, :] = -2.0
            layer.W2[find_temp + i, r] = 1.0
            layer.W1[r + 1, cfg.idx_tag + i] = -1.0
            layer.W1[r + 1, gate] = 2.0
            layer.b1[r + 1, :] = -2.0
            layer.W2[find_temp + i, r + 1] = -1.0
            r += 2

        return layer

    def _init_scratchpad_correction(self) -> TransformerLayer:
        """Snap scr_sub/scr_min back to bipolar in scratchpad columns."""
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn

        layer = TransformerLayer(cfg, logn, 12 * N, num_heads=1)

        eps = 0.1
        scale = 1.0 / (1.0 - 2.0 * eps)
        large = 100.0
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N

        row = 0
        for target in (scr_sub, scr_min):
            for i in range(N):
                r = row + 6 * i
                layer.W1[r, target + i] = 1.0
                layer.W1[r + 1, target + i] = 1.0
                layer.W1[r + 2, target + i] = 1.0
                layer.W1[r + 3, target + i] = 1.0
                layer.W1[r + 4, target + i] = -1.0
                layer.W1[r + 5, target + i] = 1.0

                # Gate to scratchpad columns only.
                for k in range(6):
                    layer.W1[r + k, -1] = large

                layer.b1[r, :] = 1.0 - eps - large
                layer.b1[r + 1, :] = eps - large
                layer.b1[r + 2, :] = -eps - large
                layer.b1[r + 3, :] = -1.0 + eps - large
                layer.b1[r + 4, :] = -large
                layer.b1[r + 5, :] = -large

                layer.W2[target + i, r] = scale
                layer.W2[target + i, r + 1] = -scale
                layer.W2[target + i, r + 2] = scale
                layer.W2[target + i, r + 3] = -scale
                layer.W2[target + i, r + 4] = 1.0
                layer.W2[target + i, r + 5] = -1.0

            layer.b2[target:target + N, :cfg.s] = -1.0
            row += 6 * N

        return layer

    def _init_direct_subtract(self) -> TransformerLayer:
        """
        Single-layer direct subtraction: scr_min = scr_min - scr_sub.

        Replaces the 3-layer two's complement approach (flip, add-1, add)
        with a single layer that computes a - b directly using the
        6-threshold-per-bit pattern with:
          - Negated weights for scr_sub (implements bit-flip)
          - s_bias = 2^i (carry_in = 1, completes two's complement)

        The partial binary sum at position i (1-indexed from LSB) is:
          S_i = sum_{j=1}^{i} [bit_a_j + ~bit_b_j] * 2^(j-1) + 1
              = sum_{j=1}^{i} (a_j - b_j) * 2^(j-2) + 2^i

        where a = scr_min (minuend) and b = scr_sub (subtrahend).
        The 6 ReLU thresholds extract the result bit from S_i.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn

        scr_sub = cfg.idx_scratchpad       # subtrahend (b)
        scr_min = cfg.idx_scratchpad + N    # minuend (a), also output

        # 6*N for addition with borrow chain, 4*N for clearing scr_sub + scr_min,
        # 2 + 2*N for clearing the ARGMAX marker and find_temp
        num_ffn = 10 * N + 2 + 2 * N
        layer = TransformerLayer(cfg, logn, num_ffn, num_heads=2 if cfg.vector else 1)
        h1 = '1' if cfg.vector else ''

        # ARGMAX readback: Q=K on the marker (buf_a[0]), which is 1 only at
        # column 0 and the winning column after L3. They tie 50/50, so column 0
        # receives 2 * 0.5 * find_temp = the winner's tag in scr_min. Every
        # other column has a zero key and a zero find_temp, and for any other
        # opcode the whole head is zero.
        gate = cfg.idx_buffer
        find_temp = cfg.idx_buffer + 3 * N
        getattr(layer, f'Q{h1}')[0, gate] = 1.0
        setattr(layer, f'K{h1}', getattr(layer, f'Q{h1}').clone())
        for i in range(N):
            getattr(layer, f'V{h1}')[scr_min + i, find_temp + i] = 2.0

        # Section 1: Direct subtraction via 6-threshold pattern
        # For each bit position i (1-indexed from LSB):
        #   h = sum_{j=1}^{i} elem_j * (scr_min[N-j] - scr_sub[N-j])
        #   6 ReLU rows threshold h to extract the result bit
        for i in range(1, N + 1):
            row_base = 6 * (N - i)

            for j in range(1, i + 1):
                elem = (2**(j-1)) / 2
                # scr_min with positive weight (minuend)
                layer.W1[row_base, scr_min + N - j] = elem
                layer.W1[row_base + 1, scr_min + N - j] = elem
                layer.W1[row_base + 2, scr_min + N - j] = -elem
                layer.W1[row_base + 3, scr_min + N - j] = -elem
                layer.W1[row_base + 4, scr_min + N - j] = elem
                layer.W1[row_base + 5, scr_min + N - j] = elem
                # scr_sub with NEGATIVE weight (negation = bit flip)
                layer.W1[row_base, scr_sub + N - j] = -elem
                layer.W1[row_base + 1, scr_sub + N - j] = -elem
                layer.W1[row_base + 2, scr_sub + N - j] = elem
                layer.W1[row_base + 3, scr_sub + N - j] = elem
                layer.W1[row_base + 4, scr_sub + N - j] = -elem
                layer.W1[row_base + 5, scr_sub + N - j] = -elem

            # Biases: s_bias = 2^i (carry_in = 1, vs 2^i - 1 for plain add)
            s_bias = 2**i
            layer.b1[row_base, 0] = s_bias - 2**(i-1) + 1
            layer.b1[row_base + 1, 0] = s_bias - 2**(i-1)
            layer.b1[row_base + 2, 0] = -s_bias + 2**i
            layer.b1[row_base + 3, 0] = -s_bias + 2**i - 1
            layer.b1[row_base + 4, 0] = s_bias - 3 * 2**(i-1) + 1
            layer.b1[row_base + 5, 0] = s_bias - 3 * 2**(i-1)

            layer.W2[scr_min + N - i, row_base:row_base + 6] = 2 * torch.tensor(
                [1, -1, 1, -1, 1, -1], dtype=torch.float32)

        # b2 baseline for scr_min (same as original add layer)
        layer.b2[scr_min:scr_min + N, 0] = -3.0

        # Section 2: Clear scr_sub and scr_min
        # Subtract the operand preserved by the residual so only the result remains.
        offset = 6 * N

        # Clear scr_sub: scr_sub[i] -= scr_sub[i]
        layer.W1[offset:offset + N, scr_sub:scr_sub + N] = torch.eye(N)
        layer.W1[offset + N:offset + 2*N, scr_sub:scr_sub + N] = -torch.eye(N)
        layer.W2[scr_sub:scr_sub + N, offset:offset + N] = -torch.eye(N)
        layer.W2[scr_sub:scr_sub + N, offset + N:offset + 2*N] = torch.eye(N)

        # Clear scr_min: scr_min[i] -= scr_min[i] (result replaces via Section 1)
        layer.W1[offset + 2*N:offset + 3*N, scr_min:scr_min + N] = torch.eye(N)
        layer.W1[offset + 3*N:offset + 4*N, scr_min:scr_min + N] = -torch.eye(N)
        layer.W2[scr_min:scr_min + N, offset + 2*N:offset + 3*N] = -torch.eye(N)
        layer.W2[scr_min:scr_min + N, offset + 3*N:offset + 4*N] = torch.eye(N)

        # Section 3: Clear the ARGMAX marker and find_temp everywhere
        offset = 10 * N
        layer.W1[offset, gate] = 1.0
        layer.W2[gate, offset] = -1.0
        layer.W1[offset + 1, gate] = -1.0
        layer.W2[gate, offset + 1] = 1.0
        offset += 2
        layer.W1[offset:offset + N, find_temp:find_temp + N] = torch.eye(N)
        layer.W1[offset + N:offset + 2*N, find_temp:find_temp + N] = -torch.eye(N)
        layer.W2[find_temp:find_temp + N, offset:offset + N] = -torch.eye(N)
        layer.W2[find_temp:find_temp + N, offset + N:offset + 2*N] = torch.eye(N)

        # Vector unit: broadcast head, column-0 cleanup, per-column setup
        if cfg.vector:
            self._vector_broadcast_head(layer, 2)
            col0, mem = self._vector_l4_rows()
            col0.append_to(layer)
            mem.append_to(layer, col0=False)

        return layer

    def _init_cond_branch(self) -> List[TransformerLayer]:
        """Conditional branching with extended opcodes (2 layers).

        Layer 1 (merged): Branch flag computation + PC increment.
          These are independent FFN computations (flag reads scratchpad/opcodes,
          PC+1 reads PC rows) merged into a single layer.
        Layer 2: Select between branch target and PC+1 based on flag.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        large_const = 100.0

        # Combine branch-flag and PC-increment rows in one FFN. The fixed spare
        # rows cover scalar opcode suppression; page two adds one per opcode.
        page2_ops = PAGE2_OPCODES if cfg.vector else (OP_JMPI, OP_FALIGN, OP_FNORM) if cfg.page2 else ()
        num_ffn_flag = 2 + 2*N + 41 + len(page2_ops)
        num_ffn_pc = 8 * logn
        num_ffn = num_ffn_flag + num_ffn_pc
        layer1 = TransformerLayer(cfg, logn, num_ffn, num_heads=1)
        
        ref_result = cfg.idx_scratchpad + N
        ref_flag = cfg.idx_scratchpad
        addr_a = cfg.idx_scratch_cmd
        bit32 = logn - 6
        bit16 = logn - 5
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        
        # SUBLEQ mode gate (opcode bit 32)
        subleq_row = 2 + 2*N
        layer1.W1[subleq_row, mode_bit0] = 1.0
        layer1.b1[subleq_row, 0] = -0.5
        
        layer1.W1[0, ref_result] = 1.0
        
        layer1.W1[1, ref_result:ref_result+N] = -1.0
        layer1.b1[1, 0] = -N + 1
        
        layer1.W1[2:2+N, ref_result:ref_result+N] = torch.eye(N)
        layer1.W1[2+N:2+2*N, ref_result:ref_result+N] = -torch.eye(N)
        
        jmp_row = subleq_row + 1
        jmp_pat = opcode_pattern(OP_JMP, logn)
        for i in range(logn):
            layer1.W1[jmp_row, addr_a + i] = jmp_pat[i].item()
        layer1.b1[jmp_row, 0] = -logn + 0.5
        
        halt_row = jmp_row + 1
        halt_pat = opcode_pattern(OP_HALT, logn)
        for i in range(logn):
            layer1.W1[halt_row, addr_a + i] = halt_pat[i].item()
        layer1.b1[halt_row, 0] = -logn + 0.5
        
        # Suppress SUBLEQ for opcodes 4-7 (JZ, JNZ, INC, DEC)
        suppress_row = halt_row + 1
        for i in range(logn - 3):
            layer1.W1[suppress_row, addr_a + i] = -1.0
        layer1.W1[suppress_row, addr_a + (logn - 3)] = 1.0
        layer1.b1[suppress_row, 0] = -(logn - 3)
        
        # Suppress SUBLEQ for AND (opcode 12)
        suppress_and_row = suppress_row + 1
        and_pat = opcode_pattern(OP_AND, logn)
        for i in range(logn):
            layer1.W1[suppress_and_row, addr_a + i] = and_pat[i].item()
        layer1.b1[suppress_and_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for OR (opcode 13)
        suppress_or_row = suppress_and_row + 1
        or_pat = opcode_pattern(OP_OR, logn)
        for i in range(logn):
            layer1.W1[suppress_or_row, addr_a + i] = or_pat[i].item()
        layer1.b1[suppress_or_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for XOR (opcode 14)
        suppress_xor_row = suppress_or_row + 1
        xor_pat = opcode_pattern(OP_XOR, logn)
        for i in range(logn):
            layer1.W1[suppress_xor_row, addr_a + i] = xor_pat[i].item()
        layer1.b1[suppress_xor_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for MOV (opcode 1) - MOV also shouldn't branch
        suppress_mov_row = suppress_xor_row + 1
        mov_pat = opcode_pattern(OP_MOV, logn)
        for i in range(logn):
            layer1.W1[suppress_mov_row, addr_a + i] = mov_pat[i].item()
        layer1.b1[suppress_mov_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for ADD (opcode 2) - ADD shouldn't branch
        suppress_add_row = suppress_mov_row + 1
        add_pat = opcode_pattern(OP_ADD, logn)
        for i in range(logn):
            layer1.W1[suppress_add_row, addr_a + i] = add_pat[i].item()
        layer1.b1[suppress_add_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SUB (opcode 15) - SUB shouldn't branch
        suppress_sub_row = suppress_add_row + 1
        sub_pat = opcode_pattern(OP_SUB, logn)
        for i in range(logn):
            layer1.W1[suppress_sub_row, addr_a + i] = sub_pat[i].item()
        layer1.b1[suppress_sub_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SHL (opcode 8)
        suppress_shl_row = suppress_sub_row + 1
        shl_pat = opcode_pattern(OP_SHL, logn)
        for i in range(logn):
            layer1.W1[suppress_shl_row, addr_a + i] = shl_pat[i].item()
        layer1.b1[suppress_shl_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SHR (opcode 9)
        suppress_shr_row = suppress_shl_row + 1
        shr_pat = opcode_pattern(OP_SHR, logn)
        for i in range(logn):
            layer1.W1[suppress_shr_row, addr_a + i] = shr_pat[i].item()
        layer1.b1[suppress_shr_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for LOAD (opcode 11)
        suppress_load_row = suppress_shr_row + 1
        load_pat = opcode_pattern(OP_LOAD, logn)
        for i in range(logn):
            layer1.W1[suppress_load_row, addr_a + i] = load_pat[i].item()
        layer1.b1[suppress_load_row, 0] = -logn + 1.0

        # JZ: branch if zero
        jz_row = suppress_load_row + 1
        jz_pat = opcode_pattern(OP_JZ, logn)
        for i in range(logn):
            layer1.W1[jz_row, addr_a + i] = jz_pat[i].item()
        layer1.W1[jz_row, ref_result:ref_result+N] = -1.0
        layer1.b1[jz_row, 0] = -(logn + N) + 0.5
        
        # JNZ: branch if not zero
        # A one-bit opcode mismatch costs 2*jnz_w and the result bits add at
        # most 2*N, so jnz_w must exceed N (10 sufficed only up to N = 9).
        jnz_row = jz_row + 1
        jnz_pat = opcode_pattern(OP_JNZ, logn)
        jnz_w = max(10.0, N + 2.0)
        for i in range(logn):
            layer1.W1[jnz_row, addr_a + i] = jnz_pat[i].item() * jnz_w
        layer1.W1[jnz_row, ref_result:ref_result+N] = 1.0
        layer1.b1[jnz_row, 0] = -(logn * jnz_w - N + 0.5)

        # CMP branches on a negative value only. Cancel the default zero-result
        # contribution so zero does not take the branch.
        cmp_zero_row = jnz_row + 1
        cmp_pat = opcode_pattern(OP_CMP, logn)
        for i in range(logn):
            layer1.W1[cmp_zero_row, addr_a + i] = cmp_pat[i].item()
        layer1.W1[cmp_zero_row, ref_result:ref_result+N] = -1.0
        layer1.b1[cmp_zero_row, 0] = -(logn + N) + 0.5

        # Suppress SUBLEQ for FIND (opcode 16)
        suppress_find_row = cmp_zero_row + 1
        find_pat = opcode_pattern(OP_FIND, logn)
        for i in range(logn):
            layer1.W1[suppress_find_row, addr_a + i] = find_pat[i].item()
        layer1.b1[suppress_find_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SWAP (opcode 17)
        suppress_swap_row = suppress_find_row + 1
        swap_pat = opcode_pattern(OP_SWAP, logn)
        for i in range(logn):
            layer1.W1[suppress_swap_row, addr_a + i] = swap_pat[i].item()
        layer1.b1[suppress_swap_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for CMOV (opcode 18)
        suppress_cmov_row = suppress_swap_row + 1
        cmov_pat = opcode_pattern(OP_CMOV, logn)
        for i in range(logn):
            layer1.W1[suppress_cmov_row, addr_a + i] = cmov_pat[i].item()
        layer1.b1[suppress_cmov_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for MULACC (opcode 19)
        suppress_mulacc_row = suppress_cmov_row + 1
        mulacc_pat = opcode_pattern(OP_MULACC, logn)
        for i in range(logn):
            layer1.W1[suppress_mulacc_row, addr_a + i] = mulacc_pat[i].item()
        layer1.b1[suppress_mulacc_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for STORE (opcode 20)
        suppress_store_row = suppress_mulacc_row + 1
        store_pat = opcode_pattern(OP_STORE, logn)
        for i in range(logn):
            layer1.W1[suppress_store_row, addr_a + i] = store_pat[i].item()
        layer1.b1[suppress_store_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for opcodes 21-31 (none of them branch)
        suppress_bayes_rows = list(range(suppress_store_row + 1,
                                         suppress_store_row + 1 + len(BAYES_OPCODES)))
        assert suppress_bayes_rows[-1] < num_ffn_flag
        for r, opcode in zip(suppress_bayes_rows, BAYES_OPCODES):
            pat = opcode_pattern(opcode, logn)
            for i in range(logn):
                layer1.W1[r, addr_a + i] = pat[i].item()
            layer1.b1[r, 0] = -logn + 1.0

        # Page 2: JMPI always branches (to the target L2 put in addr_c, like
        # JMP's +2); every other page-2 opcode never does.
        page2_rows = list(range(2 + 2*N + 41, 2 + 2*N + 41 + len(page2_ops)))
        for r, opcode in zip(page2_rows, page2_ops):
            pat = opcode_pattern(opcode, logn)
            for i in range(logn):
                layer1.W1[r, addr_a + i] = pat[i].item()
            layer1.b1[r, 0] = -logn + 1.0

        layer1.W2[ref_flag, 0] = 1.0
        layer1.W2[ref_flag, 1] = 1.0
        layer1.W2[ref_flag, jmp_row] = 2.0
        layer1.W2[ref_flag, halt_row] = 2.0
        layer1.W2[ref_flag, suppress_row] = -1.0
        layer1.W2[ref_flag, suppress_and_row] = -1.0
        layer1.W2[ref_flag, suppress_or_row] = -1.0
        layer1.W2[ref_flag, suppress_xor_row] = -1.0
        layer1.W2[ref_flag, suppress_mov_row] = -1.0
        layer1.W2[ref_flag, suppress_add_row] = -1.0
        layer1.W2[ref_flag, suppress_sub_row] = -1.0
        layer1.W2[ref_flag, suppress_shl_row] = -1.0
        layer1.W2[ref_flag, suppress_shr_row] = -1.0
        layer1.W2[ref_flag, suppress_load_row] = -1.0
        layer1.W2[ref_flag, suppress_find_row] = -1.0
        layer1.W2[ref_flag, suppress_swap_row] = -1.0
        layer1.W2[ref_flag, suppress_cmov_row] = -1.0
        layer1.W2[ref_flag, suppress_mulacc_row] = -1.0
        layer1.W2[ref_flag, suppress_store_row] = -1.0
        for r in suppress_bayes_rows:
            layer1.W2[ref_flag, r] = -1.0
        for r, opcode in zip(page2_rows, page2_ops):
            layer1.W2[ref_flag, r] = 2.0 if opcode == OP_JMPI else -1.0
        layer1.W2[ref_flag, jz_row] = 2.0
        layer1.W2[ref_flag, jnz_row] = 2.0
        layer1.W2[ref_flag, cmp_zero_row] = -2.0
        
        layer1.W2[ref_result:ref_result+N, 2:2+N] = -torch.eye(N)
        layer1.W2[ref_result:ref_result+N, 2+N:2+2*N] = torch.eye(N)

        # PC+1 (merged into same layer, offset by num_ffn_flag)
        ref_pc = cfg.idx_pc
        pc_off = num_ffn_flag

        for i in range(1, logn + 1):
            for j in range(1, i + 1):
                elem = (2**(j-1)) / 2.0
                rb = pc_off + 6 * (logn - i)
                layer1.W1[rb:rb+6, ref_pc + logn - j] = torch.tensor([elem, elem, -elem, -elem, elem, elem])
            s_bias = (2**i - 1) / 2.0 + 1
            rb = pc_off + 6 * (logn - i)
            layer1.b1[rb:rb+6, 0] = torch.tensor([
                s_bias - 2**(i-1) + 1, s_bias - 2**(i-1),
                -s_bias + 2**i, -s_bias + 2**i - 1,
                s_bias - 3*2**(i-1) + 1, s_bias - 3*2**(i-1)
            ])
            layer1.W2[ref_pc + logn - i, rb:rb+6] = 2*torch.tensor([1,-1,1,-1,1,-1], dtype=torch.float32)
        layer1.b2[ref_pc:ref_pc+logn, 0] = -3.0

        off = pc_off + 6*logn
        layer1.W1[off:off+logn, ref_pc:ref_pc+logn] = torch.eye(logn)
        layer1.W1[off+logn:off+2*logn, ref_pc:ref_pc+logn] = -torch.eye(logn)
        layer1.W2[ref_pc:ref_pc+logn, off:off+logn] = -torch.eye(logn)
        layer1.W2[ref_pc:ref_pc+logn, off+logn:off+2*logn] = torch.eye(logn)

        # L7: Select PC
        layer2 = TransformerLayer(cfg, logn, 6*logn + 2, num_heads=1)
        ref_c = cfg.idx_scratch_cmd + 2*logn

        layer2.W1[:logn, ref_c:ref_c+logn] = torch.eye(logn)
        layer2.W1[:logn, ref_flag] = 1.0
        layer2.b1[:logn, 0] = -1.0
        layer2.W1[logn:2*logn, ref_c:ref_c+logn] = -torch.eye(logn)
        layer2.W1[logn:2*logn, ref_flag] = 1.0
        layer2.b1[logn:2*logn, 0] = -1.0

        layer2.W1[2*logn:3*logn, ref_pc:ref_pc+logn] = torch.eye(logn)
        layer2.W1[2*logn:3*logn, ref_flag] = -1.0
        layer2.W1[3*logn:4*logn, ref_pc:ref_pc+logn] = -torch.eye(logn)
        layer2.W1[3*logn:4*logn, ref_flag] = -1.0

        layer2.W1[4*logn:5*logn, ref_pc:ref_pc+logn] = torch.eye(logn)
        layer2.W1[5*logn:6*logn, ref_pc:ref_pc+logn] = -torch.eye(logn)

        layer2.W2[ref_pc:ref_pc+logn, :logn] = torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, logn:2*logn] = -torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 2*logn:3*logn] = torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 3*logn:4*logn] = -torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 4*logn:5*logn] = -torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 5*logn:6*logn] = torch.eye(logn)

        layer2.W1[6*logn, ref_flag] = 1.0
        layer2.W1[6*logn + 1, ref_flag] = -1.0
        layer2.W2[ref_flag, 6*logn] = -1.0
        layer2.W2[ref_flag, 6*logn + 1] = 1.0

        # Gate cond-branch FFN rows to scratchpad columns only to prevent
        # PC/scratchpad leakage into command/memory columns.
        for layer in (layer1, layer2):
            layer.W1[:, -1] = large_const
            layer.b1[:, :] -= large_const

        # Vector unit (memory columns): VLOGADD's add and VRAND's last round
        # in L6, payload cleanup in L7.
        if cfg.vector:
            self._vector_l6_rows().append_to(layer1, col0=False)
            self._vector_l7_rows().append_to(layer2, col0=False)

        return [layer1, layer2]

    # Opcodes 21-31: fixed-point and log-domain arithmetic
    # All of these are decoded in L2's FFN, which sees mem[b], mem[c] (and,
    # for MULSTEP/DIVSTEP, mem[b^1]) as +/-0.5 buffer bits. Two tools beyond
    # the per-bit corrections the older opcodes use:
    #   - Exact thresholds. Operands are integers, so a weighted bit sum such
    #     as mem[c] - mem[b] is an exact integer x, and
    #     ReLU(x - t + 1) - ReLU(x - t) = [x >= t]. Gating those rows on the
    #     opcode (and on individual bits) gives comparisons, windows and
    #     carry-outs in a single layer (_GatedRows.step).
    #   - Bit extraction. Bit j of a sum depends only on the partial sum S_j
    #     of operand bits 0..j, and is [S_j >= 2^j] - [S_j >= 2^(j+1)] +
    #     [S_j >= 3*2^j]; MULSTEP uses this to add and shift in one layer.

    def _argmax_scales(self) -> Tuple[float, float]:
        """ARGMAX selection-head score scales.

        A separates distinct values by more than the index tie-break spreads
        inside a block (2^t <= s); M makes any column outside the block lose
        to every column inside it.
        """
        cfg = self.cfg
        A = float(cfg.s)
        M = A * (2 ** cfg.N - 1) + cfg.n
        # Scores are summed in float64 (strandedcpu.subleq._scores); the key and query
        # entries themselves must stay exact in float32.
        if M * cfg.logn + A * 2 ** cfg.N + cfg.n >= 2 ** 53 or M >= 2 ** 24:
            warnings.warn(f"ARGMAX scores exceed the exact integer range for "
                          f"N={cfg.N}, n={cfg.n}; ARGMAX may pick the wrong column.")
        return A, M

    def _bayes_decode_rows(self) -> _GatedRows:
        """L2 FFN rows for opcodes 21-31 and the carry flag."""
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        buf_a = cfg.idx_buffer
        buf_b = cfg.idx_buffer + N
        buf_c = cfg.idx_buffer + 2 * N
        load_temp = cfg.idx_buffer + 4 * N
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N
        addr_a = cfg.idx_scratch_cmd
        addr_b = addr_a + logn
        addr_c = addr_a + 2 * logn
        carry = cfg.idx_carry
        k = int(np.log2(cfg.s))
        assert 1 + logn <= 2 * N, "ARGMAX key needs 1 + logn rows of buf_a/buf_b"

        def op(opcode):
            return _opcode_lits(addr_a, opcode, logn)

        g = _GatedRows()
        zero_sub = {scr_sub + i: -1.0 for i in range(N)}  # scr_sub = 0 (all -1)

        # L1 parks pos(b^1) - opcode in load_temp for MULSTEP/DIVSTEP; L2 head 1
        # has consumed it, so clear it before LOAD/XLAT write their pointers.
        for i in range(logn):
            g.clear(load_temp + i)

        # MAX (21) / LOGADD (22)
        # x = mem[c] - mem[b]. When x >= 1, replace scr_min (= mem[b]) by mem[c]
        # bit by bit (MOV-style corrections gated on the threshold).
        x = _lin((1, _word(buf_c, N, signed=True)), (-1, _word(buf_b, N, signed=True)))
        for opcode in (OP_MAX, OP_LOGADD):
            g.gate(op(opcode), zero_sub)
            for i in range(N):
                g.step(x, 1, op(opcode) + [_half(buf_c + i, 1), _half(buf_b + i, 0)],
                       {scr_min + i: 2.0})
                g.step(x, 1, op(opcode) + [_half(buf_c + i, 0), _half(buf_b + i, 1)],
                       {scr_min + i: -2.0})

        # LOGADD: scr_sub = -T[|x|], so L4 computes max + T[|x|]. T is a
        # non-increasing staircase, so emit one window per run of equal values
        # (both signs of x share it).
        table = logadd_table(cfg.log_frac_bits)
        d = 0
        while d < len(table):
            hi = d
            while hi + 1 < len(table) and table[hi + 1] == table[d]:
                hi += 1
            out = {scr_sub + i: 2.0 for i, bit in enumerate(_bits(-table[d], N)) if bit}
            if d == 0:
                g.window(x, -hi, hi, op(OP_LOGADD), out)
            else:
                g.window(x, d, hi, op(OP_LOGADD), out)
                g.window(x, -hi, -d, op(OP_LOGADD), out)
            d = hi + 1

        # XLAT (23): mem[b] = mem[c | mem[b]]
        # Same path as LOAD: scr_min = 0 here, L3 head 1 adds the word at
        # load_temp. The address is c with mem[b] OR-ed into the bits below
        # log2(s), so a table must be aligned to its size.
        o = op(OP_XLAT)
        g.gate(o, zero_sub)
        for i in range(N):
            g.gate(o + [_half(buf_b + i, 1)], {scr_min + i: -1.0})
            g.gate(o + [_half(buf_b + i, 0)], {scr_min + i: 1.0})
        for i in range(logn):
            p = logn - 1 - i                    # address bit position
            idx = buf_b + N - 1 - p             # same bit of mem[b]
            g.gate(o + [_full(addr_c + i, 1)], {load_temp + i: 1.0})
            if p >= k or p >= N:
                g.gate(o + [_full(addr_c + i, 0)], {load_temp + i: -1.0})
            else:
                g.gate(o + [_full(addr_c + i, 0), _half(idx, 1)], {load_temp + i: 1.0})
                g.gate(o + [_full(addr_c + i, 0), _half(idx, 0)], {load_temp + i: -1.0})

        # SHIFT (24): mem[b] <<= mem[c], arithmetic >> when negative
        # Each shift amount in (-N, N) is an exact match on mem[c]'s bits;
        # |amount| >= N saturates (left: 0, right: sign fill).
        o = op(OP_SHIFT)
        g.gate(o, zero_sub)
        for amount in range(-(N - 1), N):
            if amount == 0:
                continue
            match = o + [_half(buf_c + j, bit) for j, bit in enumerate(_bits(amount, N))]
            for i in range(N):
                src = i + amount
                if amount < 0:
                    src = max(src, 0)           # sign bit shifts in
                if src == i:
                    continue
                if src >= N:                    # zero shifts in
                    g.gate(match + [_half(buf_b + i, 1)], {scr_min + i: -2.0})
                else:
                    g.gate(match + [_half(buf_b + i, 0), _half(buf_b + src, 1)], {scr_min + i: 2.0})
                    g.gate(match + [_half(buf_b + i, 1), _half(buf_b + src, 0)], {scr_min + i: -2.0})
        amount = _word(buf_c, N, signed=True)
        for i in range(N):
            g.step(amount, N, o + [_half(buf_b + i, 1)], {scr_min + i: -2.0})
            if i > 0:
                g.step(_lin((-1, amount)), N, o + [_half(buf_b + i, 0), _half(buf_b, 1)],
                       {scr_min + i: 2.0})
                g.step(_lin((-1, amount)), N, o + [_half(buf_b + i, 1), _half(buf_b, 0)],
                       {scr_min + i: -2.0})

        # Carry flag: ADD, SUB and ADC (25) replace it
        # ADD/ADC: carry out of the unsigned sum. SUB: no borrow (mem[b] >= mem[c]
        # unsigned), so SBC is XOR with all-ones followed by ADC.
        ub = _word(buf_b, N)
        uc = _word(buf_c, N)
        for opcode, total, t in ((OP_ADD, _lin((1, ub), (1, uc)), 2 ** N),
                                 (OP_SUB, _lin((1, ub), (-1, uc)), 0),
                                 (OP_ADC, _lin((1, ub), (1, uc), (1, ({carry: 1.0}, 0.0))), 2 ** N)):
            g.copy(carry, op(opcode), {carry: -1.0})
            g.step(total, t, op(opcode), {carry: 1.0})

        # ADC: scr_sub = -(mem[c] + carry) = ~mem[c] + (1 - carry). The +1 carry
        # chain is ADD's, gated on carry == 0.
        o = op(OP_ADC)
        for i in range(N):
            g.gate(o + [_half(buf_c + i, 0)], {scr_sub + i: 1.0})
            g.gate(o + [_half(buf_c + i, 1)], {scr_sub + i: -1.0})
        for i in range(N):
            lower_zero = [_half(buf_c + j, 0) for j in range(i + 1, N)]
            lits = o + [_flag(carry, 0)] + lower_zero
            g.gate(lits + [_half(buf_c + i, 1)], {scr_sub + i: 2.0})
            g.gate(lits + [_half(buf_c + i, 0)], {scr_sub + i: -2.0})

        # MULSTEP (26): (hi:lo) = (mem[b]:mem[b^1]), multiplicand mem[c]
        # Right-shift multiply: S = hi + (LSB(lo) ? mc : 0) (N+1 bits);
        # hi' = S >> 1, lo' = (S & 1) : (lo >> 1). N steps from hi = 0 leave
        # the full 2N-bit unsigned product.
        hi, lo, mc = buf_b, buf_a, buf_c
        lsb = lo + N - 1
        o = op(OP_MULSTEP)
        g.gate(o, zero_sub)
        # LSB clear: hi' = hi >>> 1
        for i in range(1, N):
            g.gate(o + [_half(lsb, 0), _half(hi + i, 0), _half(hi + i - 1, 1)], {scr_min + i: 2.0})
            g.gate(o + [_half(lsb, 0), _half(hi + i, 1), _half(hi + i - 1, 0)], {scr_min + i: -2.0})
        g.gate(o + [_half(lsb, 0), _half(hi, 1)], {scr_min: -2.0})
        # LSB set: drop scr_min to 0, then add bit N-i of hi + mc into bit i
        added = o + [_half(lsb, 1)]
        for i in range(N):
            g.gate(added + [_half(hi + i, 1)], {scr_min + i: -2.0})
        g.step(_lin((1, _word(hi, N)), (1, _word(mc, N))), 2 ** N, added, {scr_min: 2.0})
        for i in range(1, N):
            j = N - i
            S = _lin((1, _word(hi, N, upto=j)), (1, _word(mc, N, upto=j)))
            g.step(S, 2 ** j, added, {scr_min + i: 2.0})
            g.step(S, 2 ** (j + 1), added, {scr_min + i: -2.0})
            g.step(S, 3 * 2 ** j, added, {scr_min + i: 2.0})
        # lo' goes out through buf_c (L5 head 2 writes 2*buf_c to addr_c = b^1)
        for i in range(1, N):
            g.gate(o + [_half(lo + i - 1, 1), _half(mc + i, 0)], {mc + i: 1.0})
            g.gate(o + [_half(lo + i - 1, 0), _half(mc + i, 1)], {mc + i: -1.0})
        for h, l, m, old in product((0, 1), repeat=4):
            new = h ^ (l & m)                   # bit 0 of S
            if new != old:
                g.gate(o + [_half(hi + N - 1, h), _half(lsb, l), _half(mc + N - 1, m), _half(mc, old)],
                       {mc: 1.0 if new else -1.0})
        self._pair_writeback(g, o)

        # DIVSTEP (27): (rem:quo) = (mem[b]:mem[b^1]), divisor mem[c]
        # Restoring division: R' = 2*rem + MSB(quo); if R' >= D subtract D and
        # shift a 1 into quo. N steps from rem = 0 leave quotient and remainder.
        # The subtraction is L4's: scr_min = (rem << 1) | MSB(quo) and
        # scr_sub = D or 0. R' - D fits in N bits whenever it is taken.
        rem, quo, D = buf_b, buf_a, buf_c
        o = op(OP_DIVSTEP)
        T = _lin((2, _word(rem, N)), (1, ({quo: 1.0}, 0.5)), (-1, _word(D, N)))  # R' - D
        g.gate(o, zero_sub)
        for i in range(N):
            g.step(T, 0, o + [_half(D + i, 1)], {scr_sub + i: 2.0})
        for i in range(N):
            src = rem + i + 1 if i < N - 1 else quo
            g.gate(o + [_half(rem + i, 0), _half(src, 1)], {scr_min + i: 2.0})
            g.gate(o + [_half(rem + i, 1), _half(src, 0)], {scr_min + i: -2.0})
        # quo' = (quo << 1) | [R' >= D], out through buf_c
        for i in range(N - 1):
            g.gate(o + [_half(quo + i + 1, 1), _half(D + i, 0)], {D + i: 1.0})
            g.gate(o + [_half(quo + i + 1, 0), _half(D + i, 1)], {D + i: -1.0})
        g.step(T, 0, o + [_half(D + N - 1, 0)], {D + N - 1: 1.0})
        g.step(_lin((-1, T)), 1, o + [_half(D + N - 1, 1)], {D + N - 1: -1.0})
        self._pair_writeback(g, o)

        # LFSR (28): Galois step, mem[b] = (mem[b] >>> 1) ^ (LSB ? mem[c] : 0)
        o = op(OP_LFSR)
        g.gate(o, zero_sub)
        lsb = buf_b + N - 1
        for i in range(N):
            ins = list(dict.fromkeys([buf_b + i, lsb, buf_c + i] + ([buf_b + i - 1] if i else [])))
            for vals in product((0, 1), repeat=len(ins)):
                v = dict(zip(ins, vals))
                new = (v[buf_b + i - 1] if i else 0) ^ (v[lsb] & v[buf_c + i])
                if new != v[buf_b + i]:
                    g.gate(o + [_half(row, bit) for row, bit in v.items()],
                           {scr_min + i: 2.0 if new else -2.0})

        # CLZ (29): mem[b] = leading zeros of mem[c] (N for zero)
        o = op(OP_CLZ)
        g.gate(o, zero_sub)
        for i in range(N):
            g.gate(o + [_half(buf_b + i, 1)], {scr_min + i: -2.0})
        for count in range(N + 1):
            lits = [_half(buf_c + j, 0) for j in range(count)]
            if count < N:
                lits.append(_half(buf_c + count, 1))
            out = {scr_min + i: 2.0 for i, bit in enumerate(_bits(count, N)) if bit}
            if out:
                g.gate(o + lits, out)

        # ABS (30): mem[b] = |mem[c]| (wraps at the most negative value)
        # c >= 0: scr_min = c, scr_sub = 0.  c < 0: scr_min = 0, scr_sub = c.
        o = op(OP_ABS)
        g.gate(o, zero_sub)
        for i in range(N):
            g.gate(o + [_half(buf_c, 1), _half(buf_c + i, 1)], {scr_sub + i: 2.0})
            g.gate(o + [_half(buf_c, 0), _half(buf_c + i, 1), _half(buf_b + i, 0)], {scr_min + i: 2.0})
            g.gate(o + [_half(buf_c, 0), _half(buf_c + i, 0), _half(buf_b + i, 1)], {scr_min + i: -2.0})
            g.gate(o + [_half(buf_c, 1), _half(buf_b + i, 1)], {scr_min + i: -2.0})

        # ARGMAX (31): mem[b] = argmax_i mem[c + i], 0 <= i < 2**mem[b]
        # L3 head 3 sends a marker from column 0 to the winning column, the
        # winner copies its tag (memory offset) to find_temp, and L4 reads it
        # back into scr_min. Here: scr_min = 0, scr_sub = c - s (block base
        # offset, so L4 leaves the index within the block), and the selection
        # key in buf_a/buf_b (free until L5 at column 0):
        #   buf_a[0]      = 1, the gate and marker
        #   buf_a[1 + i]  = +/-M for address bits at or above log2(block)
        #                   (matching c's bit), 0 below
        o = op(OP_ARGMAX)
        g.gate(o, zero_sub)
        for p in range(min(N, logn)):
            cbit = addr_c + logn - 1 - p
            out = {scr_sub + N - 1 - p: 2.0}
            if p < k:
                g.gate(o + [_full(cbit, 1)], out)
                continue
            # c - 2^k flips bit p when address bits k..p-1 are all zero
            g.gate(o + [_full(addr_c + logn - 1 - q, 0) for q in range(k, p)] + [_full(cbit, 0)], out)
            for q in range(k, p):
                first_set = [_full(addr_c + logn - 1 - u, 0) for u in range(k, q)]
                g.gate(o + first_set + [_full(addr_c + logn - 1 - q, 1), _full(cbit, 1)], out)
        for i in range(N):
            g.gate(o + [_half(buf_b + i, 1)], {scr_min + i: -1.0})
            g.gate(o + [_half(buf_b + i, 0)], {scr_min + i: 1.0})
        g.gate(o, {buf_a: 1.0})
        _, M = self._argmax_scales()
        size = _word(buf_b, N, signed=True)
        for i in range(logn):
            above = _lin((-1, size), (1, ({}, float(logn - 1 - i))))   # bit - log2(block)
            g.step(above, 0, o + [_full(addr_c + i, 1)], {buf_a + 1 + i: M})
            g.step(above, 0, o + [_full(addr_c + i, 0)], {buf_a + 1 + i: -M})

        return g

    def _pair_writeback(self, g: _GatedRows, lits) -> None:
        """Point addr_c at b^1 so L5 head 2 (SWAP's second write) stores buf_c there."""
        logn = self.cfg.logn
        addr_b = self.cfg.idx_scratch_cmd + logn
        addr_c = addr_b + logn
        for i in range(logn):
            flip = i == logn - 1
            g.gate(lits + [_full(addr_c + i, 0), _full(addr_b + i, flip ^ 1)], {addr_c + i: 2.0})
            g.gate(lits + [_full(addr_c + i, 1), _full(addr_b + i, flip)], {addr_c + i: -2.0})

    # Page 2 (opcodes 64-95): indirect jump, float assists, vector unit

    def _page2_decode_rows(self) -> _GatedRows:
        """L2 FFN rows for JMPI, FALIGN, FNORM, VSETVL and the column-0 side
        of vector instructions."""
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        buf_a = cfg.idx_buffer
        buf_b = cfg.idx_buffer + N
        buf_c = cfg.idx_buffer + 2 * N
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N
        addr_a = cfg.idx_scratch_cmd
        addr_c = addr_a + 2 * logn

        def op(opcode):
            return _opcode_lits(addr_a, opcode, logn)

        g = _GatedRows()
        zero_sub = {scr_sub + i: -1.0 for i in range(N)}

        # JMPI (64): PC = s + m + mem[c], mem[c] an instruction index
        # Rewrite addr_c (the target L7 selects) to that column; L6 forces the
        # branch. Bit j of base + U(mem[c]) comes from the partial sum of bits
        # 0..j, as in MULSTEP.
        o = op(OP_JMPI)
        g.gate(o, zero_sub)
        # L5 head 2 writes 2*buf_c to addr_c, which now names an instruction
        # column: zero buf_c so that write leaves its (zero) memory rows alone.
        for i in range(N):
            g.gate(o + [_half(buf_c + i, 1)], {buf_c + i: -0.5})
            g.gate(o + [_half(buf_c + i, 0)], {buf_c + i: 0.5})
        base = cfg.s + cfg.m
        for i in range(logn):
            j = logn - 1 - i
            g.gate(o + [_full(addr_c + i, 1)], {addr_c + i: -2.0})
            P = _lin((1, _word(buf_c, N, upto=min(j, N - 1))), (1, ({}, float(base % (1 << (j + 1))))))
            g.step(P, 2 ** j, o, {addr_c + i: 2.0})
            g.step(P, 2 ** (j + 1), o, {addr_c + i: -2.0})
            g.step(P, 3 * 2 ** j, o, {addr_c + i: 2.0})

        # FALIGN (65): (e, m) = (mem[b], mem[b^1]), E = mem[c]
        # If E > e: m >>= E - e (arithmetic) and e = E, so two floats aligned
        # to each other's exponent add mantissa to mantissa.
        e, m, E = buf_b, buf_a, buf_c
        o = op(OP_FALIGN)
        g.gate(o, zero_sub)
        d = _lin((1, _word(E, N, signed=True)), (-1, _word(e, N, signed=True)))
        for i in range(N):
            g.step(d, 1, o + [_half(E + i, 1), _half(e + i, 0)], {scr_min + i: 2.0})
            g.step(d, 1, o + [_half(E + i, 0), _half(e + i, 1)], {scr_min + i: -2.0})
        # m' goes out through buf_c (now holding E): bit i becomes m[i - d],
        # the sign once d reaches past it, or m[i] when d <= 0
        for i in range(N):
            keep = _lin((-1, d))
            g.step(keep, 0, o + [_half(m + i, 1), _half(E + i, 0)], {E + i: 1.0})
            g.step(keep, 0, o + [_half(m + i, 0), _half(E + i, 1)], {E + i: -1.0})
            for k in range(1, N):
                src = m + max(i - k, 0)
                g.window(d, k, k, o + [_half(src, 1), _half(E + i, 0)], {E + i: 1.0})
                g.window(d, k, k, o + [_half(src, 0), _half(E + i, 1)], {E + i: -1.0})
            g.step(d, N, o + [_half(m, 1), _half(E + i, 0)], {E + i: 1.0})
            g.step(d, N, o + [_half(m, 0), _half(E + i, 1)], {E + i: -1.0})
        self._pair_writeback(g, o)

        # FNORM (66): normalize (e, m) = (mem[b], mem[b^1])
        # Shift m left by k = (leading sign-bit run) - 1 so its two top bits
        # differ, and e -= k. m = -1 becomes -2^(N-1). m = 0 gets the most
        # negative exponent, so a zero never outranks the other operand in
        # FALIGN: scr_sub = e + 2^(N-1) (e with its sign bit flipped).
        o = op(OP_FNORM)
        g.gate(o, zero_sub)
        for i in range(N):                       # buf_c (mem[c], unused) -> 0
            g.gate(o + [_half(buf_c + i, 1)], {buf_c + i: -0.5})
            g.gate(o + [_half(buf_c + i, 0)], {buf_c + i: 0.5})
        is_zero = o + [_half(m + j, 0) for j in range(N)]
        g.gate(is_zero, {buf_c + i: -0.5 for i in range(N)})
        g.gate(is_zero + [_half(e, 0)], {scr_sub: 2.0})
        for i in range(1, N):
            g.gate(is_zero + [_half(e + i, 1)], {scr_sub + i: 2.0})
        for sign in (0, 1):
            for k in range(N - 1 + sign):
                lits = o + [_half(m + j, sign) for j in range(k + 1)]
                if k < N - 1:
                    lits.append(_half(m + k + 1, 1 - sign))
                for i in range(N - k):
                    g.gate(lits + [_half(m + i + k, 1)], {buf_c + i: 0.5})
                    g.gate(lits + [_half(m + i + k, 0)], {buf_c + i: -0.5})
                if k:
                    out = {buf_c + i: -0.5 for i in range(N - k, N)}
                    out.update({scr_sub + i: 2.0 for i, bit in enumerate(_bits(k, N)) if bit})
                    g.gate(lits, out)
        self._pair_writeback(g, o)

        if not cfg.vector:
            return g

        # VSETVL (67): VL = 2^mem[b]; vmask[bit] = [bit < mem[b]]
        o = op(OP_VSETVL)
        g.gate(o, zero_sub)
        t = _word(buf_b, N, signed=True)
        for i in range(logn):
            g.copy(cfg.idx_vmask + i, o, {cfg.idx_vmask + i: -1.0})
            g.step(_lin((1, t), (1, ({}, -float(logn - 1 - i)))), 1, o, {cfg.idx_vmask + i: 1.0})

        # --- Vector instructions: raise vgate for L4's broadcast. The column-0
        # result is discarded (L4 points both write heads at address 0).
        for opcode in VECTOR_OPCODES:
            g.gate(op(opcode), {**zero_sub, cfg.idx_vgate: 1.0})
        return g

    # Vector unit
    # A vector instruction acts on every column of an aligned block of
    # VL = 2^t memory columns at once:
    #   L2 FFN   column 0 raises vgate (above).
    #   L4 head  query-wise broadcast: every memory column takes column 0's
    #            instruction, VL mask and scalar mem[c] (one key, all queries).
    #   L4 FFN   per column: block membership, the address of its second
    #            operand (want), VRAND's first hash round.
    #   L5 head  query-wise gather: each column reads the word at want.
    #   L5 FFN   the elementwise op, written straight into the column's memory
    #            rows; L6 finishes VLOGADD, L6/L7 run VRAND's later rounds.
    # Rows reused at memory columns (all idle there in that window and
    # cleared by L8, except cmds, which L7 clears):
    #   cmds      instruction payload (a, b, c fields)
    #   load_temp VL mask, then flags (block mismatch, butterfly half, ...)
    #   buf_b     scalar mem[c] (+/-0.5), later stage data (+/-1)
    #   PC        want (gather address)
    #   scr_sub   gathered word, VRAND's first round

    def _vrows(self):
        cfg = self.cfg
        N, logn = cfg.N, cfg.logn
        lt = cfg.idx_buffer + 4 * N
        return dict(op=0, b=logn, c=2 * logn, pos=cfg.idx_pos_enc, mem=cfg.idx_memory,
                    tag=cfg.idx_tag, want=cfg.idx_pc, vm=lt, mm=lt, upper=lt + 1,
                    valid=lt + 2, ycarry=lt + 3, scal=cfg.idx_buffer + N,
                    y=cfg.idx_scratchpad, memcol=cfg.idx_memcol, vcarry=cfg.idx_vcarry)

    def _vop(self, opcode, ignore=0):
        """Memory-column literals: a vector payload with this opcode."""
        v = self._vrows()
        return [_flag(v['memcol'], 1)] + _opcode_lits(v['op'], opcode, self.cfg.logn, ignore)

    def _vector_broadcast_head(self, layer, h):
        """Query-wise head h of L4: key = vgate (column 0 only), query = memcol."""
        cfg = self.cfg
        N, logn = cfg.N, cfg.logn
        v = self._vrows()
        Q = getattr(layer, f'Q{h}')
        K = getattr(layer, f'K{h}')
        V = getattr(layer, f'V{h}')
        Q[0, cfg.idx_memcol] = 1.0
        K[0, cfg.idx_vgate] = 1.0
        V[0:3 * logn, cfg.idx_scratch_cmd:cfg.idx_scratch_cmd + 3 * logn] = torch.eye(3 * logn)
        V[v['vm']:v['vm'] + logn, cfg.idx_vmask:cfg.idx_vmask + logn] = torch.eye(logn)
        buf_c = cfg.idx_buffer + 2 * N
        V[v['scal']:v['scal'] + N, buf_c:buf_c + N] = torch.eye(N)
        layer.query_wise.add(h)

    def _vector_gather_head(self, layer, h):
        """Query-wise head h of L5: query = want (PC rows), key = pos_enc."""
        cfg = self.cfg
        N, logn = cfg.N, cfg.logn
        v = self._vrows()
        Q = getattr(layer, f'Q{h}')
        K = getattr(layer, f'K{h}')
        V = getattr(layer, f'V{h}')
        Q[:, v['want']:v['want'] + logn] = torch.eye(logn)
        K[:, v['pos']:v['pos'] + logn] = torch.eye(logn)
        V[v['y']:v['y'] + N, v['mem']:v['mem'] + N] = torch.eye(N)
        V[v['ycarry'], v['vcarry']] = 1.0
        layer.query_wise.add(h)

    def _vector_l4_rows(self):
        """L4 FFN: column-0 cleanup, and per-column precompute."""
        cfg = self.cfg
        N, logn = cfg.N, cfg.logn
        v = self._vrows()
        k_s = int(np.log2(cfg.s))
        addr_b = cfg.idx_scratch_cmd + logn
        col0 = _GatedRows()
        col0.clear(cfg.idx_vgate)
        for a in (addr_b, addr_b + logn):         # write heads -> address 0, a no-op
            for i in range(logn):
                col0.gate([_flag(cfg.idx_vgate, 1), _full(a + i, 1)], {a + i: -2.0})

        g = _GatedRows()
        bit64 = v['op'] + logn - 7                 # set in every vector opcode
        page = [_flag(v['memcol'], 1), _full(bit64, 1)]
        vm = lambda p: v['vm'] + logn - 1 - p     # VL-mask row of address bit p
        pos = lambda p: v['pos'] + logn - 1 - p
        crow = lambda p: v['c'] + logn - 1 - p
        want = lambda p: v['want'] + logn - 1 - p
        scal = lambda p: v['scal'] + N - 1 - p    # bit p of mem[c]

        # VL mask -> flags. mm = 2 * (block-address bits that differ from b's);
        # a column is in the block iff mm = 0.
        for i in range(logn):
            g.copy(v['vm'] + i, page, {v['vm'] + i: -1.0})
        for i in range(logn):
            for sgn in (1.0, -1.0):
                g.rows.append(({v['pos'] + i: sgn, v['b'] + i: -sgn, v['vm'] + i: -2.0,
                                v['memcol']: 4.0, bit64: 2.0}, -6.0, {v['mm']: 1.0}))

        def set_want(lits, p, src_row, flip=False):
            """want bit p = the +/-1 value at src_row (negated if flip)."""
            g.gate(lits + [_full(src_row, 1)], {want(p): -1.0 if flip else 1.0})
            g.gate(lits + [_full(src_row, 0)], {want(p): 1.0 if flip else -1.0})

        # Block source: want = (own address in the block) | c's block address
        for opcode in (OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX, OP_VLOGADD, OP_VCADD):
            for p in range(logn):
                set_want(self._vop(opcode) + [_flag(vm(p), 1)], p, pos(p))
                set_want(self._vop(opcode) + [_flag(vm(p), 0)], p, crow(p))
        # XOR partner: want = own address ^ (mem[c] within the block)
        for opcode in (OP_VPERMX, OP_VADDX, OP_VBFLY, OP_VMAXX, OP_VLOGADDX):
            for p in range(logn):
                set_want(self._vop(opcode) + [_flag(vm(p), 0)], p, pos(p))
                if p < N:
                    set_want(self._vop(opcode) + [_flag(vm(p), 1), _half(scal(p), 0)], p, pos(p))
                    set_want(self._vop(opcode) + [_flag(vm(p), 1), _half(scal(p), 1)], p, pos(p), flip=True)
                else:
                    set_want(self._vop(opcode) + [_flag(vm(p), 1)], p, pos(p))
        # VXLAT: want = c | (own word, bits below log2(s)), as XLAT
        for p in range(logn):
            lits = self._vop(OP_VXLAT)
            g.gate(lits + [_full(crow(p), 1)], {want(p): 1.0})
            if p < min(k_s, N):
                x = v['mem'] + N - 1 - p
                g.gate(lits + [_full(crow(p), 0), _full(x, 1)], {want(p): 1.0})
                g.gate(lits + [_full(crow(p), 0), _full(x, 0)], {want(p): -1.0})
            else:
                g.gate(lits + [_full(crow(p), 0)], {want(p): -1.0})
        # VREV / VTRANS: c's block, low t address bits reversed / halves swapped.
        # t is read off the VL mask (bit t-1 set, bit t clear).
        for t in range(k_s + 1):
            exact = ([_flag(vm(t - 1), 1)] if t else []) + ([_flag(vm(t), 0)] if t < logn else [])
            for p in range(logn):
                if p >= t:
                    set_want(self._vop(OP_VREV) + exact, p, crow(p))
                    set_want(self._vop(OP_VTRANS) + exact, p, crow(p))
                    continue
                set_want(self._vop(OP_VREV) + exact, p, pos(t - 1 - p))
                half = t // 2
                if t % 2:                           # not square: plain copy
                    set_want(self._vop(OP_VTRANS) + exact, p, pos(p))
                else:
                    set_want(self._vop(OP_VTRANS) + exact, p, pos(p + half if p < half else p - half))
        # VBFLY: which half of the butterfly this column is (mem[c] = 2^s)
        for p in range(min(N, logn)):
            g.gate(self._vop(OP_VBFLY) + [_half(scal(p), 1), _full(pos(p), 1)], {v['upper']: 1.0})
        # VRAND round 1: u = seed + sum_p c[p] * tag_p (mod 2^N) into scr_sub,
        # seed = mem[c], tag = the column's memory offset (vrand_constants)
        lits = self._vop(OP_VRAND)
        coef, _, _, _ = vrand_constants(N)
        g.gate(lits, {v['y'] + i: -1.0 for i in range(N)})
        for j in range(N):
            mod = 1 << (j + 1)
            w = {scal(p): float(1 << p) for p in range(j + 1)}
            const = 0.5 * sum(1 << p for p in range(j + 1))
            for q in range(N):
                cq = coef[q] % mod
                w[v['tag'] + N - 1 - q] = w.get(v['tag'] + N - 1 - q, 0.0) + 0.5 * cq
                const += 0.5 * cq
            top = (mod - 1) + sum(c % mod for c in coef)
            self._mod_bit(g, lits, (w, const), top, j, v['y'] + N - 1 - j)
        return col0, g

    def _mod_bit(self, g, lits, P, top, j, out):
        """Add +/-2 at out for bit j of the integer form P (0 <= P <= top):
        [P mod 2^(j+1) >= 2^j] as a staircase of steps."""
        mod = 1 << (j + 1)
        q = 0
        while (1 << j) + q * mod <= top:
            g.step(P, (1 << j) + q * mod, lits, {out: 2.0})
            g.step(P, (q + 1) * mod, lits, {out: -2.0})
            q += 1

    def _sum_bits(self, g, lits, partial, dest):
        """Write the low N bits of an integer sum into the +/-1 rows at dest.
        partial(j) is the sum restricted to operand bits 0..j (plus any carry
        in); its bit j is the result's, and it stays below 2^(j+2)."""
        N = self.cfg.N
        for i in range(N):
            j = N - 1 - i
            g.gate(lits + [_full(dest + i, 1)], {dest + i: -2.0})
            S = partial(j)
            g.step(S, 2 ** j, lits, {dest + i: 2.0})
            g.step(S, 2 ** (j + 1), lits, {dest + i: -2.0})
            g.step(S, 3 * 2 ** j, lits, {dest + i: 2.0})

    def _vector_l5_rows(self) -> _GatedRows:
        """L5 FFN: the elementwise ops, written into memory rows."""
        cfg = self.cfg
        N, logn = cfg.N, cfg.logn
        v = self._vrows()
        X, Y, S = v['mem'], v['y'], v['scal']
        inside = (v['mm'], -1.0, 1.0)              # mm == 0
        word = lambda row, j=None, full=True: _word(row, N, upto=j, full=full)
        g = _GatedRows()

        def copy_into_x(lits, src, src_full=True):
            lit = _full if src_full else _half
            for i in range(N):
                g.gate(lits + [_full(X + i, 0), lit(src + i, 1)], {X + i: 2.0})
                g.gate(lits + [_full(X + i, 1), lit(src + i, 0)], {X + i: -2.0})

        def set_vcarry(lits, total):
            g.copy(v['vcarry'], lits, {v['vcarry']: -1.0})
            g.step(total, 2 ** N, lits, {v['vcarry']: 1.0})

        xor = OP_VADDX - OP_VADD                    # the partner-source bit
        # copies
        copy_into_x(self._vop(OP_VMOV, xor) + [inside], Y)
        for opcode in (OP_VREV, OP_VTRANS, OP_VXLAT):
            copy_into_x(self._vop(opcode) + [inside], Y)
        copy_into_x(self._vop(OP_VFILL) + [inside], S, src_full=False)

        # VADD / VADDX: x + y, carry out
        lits = self._vop(OP_VADD, xor) + [inside]
        self._sum_bits(g, lits, lambda j: _lin((1, word(X, j)), (1, word(Y, j))), X)
        set_vcarry(lits, _lin((1, word(X)), (1, word(Y))))
        # VSUB: x + ~y + 1, carry = no borrow
        lits = self._vop(OP_VSUB) + [inside]
        sub = lambda j: _lin((1, word(X, j)), (-1, word(Y, j)), (1, ({}, float(2 ** (j + 1)))))
        self._sum_bits(g, lits, sub, X)
        set_vcarry(lits, sub(N - 1))
        # VCADD: x + carry of the corresponding column of c's block
        lits = self._vop(OP_VCADD) + [inside]
        cadd = lambda j: _lin((1, word(X, j)), (1, ({v['ycarry']: 1.0}, 0.0)))
        self._sum_bits(g, lits, cadd, X)
        set_vcarry(lits, cadd(N - 1))
        # VBFLY: lower half x + partner, upper half partner - x
        lits = self._vop(OP_VBFLY) + [inside]
        self._sum_bits(g, lits + [_flag(v['upper'], 0)],
                       lambda j: _lin((1, word(X, j)), (1, word(Y, j))), X)
        self._sum_bits(g, lits + [_flag(v['upper'], 1)],
                       lambda j: _lin((1, word(Y, j)), (-1, word(X, j)), (1, ({}, float(2 ** (j + 1))))), X)

        # VMAX / VLOGADD (and the X forms): replace x by y where y > x
        d = _lin((1, _word(Y, N, signed=True, full=True)), (-1, _word(X, N, signed=True, full=True)))
        for opcode in (OP_VMAX, OP_VLOGADD):
            lits = self._vop(opcode, xor) + [inside]
            for i in range(N):
                g.step(d, 1, lits + [_full(Y + i, 1), _full(X + i, 0)], {X + i: 2.0})
                g.step(d, 1, lits + [_full(Y + i, 0), _full(X + i, 1)], {X + i: -2.0})
        # VLOGADD: addend T[|d|] (as +/-1 bits) into buf_b for L6, valid = 1
        lits = self._vop(OP_VLOGADD, xor) + [inside]
        for i in range(N):
            g.gate(lits + [_half(S + i, 1)], {S + i: -0.5})
            g.gate(lits + [_half(S + i, 0)], {S + i: 0.5})
        g.gate(lits, {**{S + i: -1.0 for i in range(N)}, v['valid']: 1.0})
        table = logadd_table(cfg.log_frac_bits)
        lo = 0
        while lo < len(table):
            hi = lo
            while hi + 1 < len(table) and table[hi + 1] == table[lo]:
                hi += 1
            out = {S + i: 2.0 for i, bit in enumerate(_bits(table[lo], N)) if bit}
            if lo == 0:
                g.window(d, -hi, hi, lits, out)
            else:
                g.window(d, lo, hi, lits, out)
                g.window(d, -hi, -lo, lits, out)
            lo = hi + 1

        # VSHIFT: as SHIFT, amount = mem[c]
        lits = self._vop(OP_VSHIFT) + [inside]
        for amount in range(-(N - 1), N):
            if amount == 0:
                continue
            match = lits + [_half(S + j, bit) for j, bit in enumerate(_bits(amount, N))]
            for i in range(N):
                src = max(i + amount, 0) if amount < 0 else i + amount
                if src == i:
                    continue
                if src >= N:
                    g.gate(match + [_full(X + i, 1)], {X + i: -2.0})
                else:
                    g.gate(match + [_full(X + i, 0), _full(X + src, 1)], {X + i: 2.0})
                    g.gate(match + [_full(X + i, 1), _full(X + src, 0)], {X + i: -2.0})
        amount = _word(S, N, signed=True)
        for i in range(N):
            g.step(amount, N, lits + [_full(X + i, 1)], {X + i: -2.0})
            if i:
                g.step(_lin((-1, amount)), N, lits + [_full(X + i, 0), _full(X, 1)], {X + i: 2.0})
                g.step(_lin((-1, amount)), N, lits + [_full(X + i, 1), _full(X, 0)], {X + i: -2.0})

        # VRAND round 2: v = u ^ (u >> h1) (u from L4 in scr_sub), into buf_b
        lits = self._vop(OP_VRAND)
        _, _, h1, _ = vrand_constants(N)
        for i in range(N):
            g.gate(lits + [_half(S + i, 1)], {S + i: -0.5})
            g.gate(lits + [_half(S + i, 0)], {S + i: 0.5})
            if i < h1:
                g.gate(lits + [_full(Y + i, 1)], {S + i: 1.0})
                g.gate(lits + [_full(Y + i, 0)], {S + i: -1.0})
                continue
            for a, b in product((0, 1), repeat=2):
                g.gate(lits + [_full(Y + i, a), _full(Y + i - h1, b)], {S + i: 1.0 if a ^ b else -1.0})
        return g

    def _vector_l6_rows(self) -> _GatedRows:
        """L6 FFN: VLOGADD's addition, VRAND's third round."""
        cfg = self.cfg
        N = cfg.N
        v = self._vrows()
        X, S = v['mem'], v['scal']
        g = _GatedRows()
        xor = OP_VADDX - OP_VADD
        lits = self._vop(OP_VLOGADD, xor) + [_flag(v['valid'], 1)]
        self._sum_bits(g, lits, lambda j: _lin((1, _word(X, N, upto=j, full=True)),
                                               (1, _word(S, N, upto=j, full=True))), X)
        # VRAND round 3: w = K * v mod 2^N (v in buf_b), into scr_sub
        lits = self._vop(OP_VRAND)
        Y = v['y']
        _, K, _, _ = vrand_constants(N)
        for i in range(N):
            g.gate(lits + [_full(Y + i, 1)], {Y + i: -1.0})
            g.gate(lits + [_full(Y + i, 0)], {Y + i: 1.0})
        g.gate(lits, {Y + i: -1.0 for i in range(N)})
        for j in range(N):
            mod = 1 << (j + 1)
            P = ({S + N - 1 - p: 0.5 * ((K << p) % mod) for p in range(j + 1)},
                 0.5 * sum((K << p) % mod for p in range(j + 1)))
            self._mod_bit(g, lits, P, sum((K << p) % mod for p in range(j + 1)), j, Y + N - 1 - j)
        return g

    def _vector_l7_rows(self) -> _GatedRows:
        """L7 FFN: VRAND's last round and write; clear the payload from the
        memory columns' cmds rows (L8 does not clear cmds rows, and L1 reads
        them)."""
        cfg = self.cfg
        N = cfg.N
        v = self._vrows()
        X, Y = v['mem'], v['y']
        g = _GatedRows()
        # VRAND round 4: out = w ^ (w >> h2), written to memory in the block
        lits = self._vop(OP_VRAND) + [(v['mm'], -1.0, 1.0)]
        _, _, _, h2 = vrand_constants(N)
        for i in range(N):
            ins = list(dict.fromkeys([X + i, Y + i] + ([Y + i - h2] if i >= h2 else [])))
            for vals in product((0, 1), repeat=len(ins)):
                bit = dict(zip(ins, vals))
                w = bit[Y + i] ^ (bit[Y + i - h2] if i >= h2 else 0)
                if w != bit[X + i]:
                    g.gate(lits + [_full(r, b) for r, b in bit.items()], {X + i: 2.0 if w else -2.0})
        for r in range(3 * cfg.logn):
            for sgn in (1.0, -1.0):
                g.rows.append(({r: sgn, cfg.idx_memcol: 2.0}, -2.0, {r: -sgn}))
        return g


def init_state(cfg, memory, commands):
    """Initialize state tensor."""
    X = torch.zeros(cfg.d_model, cfg.n)
    X[-1, :cfg.s] = 1.0

    # Position encoding for columns 1 to n-1
    # Column zero has no position encoding.
    # Its address projection therefore contains PC alone.
    for col in range(1, cfg.n):
        X[cfg.idx_pos_enc:cfg.idx_pos_enc + cfg.logn, col] = to_bipolar(col, cfg.logn)

    for i, val in enumerate(memory):
        if i >= cfg.m:
            break
        col = cfg.s + i
        X[cfg.idx_memory:cfg.idx_memory + cfg.N, col] = signed_to_bipolar(val, cfg.N)

    # Store each memory column's relative index for content selection and VRAND.
    if hasattr(cfg, 'idx_tag'):
        for i in range(cfg.m):
            col = cfg.s + i
            X[cfg.idx_tag:cfg.idx_tag + cfg.N, col] = signed_to_bipolar(i, cfg.N)
    if getattr(cfg, 'vector', False):
        X[cfg.idx_memcol, cfg.s:cfg.s + cfg.m] = 1.0

    for i, (a, b, c) in enumerate(commands):
        col = cfg.s + cfg.m + i
        if col >= cfg.n:
            break
        X[:cfg.logn, col] = to_bipolar(a, cfg.logn)
        X[cfg.logn:2*cfg.logn, col] = to_bipolar(b, cfg.logn)
        X[2*cfg.logn:3*cfg.logn, col] = to_bipolar(c, cfg.logn)

    pc = cfg.s + cfg.m
    X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0] = to_bipolar(pc, cfg.logn)

    return X


def read_memory(X, cfg):
    """Read memory as integers."""
    memory = []
    for i in range(cfg.m):
        col = cfg.s + i
        bipolar = X[cfg.idx_memory:cfg.idx_memory + cfg.N, col]
        memory.append(signed_from_bipolar(bipolar))
    return memory


def get_pc(X, cfg):
    """Get PC value."""
    return from_bipolar(X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0])




if __name__ == "__main__":
    cfg = Config(s=32, m=8, n=64, N=8)
    ext = Computer(cfg)
    
    print("StrandedCPU transformer ready")
    print(f"Config: s={cfg.s}, m={cfg.m}, n={cfg.n}, N={cfg.N}, logn={cfg.logn}")
    print(f"d_model = {cfg.d_model}")
    print(f"Buffer rows = {cfg.nrows_buffer}")
