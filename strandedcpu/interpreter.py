"""Execute the ISA directly on a flat memory array.

Instruction operands use column addresses: memory word i is column s+i.
LOAD and STORE map pointers to s + (unsigned(p) mod s). XLAT instead ORs
those low pointer bits into its table's column address. Optional compiler
check metadata stops invalid arithmetic before an instruction executes.
"""

from typing import Dict, List, Optional, Tuple

from fractions import Fraction

import strandedcpu.reference as ref
from strandedcpu.compiler import DIV_MARKS, ExecutionFault
from strandedcpu.transformer import (
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC, OP_SHL, OP_SHR,
    OP_CMP, OP_LOAD, OP_AND, OP_OR, OP_XOR, OP_SUB, OP_FIND, OP_SWAP, OP_CMOV,
    OP_MULACC, OP_STORE, OP_MAX, OP_LOGADD, OP_XLAT, OP_SHIFT, OP_ADC, OP_MULSTEP,
    OP_DIVSTEP, OP_LFSR, OP_CLZ, OP_ABS, OP_ARGMAX,
    OP_JMPI, OP_FALIGN, OP_FNORM, OP_VSETVL, OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX,
    OP_VLOGADD, OP_VCADD, OP_VFILL, OP_VSHIFT, OP_VXLAT, OP_VREV, OP_VTRANS, OP_VRAND,
    OP_VPERMX, OP_VADDX, OP_VBFLY, OP_VMAXX, OP_VLOGADDX, VECTOR_OPCODES, PAGE2_OPCODES,
)


class ISAInterpreter:
    def __init__(self, s: int, m: int, N: int, memory: List[int],
                 commands: List[Tuple[int, int, int]], log_frac_bits: int = 2,
                 checks: Optional[Dict[int, tuple]] = None):
        """Initialize memory and optional compile_c meta['checks'].

        Checks raise ExecutionFault before the instruction on overflow or invalid
        inputs. Integer division remainders and fix rounding update self.inexact;
        modulo operations do not increment the division-remainder count.
        """
        self.s, self.m, self.N = s, m, N
        self.log_frac_bits = log_frac_bits
        self.memory = [ref.wrap(v, N) for v in memory] + [0] * (m - len(memory))
        self.commands = commands
        self.pc = s + m
        self.carry = 0
        self.vl_log = 0                 # VSETVL
        self.vcarry = [0] * m           # per memory word (vector carry)
        self.checks = checks or {}
        self.inexact = {'int_div_remainder': 0, 'fix_rounding': 0}
        self.executed = 0

    def get(self, addr: int) -> int:
        i = addr - self.s
        return self.memory[i] if 0 <= i < self.m else 0

    def set(self, addr: int, value: int) -> None:
        i = addr - self.s
        if 0 <= i < self.m:
            self.memory[i] = ref.wrap(value, self.N)

    def pointer(self, p: int) -> int:
        return self.s + ref.unsigned(p, self.N) % self.s

    def step(self) -> bool:
        """Execute one instruction. Returns True once halted."""
        i = self.pc - self.s - self.m
        if self.pc == 0 or not 0 <= i < len(self.commands):
            self.pc = 0
            return True
        a, b, c = self.commands[i]
        N = self.N
        if i in self.checks:
            self._check(self.checks[i])
        self.executed += 1
        self.pc += 1
        vb, vc = self.get(b), self.get(c)

        if a >= 32 and a not in PAGE2_OPCODES:     # SUBLEQ
            r = ref.wrap(vb - self.get(a), N)
            self.set(b, r)
            if r <= 0:
                self.pc = c
        elif a == OP_HALT:
            self.pc = 0
            return True
        elif a == OP_MOV:
            self.set(b, vc)
        elif a == OP_ADD:
            self.carry = ref.add_carry(vb, vc, N)
            self.set(b, vb + vc)
        elif a == OP_SUB:
            self.carry = ref.sub_carry(vb, vc, N)
            self.set(b, vb - vc)
        elif a == OP_JMP:
            self.pc = c
        elif a == OP_JZ:
            if vb == 0:
                self.pc = c
        elif a == OP_JNZ:
            if vb != 0:
                self.pc = c
        elif a == OP_CMP:
            if vb < 0:
                self.pc = c
        elif a == OP_INC:
            self.set(b, vb + 1)
        elif a == OP_DEC:
            self.set(b, vb - 1)
        elif a == OP_SHL:
            self.set(b, vb << 1)
        elif a == OP_SHR:
            self.set(b, vb >> 1)
        elif a == OP_LOAD:
            self.set(b, self.get(self.pointer(vc)))
        elif a == OP_STORE:
            self.set(self.pointer(vc), vb)
        elif a == OP_AND:
            self.set(b, vb & vc)
        elif a == OP_OR:
            self.set(b, vb | vc)
        elif a == OP_XOR:
            self.set(b, vb ^ vc)
        elif a == OP_FIND:
            self.set(b, next((j for j, v in enumerate(self.memory) if v == vc), 0))
        elif a == OP_SWAP:
            self.set(b, vc)
            self.set(c, vb)
        elif a == OP_CMOV:
            if vb < 0:
                self.set(b, vc)
        elif a == OP_MULACC:
            self.set(b, (vb << 1) + (vc if vb < 0 else 0))
        elif a == OP_MAX:
            self.set(b, ref.max_(vb, vc))
        elif a == OP_LOGADD:
            self.set(b, ref.logadd(vb, vc, N, self.log_frac_bits))
        elif a == OP_XLAT:
            self.set(b, self.get(ref.xlat_address(vb, c, self.s, N)))
        elif a == OP_SHIFT:
            self.set(b, ref.shift(vb, vc, N))
        elif a == OP_ADC:
            r, self.carry = ref.adc(vb, vc, self.carry, N)
            self.set(b, r)
        elif a == OP_MULSTEP:
            hi, lo = ref.mulstep(vb, self.get(b ^ 1), vc, N)
            self.set(b, hi)
            self.set(b ^ 1, lo)
        elif a == OP_DIVSTEP:
            rem, quo = ref.divstep(vb, self.get(b ^ 1), vc, N)
            self.set(b, rem)
            self.set(b ^ 1, quo)
        elif a == OP_LFSR:
            self.set(b, ref.lfsr(vb, vc, N))
        elif a == OP_CLZ:
            self.set(b, ref.clz(vc, N))
        elif a == OP_ABS:
            self.set(b, ref.abs_(vc, N))
        elif a == OP_ARGMAX:
            self.set(b, ref.argmax([self.get(c + j) for j in range(1 << vb)]))
        elif a == OP_JMPI:
            self.pc = self.s + self.m + ref.unsigned(vc, N)
        elif a == OP_FALIGN:
            e, m = ref.falign(vb, self.get(b ^ 1), vc, N)
            self.set(b, e)
            self.set(b ^ 1, m)
        elif a == OP_FNORM:
            e, m = ref.fnorm(vb, self.get(b ^ 1), N)
            self.set(b, e)
            self.set(b ^ 1, m)
        elif a == OP_VSETVL:
            self.vl_log = vb
        elif a in VECTOR_OPCODES:
            self.vector(a, b, c, vc)
        else:
            raise ValueError(f"unknown opcode {a}")
        return False

    def _check(self, check: tuple) -> None:
        kind, *slots = check
        N = self.N
        word = lambda slot: self.memory[slot]
        lo, hi = -(1 << (N - 1)), (1 << (N - 1)) - 1

        def fix(h, l):
            return Fraction(word(h) * (1 << N) + ref.unsigned(word(l), N), 1 << N)

        def need(ok, reason, detail):
            if not ok:
                raise ExecutionFault(reason, detail)

        if kind in ('add', 'sub', 'mul'):
            x, y = word(slots[0]), word(slots[1])
            v = x + y if kind == 'add' else x - y if kind == 'sub' else x * y
            need(lo <= v <= hi, 'overflow', f"{kind}: {x} and {y} give {v}")
        elif kind in ('div', 'mod'):
            x, y = word(slots[0]), word(slots[1])
            if len(slots) > 2 and y == 0:           # a division the root runtime makes to stop
                raise ExecutionFault(*DIV_MARKS[slots[2]])
            need(y != 0, 'runtime_error', 'division_by_zero')
            need(not (x == lo and y == -1), 'overflow', f"{kind}: {x} / {y}")
            if kind == 'div' and x % y:     # a remainder (%) is exact: not counted
                self.inexact['int_div_remainder'] += 1
        elif kind in ('shl', 'shr'):
            x, k = word(slots[0]), word(slots[1])
            need(0 <= k < N, 'runtime_error', f"shift_amount: {k}")
            if kind == 'shl':
                need(lo <= x * (1 << k) <= hi, 'overflow', f"shl: {x} << {k}")
        elif kind == 'abs':
            need(word(slots[0]) != lo, 'overflow', f"abs: {word(slots[0])}")
        elif kind == 'bounds':
            ptr, base, size = word(slots[0]), slots[1], slots[2]
            need(base <= ptr < base + size, 'runtime_error', f"index_out_of_bounds: {ptr - base} of {size}")
        elif kind in ('fadd', 'fsub', 'fmul', 'fdiv', 'fdivi', 'fneg'):
            top = Fraction(1 << (N - 1))
            a = fix(slots[0], slots[1])
            if kind == 'fneg':
                v = -a
            elif kind == 'fdivi':
                k = word(slots[2])
                need(k != 0, 'runtime_error', 'division_by_zero')
                v = a / k
            else:
                b = fix(slots[2], slots[3])
                if kind == 'fdiv':
                    need(b != 0, 'runtime_error', 'division_by_zero')
                v = a + b if kind == 'fadd' else a - b if kind == 'fsub' else a * b if kind == 'fmul' else a / b
            need(-top <= v < top, 'overflow', f"{kind}: {float(v)}")
            if (v * (1 << N)).denominator != 1:
                self.inexact['fix_rounding'] += 1
        elif kind == 'rcast':       # rational (n, d) to int: drops a fraction unless d divides n
            if word(slots[1]) and word(slots[0]) % word(slots[1]):
                self.inexact['fix_rounding'] += 1
        elif kind == 'fcast':
            if ref.unsigned(word(slots[1]), N):
                self.inexact['fix_rounding'] += 1
        else:
            raise ValueError(f"unknown check {kind}")

    def vector(self, a: int, b: int, c: int, scalar: int) -> None:
        """One vector instruction over the aligned VL-word blocks at b and c.
        Every operand is read before any element is written."""
        N, t = self.N, self.vl_log
        vl = 1 << t
        bb, cb = b & ~(vl - 1), c & ~(vl - 1)
        old = list(self.memory)
        oldc = list(self.vcarry)
        get = lambda col: old[col - self.s] if 0 <= col - self.s < self.m else 0
        carry_at = lambda col: oldc[col - self.s] if 0 <= col - self.s < self.m else 0
        m = ref.unsigned(scalar, N) & (vl - 1)
        for i in range(vl):
            col = bb + i
            j = col - self.s
            if not 0 <= j < self.m:
                continue
            x = old[j]
            if a in (OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX, OP_VLOGADD):
                y = get(cb + i)
            elif a in (OP_VPERMX, OP_VADDX, OP_VBFLY, OP_VMAXX, OP_VLOGADDX):
                y = get(bb + (i ^ m))
            if a in (OP_VMOV, OP_VPERMX):
                r = y
            elif a in (OP_VADD, OP_VADDX):
                r = x + y
                self.vcarry[j] = ref.add_carry(x, y, N)
            elif a == OP_VSUB:
                r = x - y
                self.vcarry[j] = ref.sub_carry(x, y, N)
            elif a in (OP_VMAX, OP_VMAXX):
                r = max(x, y)
            elif a in (OP_VLOGADD, OP_VLOGADDX):
                r = ref.logadd(x, y, N, self.log_frac_bits)
            elif a == OP_VCADD:
                k = carry_at(cb + i)
                r = x + k
                self.vcarry[j] = (ref.unsigned(x, N) + k) >> N
            elif a == OP_VBFLY:
                r = y - x if (col & ref.unsigned(scalar, N)) else x + y
            elif a == OP_VFILL:
                r = scalar
            elif a == OP_VSHIFT:
                r = ref.shift(x, scalar, N)
            elif a == OP_VXLAT:
                r = get(ref.xlat_address(x, c, self.s, N))
            elif a == OP_VREV:
                r = get(cb + ref.bitrev(i, t))
            elif a == OP_VTRANS:
                r = get(cb + ref.transpose_index(i, t))
            elif a == OP_VRAND:
                r = ref.vrand(scalar, j, N)
            self.memory[j] = ref.wrap(r, N)

    def run(self, max_steps: int = 1_000_000) -> int:
        """Run until step() reports halt; return the number of step calls.

        The final call can detect an already halted or out-of-program PC without
        executing an instruction. self.executed counts fetched instructions.
        """
        for steps in range(max_steps):
            if self.step():
                return steps + 1
        raise RuntimeError(f"no HALT within {max_steps} steps")
