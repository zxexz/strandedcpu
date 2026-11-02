"""Compile and run C programs, or compare their results with answer options.

    from strandedcpu.api import run_function
    out = run_function("int x; int y; int main() { y = x * x; return y; }", {"x": 7})
    out["y"], out["__retval"]          # 49, 49

run_function uses the ISA interpreter unless backend="transformer" is set.
Inputs populate global memory after compilation; declare input globals without
initializers so the program does not overwrite them. Arrays accept lists.
Use score_options for checked arithmetic and option matching.
"""

from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Optional, Union

from strandedcpu.compiler import ExecutionFault, compile_c, fix_value
from strandedcpu.interpreter import ISAInterpreter

Value = Union[int, float, List[int]]


def _store(memory: List[int], addr: int, value: int, N: int) -> None:
    value &= (1 << N) - 1
    memory[addr] = value - (1 << N) if value >= 1 << (N - 1) else value


@lru_cache(maxsize=1024)
def _compiled(source: str, s: int, m: int, N: int, numbers: str = 'binary'):
    """compile_c is deterministic; callers copy the memory list before writing."""
    cfg, memory, commands, meta = compile_c(source, s=s, m=m, n=4096, N=N, numbers=numbers)
    return cfg, tuple(memory), tuple(commands), meta


def _fix_in(memory: List[int], addr: int, value, N: int, numbers: str) -> bool:
    """Store a number into the fix pair at addr; False if it does not fit."""
    q = Fraction(repr(value)) if isinstance(value, float) else Fraction(value)
    if numbers == 'rational':
        if max(abs(q.numerator), q.denominator) >= 1 << (N - 1):
            return False
        memory[addr], memory[addr + 1] = q.numerator, q.denominator
        return True
    raw = q * (1 << N)
    if raw.denominator != 1 or not -(1 << (2 * N - 1)) <= raw < 1 << (2 * N - 1):
        return False
    raw = int(raw)
    _store(memory, addr, raw >> N, N)
    _store(memory, addr + 1, raw & ((1 << N) - 1), N)
    return True


def _fix_out(final: List[int], addr: int, N: int, numbers: str) -> Fraction:
    if numbers == 'rational':
        return Fraction(final[addr], final[addr + 1])
    return Fraction(final[addr]) + Fraction(final[addr + 1] & ((1 << N) - 1), 1 << N)


@lru_cache(maxsize=8)
def _transformer(s: int, m: int, n: int, N: int, device: str = 'cpu'):
    from strandedcpu.transformer import Config, Computer
    cfg = Config(s=s, m=m, n=n, N=N)
    return cfg, Computer(cfg).to(device)


def _transformer_graph_run(comp, tcfg, X, max_steps: int, chunk: int = 64):
    """Run the transformer on the GPU as one captured CUDA graph per step.
    A halted state (PC = 0) is held fixed by a mask on the GPU, so the PC is
    read back only every chunk steps. Returns (state, steps executed)."""
    import torch
    import strandedcpu.subleq
    pc = slice(tcfg.idx_pc, tcfg.idx_pc + tcfg.logn)
    state = X.clone()
    steps = torch.zeros((), dtype=torch.int64, device=X.device)

    def body():
        halted = (state[pc, 0] < 0).all()
        nxt = comp.step(state)
        state.copy_(torch.where(halted, state, nxt))
        steps.add_((~halted).to(torch.int64))

    strandedcpu.subleq.SKIP_EMPTY_HEADS = False
    try:
        with torch.no_grad():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                body()                                   # warm-up, required before capture
            torch.cuda.current_stream().wait_stream(side)
            state.copy_(X)
            steps.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            state.copy_(X)
            steps.zero_()
            while True:
                for _ in range(chunk):
                    graph.replay()
                if bool((state[pc, 0] < 0).all()):
                    return state, int(steps)
                if int(steps) >= max_steps:
                    raise RuntimeError(f"no HALT within {max_steps} steps")
    finally:
        strandedcpu.subleq.SKIP_EMPTY_HEADS = True


def run_function(source: str, inputs: Optional[Dict[str, Value]] = None, N: int = 16,
                 backend: str = "interpreter", s: int = 64, m: int = 128,
                 max_steps: int = 1_000_000, trace: bool = False,
                 numbers: str = 'binary', device: str = 'cpu') -> Dict[str, object]:
    """Compile C and return named variables, __retval and __steps.

    Integers return as ints, arrays as lists, and fix values as floats or exact
    Fractions in rational mode. Compilation is cached by source, configuration
    and number mode; transformer weights are cached by shape and device.

    With trace=True, __trace records each instruction's PC, opcode, b/c operands
    and changed memory words. Memory addresses in trace entries are relative to
    the start of memory. This entry point runs without compiler runtime checks;
    use score_options to stop on checked arithmetic faults.
    """
    cfg, memory, commands, meta = _compiled(source, s, m, N, numbers)
    memory = list(memory)
    slots = {k: v for k, v in meta['variables'].items()}
    fixes = meta['fix_variables']
    sizes = meta.get('sizes', {})
    for name, value in (inputs or {}).items():
        if name not in slots:
            raise KeyError(f"no global variable '{name}'")
        addr = slots[name]
        if name in fixes:
            if numbers == 'rational':
                if not _fix_in(memory, addr, value, N, numbers):
                    raise ValueError(f"{name} = {value} does not fit {N}-bit numerator and denominator")
            else:
                raw = round(float(value) * (1 << N))
                _store(memory, addr, raw >> N, N)
                _store(memory, addr + 1, raw & ((1 << N) - 1), N)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _store(memory, addr + i, int(v), N)
        else:
            _store(memory, addr, int(value), N)

    steps_trace = [] if trace else None
    if backend == "interpreter":
        vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, memory, commands, cfg.log_frac_bits)
        steps = 0
        while steps < max_steps:
            pc = vm.pc
            before = list(vm.memory) if trace else None
            halted = vm.step()
            steps += 1
            if trace:
                i = pc - cfg.s - cfg.m
                a, b, c = commands[i] if 0 <= i < len(commands) else (0, 0, 0)
                steps_trace.append({
                    'pc': pc, 'op': a, 'b': b - cfg.s, 'c': c - cfg.s,
                    'writes': {j: (before[j], vm.memory[j]) for j in range(cfg.m) if before[j] != vm.memory[j]},
                })
            if halted:
                break
        else:
            raise RuntimeError(f"no HALT within {max_steps} steps")
        final = vm.memory
    elif backend == "transformer":
        import torch
        from strandedcpu.transformer import from_bipolar, get_pc, init_state, read_memory
        n = 128
        while n < cfg.s + cfg.m + len(commands):
            n *= 2
        tcfg, comp = _transformer(cfg.s, cfg.m, n, N, device)
        X = init_state(tcfg, memory, list(commands)).to(device)
        if device != 'cpu':     # read the PC and memory from a host copy, not element by element
            gpu_read = read_memory
            get_pc = lambda X, c: from_bipolar(X[c.idx_pc:c.idx_pc + c.logn, 0].cpu())
            read_memory = lambda X, c: gpu_read(X.cpu(), c)
        steps = 0
        before = memory
        if device != 'cpu' and not trace:      # CUDA graph, halt checked every few dozen steps
            X, steps = _transformer_graph_run(comp, tcfg, X, max_steps)
        with torch.no_grad():
            while not (device != 'cpu' and not trace) and get_pc(X, tcfg) != 0:
                if steps >= max_steps:
                    raise RuntimeError(f"no HALT within {max_steps} steps")
                pc = get_pc(X, tcfg)
                X = comp.step(X)
                steps += 1
                if trace:
                    # read back from the state tensor, not the interpreter
                    after = read_memory(X, tcfg)
                    i = pc - cfg.s - cfg.m
                    a, b, c = commands[i] if 0 <= i < len(commands) else (0, 0, 0)
                    steps_trace.append({
                        'pc': pc, 'op': a, 'b': b - cfg.s, 'c': c - cfg.s,
                        'writes': {j: (before[j], after[j]) for j in range(cfg.m) if before[j] != after[j]},
                    })
                    before = after
        final = read_memory(X, tcfg)
    else:
        raise ValueError(f"unknown backend '{backend}'")

    out: Dict[str, object] = {}
    for name, addr in slots.items():
        if name in fixes:
            out[name] = (_fix_out(final, addr, N, numbers) if numbers == 'rational'
                         else fix_value(final[addr], final[addr + 1], N))
        elif sizes.get(name, 1) > 1:
            out[name] = final[addr:addr + sizes[name]]
        else:
            out[name] = final[addr]
    out['__retval'] = final[meta['retval_addr']]
    out['__steps'] = steps
    if trace:
        out['__trace'] = steps_trace
    return out


# Option scoring for decision models

import re
from fractions import Fraction

_NUMBER = re.compile(r"""
    \s*
    (?P<sign>[+-])?
    \s*
    (?P<cur>\$)?
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)       # 1200 or 1,200 (well-formed groups only)
    (?:\.(?P<frac>\d+))?
    (?:\s*/\s*(?P<den>\d+))?               # a/b
    \s*$
""", re.VERBOSE)


def parse_option_number(text: str) -> Optional[Union[int, Fraction]]:
    """Parse a signed number, currency amount, decimal or fraction; return None otherwise.

    Thousands separators must form groups of three. A non-integral value returns
    as Fraction. Units, percentages, zero denominators and decimal numerators in
    fraction notation are rejected.
    """
    if not isinstance(text, str):
        return None
    m = _NUMBER.fullmatch(text)
    if m is None:
        return None
    value = Fraction(int(m['int'].replace(',', '')))
    if m['frac'] is not None:
        if m['den'] is not None:
            return None
        value += Fraction(int(m['frac']), 10 ** len(m['frac']))
    if m['den'] is not None:
        if int(m['den']) == 0 or ',' in m['int']:
            return None
        value /= int(m['den'])
    if m['sign'] == '-':
        value = -value
    return int(value) if value.denominator == 1 else value


_TRUTH = {'true': 1, 'yes': 1, 'false': 0, 'no': 0}


def parse_option_truth(text: str) -> Optional[int]:
    """1 for True/Yes, 0 for False/No (any case, an optional final '.'), else None."""
    if not isinstance(text, str):
        return None
    return _TRUTH.get(text.strip().rstrip('.').strip().lower())


def _repository_revision():
    """(HEAD, sha256 of the uncommitted state or None). The hash covers
    `git diff HEAD` plus the path and bytes of every untracked file git does
    not ignore, so a dirty stamp can be reproduced."""
    import hashlib
    import os
    import subprocess
    package_dir = os.path.dirname(os.path.realpath(__file__))
    here = os.path.dirname(package_dir)
    git = lambda *args: subprocess.run(['git', '-C', here, *args], capture_output=True,
                                      timeout=10, check=True)
    try:
        # Installed wheels can live inside another checkout. Only stamp the
        # repository whose source tree contains this package at its root.
        repo = os.path.realpath(git('rev-parse', '--show-toplevel').stdout.decode().strip())
        if repo != here or os.path.basename(package_dir) != 'strandedcpu':
            return 'unknown', None
        head = git('rev-parse', 'HEAD').stdout.decode().strip() or 'unknown'
        diff = git('diff', 'HEAD', '--binary').stdout
        untracked = git('ls-files', '--others', '--exclude-standard', '-z').stdout.split(b'\0')
    except (OSError, subprocess.SubprocessError):
        return 'unknown', None
    untracked = sorted(u for u in untracked if u)
    if not diff and not untracked:
        return head, None
    h = hashlib.sha256(diff)
    try:
        for path in untracked:
            h.update(b'\0' + path + b'\0')
            with open(os.path.join(repo, os.fsdecode(path)), 'rb') as f:
                h.update(f.read())
    except OSError:
        # A file can change or disappear while a working tree is being edited.
        return head + '-dirty', None
    return head + '-dirty', h.hexdigest()


STRANDEDCPU_COMMIT, STRANDEDCPU_DIFF_SHA256 = _repository_revision()


# Worker results populate the caller's compilation and packed-program caches.
_PREPARED: Dict[tuple, tuple] = {}
_PREPARED_PROGRAM: Dict[tuple, object] = {}


@lru_cache(maxsize=1024)
def _compiled_checked(source: str, s: int, m: int, N: int, numbers: str = 'binary'):
    hit = _PREPARED.pop((source, s, m, N, numbers), None)
    if hit is not None:
        return hit
    cfg, memory, commands, meta = compile_c(source, s=s, m=m, n=4096, N=N, checked=True, numbers=numbers)
    return cfg, tuple(memory), tuple(commands), meta


def _prepare_remote(kind: str, src: str, entry: Optional[str], s: int, m: int, N: int, numbers: str, pack: bool):
    """Translate, compile and optionally pack one job in a worker process.

    Return enough information to retry failures in the caller, where the normal
    compile and translation error handling runs.
    """
    c_src = src
    if kind == 'python':
        from strandedcpu.python import PythonSubsetError, python_to_c
        try:
            c_src = python_to_c(src, entry)[0]
        except PythonSubsetError:
            return 'redo', None, None, None
    try:
        compiled = _compiled_checked(c_src, s, m, N, numbers)
    except Exception:
        return 'redo', c_src, None, None
    program = None
    if pack:
        import strandedcpu.cuda
        program = strandedcpu.cuda.Program(compiled[2], compiled[3]['checks'])
    return 'ok', c_src, compiled, program


_POOL = None


def precompile(jobs: List[tuple], workers: int, pack: bool) -> List[Optional[str]]:
    """Compile (kind, src, entry, s, m, N, numbers) jobs on worker processes;
    the results go to this process's compile caches. Returns each job's C
    source (None where the caller must translate it itself)."""
    global _POOL
    import concurrent.futures
    import multiprocessing
    if _POOL is None or _POOL._max_workers != workers:
        _POOL = concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('forkserver'))
    out: List[Optional[str]] = []
    for job, (status, c_src, compiled, program) in zip(jobs, _POOL.map(_prepare_remote, *zip(*[(*j, pack) for j in jobs]))):
        kind, src, entry, s, m, N, numbers = job
        out.append(c_src)
        if status == 'ok':
            _PREPARED[(c_src, s, m, N, numbers)] = compiled
            if program is not None:
                _PREPARED_PROGRAM[(c_src, s, m, N, numbers)] = program
    return out


def score_options(program: str, inputs: Dict[str, Union[int, float]], option_texts: Dict[str, str], *,
                  N: int, result: str = "return", backend: str = "interpreter",
                  s: int = 64, m: int = 128, max_steps: int = 1_000_000,
                  rounding_slack: Optional[Union[int, float, str, Fraction]] = None,
                  numbers: str = 'binary', boolean_options: bool = False) -> Dict[str, object]:
    """Run checked C and compare its numeric result with each answer option.

    Return status="ok" for one matching option, otherwise "abstain" with an
    abstain_reason. Arithmetic faults stop before the affected instruction.
    Modular library operations such as umulhi and qdiv retain their ISA semantics.
    Integer division remainders and fix rounding are counted in inexact_ops;
    they do not by themselves cause abstention.

    inputs sets uninitialized globals. result selects main's integer return
    value or a named global. numbers="rational" stores fix values as reduced
    fractions; binary mode uses N fraction bits. Float literals first pass
    through float parsing, then rationalize the parsed float's repr. Digits
    lost during parsing are not recovered. Rational numerators and denominators
    must fit the word size, including intermediate calculations.
    rounding_slack permits approximate option matches only after fix rounding;
    multiple matches still abstain. boolean_options accepts True/False and
    Yes/No when the result is exactly 0 or 1.

    The default backend is the interpreter. "auto" attempts CUDA and records a
    fallback reason when needed. "transformer" and "cuda-transformer" also
    compare with the analytical transformer, at N <= 16. CUDA supports words up
    to 64 bits; binary fix above N=32 falls back to the interpreter.

    See docs/reference/api.md for result fields and backend parameters.
    """
    backend, why = _resolve_backend(backend)
    job = _Scoring(program, inputs, option_texts, N=N, result=result, backend=backend, s=s, m=m,
                   max_steps=max_steps, rounding_slack=rounding_slack, numbers=numbers,
                   boolean_options=boolean_options)
    if job.done:
        return _note(job.out, why)
    if job.backend in ('cuda', 'cuda-transformer'):
        return _note(_run_cuda([job])[0], why)
    if job.backend not in ('interpreter', 'transformer'):
        raise ValueError(f"unknown backend '{job.backend}'")
    return _note(job.execute(), why)


_AUTO: Optional[tuple] = None


def _resolve_backend(backend: str) -> tuple:
    """'auto' is 'cuda' when a GPU is present and the kernel builds, else
    'interpreter' with the reason (reported as fallback_reason)."""
    global _AUTO
    if backend != 'auto':
        return backend, None
    if _AUTO is None:
        try:
            import strandedcpu.cuda
            if not strandedcpu.cuda.available():
                raise RuntimeError("no CUDA device")
            strandedcpu.cuda._module()
            _AUTO = ('cuda', None)
        except Exception as exc:
            _AUTO = ('interpreter', f"{type(exc).__name__}: {str(exc)[:300]}")
    return _AUTO


def _note(result: Dict[str, object], why: Optional[str]) -> Dict[str, object]:
    if why is not None:
        result['fallback_reason'] = why
    return result


def score_options_batch(calls: List[Dict[str, object]], **common) -> List[Dict[str, object]]:
    """Score calls in input order; each dict overrides the common keyword arguments.

    CUDA jobs are grouped by configuration and instruction limit for batched
    execution. compile_workers > 1 prepares programs in worker processes before
    execution; compilation caches remain in the calling process.
    """
    workers = common.pop('compile_workers', 0)
    common['backend'], why = _resolve_backend(common.get('backend', 'interpreter'))
    if workers and workers > 1 and len(calls) > 1:
        keys = []
        for call in calls:
            kw = {**common, **call}
            keys.append(('c', kw['program'], None, kw.get('s', 64), kw.get('m', 128), kw['N'],
                         kw.get('numbers', 'binary')))
        precompile(keys, workers, pack=common.get('backend') == 'cuda')
    jobs = []
    for call in calls:
        kw = {**common, **call}
        program, inputs, options = kw.pop('program'), kw.pop('inputs', {}), kw.pop('option_texts')
        jobs.append(_Scoring(program, inputs or {}, options, **kw))
    results = [job.out if job.done else None for job in jobs]
    cuda = [k for k, job in enumerate(jobs) if not job.done and job.backend in ('cuda', 'cuda-transformer')]
    for k, r in zip(cuda, _run_cuda([jobs[k] for k in cuda])):
        results[k] = r
    for k, job in enumerate(jobs):
        if results[k] is None:
            if job.backend not in ('interpreter', 'transformer'):
                raise ValueError(f"unknown backend '{job.backend}'")
            results[k] = job.execute()
    return [_note(r, why) for r in results]


@lru_cache(maxsize=1024)
def _cuda_program(source: str, s: int, m: int, N: int, numbers: str):
    import strandedcpu.cuda
    hit = _PREPARED_PROGRAM.pop((source, s, m, N, numbers), None)
    if hit is not None:
        return hit
    cfg, memory, commands, meta = _compiled_checked(source, s, m, N, numbers)
    return strandedcpu.cuda.Program(commands, meta['checks'])


def _run_cuda(jobs: List['_Scoring']) -> List[Dict[str, object]]:
    """Run prepared jobs on the GPU, one launch per (s, m, N, max_steps)."""
    import strandedcpu.cuda
    groups: Dict[tuple, List[int]] = {}
    for k, job in enumerate(jobs):
        groups.setdefault((job.cfg.s, job.cfg.m, job.cfg.N, job.max_steps, job.cfg.log_frac_bits), []).append(k)
    results: List[Optional[Dict[str, object]]] = [None] * len(jobs)
    for (s, m, N, max_steps, F), ks in groups.items():
        runs = strandedcpu.cuda.run_batch(s, m, N, [(jobs[k].memory, _cuda_program(jobs[k].program, jobs[k].s, jobs[k].m,
                                                                              N, jobs[k].numbers)) for k in ks],
                                   max_steps=max_steps, log_frac_bits=F)
        for k, run in zip(ks, runs):
            job = jobs[k]
            if run['status'] == 'fault':
                job.out['steps'] = run['steps']
                results[k] = job.finish(run['fault'].reason, run['fault'].detail)
            elif run['status'] == 'step_limit':
                job.out['steps'] = max_steps
                results[k] = job.finish('step_limit', f"no HALT within {max_steps} steps")
            else:
                results[k] = job.complete(run['memory'], run['steps'], run['inexact'])
    return results


class _Scoring:
    """Prepare a scoring call, execute it, then match its result against the options.

    Preparation can finish the call early on invalid inputs or compilation errors.
    """

    def __init__(self, program: str, inputs, option_texts, *, N: int, result: str = "return",
                 backend: str = "interpreter", s: int = 64, m: int = 128, max_steps: int = 1_000_000,
                 rounding_slack=None, numbers: str = 'binary', boolean_options: bool = False):
        import hashlib
        import time
        if backend == 'cuda-transformer' and N > 16:
            raise ValueError("the transformer is exact only up to N=16")
        fallback = None
        if backend == 'cuda' and N > 32 and numbers != 'rational':
            backend, fallback = 'interpreter', "strandedcpu.cuda checks binary fix arithmetic only up to N=32"
        self.program, self.inputs, self.option_texts = program, inputs, option_texts
        self.N, self.result, self.backend, self.s, self.m = N, result, backend, s, m
        self.max_steps, self.rounding_slack, self.numbers = max_steps, rounding_slack, numbers
        self.done = True
        self.out: Dict[str, object] = {
            'status': 'abstain', 'abstain_reason': None, 'fault_detail': None, 'value': None,
            'value_raw': None, 'parsed': {}, 'match': {}, 'match_kind': None, 'steps': 0, 'latency_s': 0.0,
            'inexact_ops': {'int_div_remainder': 0, 'fix_rounding': 0},
            'backend': backend, 'N': N, 'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
            'strandedcpu_commit': STRANDEDCPU_COMMIT, 'strandedcpu_diff_sha256': STRANDEDCPU_DIFF_SHA256,
        }
        if fallback:
            self.out['fallback_reason'] = fallback
        self.t0 = time.perf_counter()
        out = self.out

        parsed = {key: parse_option_number(text) for key, text in option_texts.items()}
        truth = {key: parse_option_truth(text) for key, text in option_texts.items()}
        self.booleans = boolean_options and bool(truth) and all(v is not None for v in truth.values())
        if self.booleans:
            parsed = truth
        self.parsed = parsed
        out['parsed'] = {k: (None if v is None else int(v) if isinstance(v, int) else float(v)) for k, v in parsed.items()}

        try:
            self.cfg, memory, self.commands, self.meta = _compiled_checked(program, s, m, N, numbers)
        except ExecutionFault as fault:
            self.finish(fault.reason, fault.detail)
            return
        except (SyntaxError, NameError, KeyError, ValueError, MemoryError, RuntimeError) as exc:
            self.finish('compile_error', f"{type(exc).__name__}: {exc}")
            return

        memory = list(memory)
        variables, fixes = self.meta['variables'], self.meta['fix_variables']
        lo, hi = -(1 << (N - 1)), (1 << (N - 1)) - 1
        for name, value in inputs.items():
            if name not in variables or (name not in fixes and self.meta['sizes'].get(name, 1) != 1):
                self.finish('compile_error', f"no scalar global '{name}'")
                return
            addr = variables[name]
            if name in fixes:
                if not _fix_in(memory, addr, value, N, numbers):
                    self.finish('input_out_of_domain', f"{name} = {value}")
                    return
            else:
                if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                    self.finish('input_out_of_domain', f"{name} = {value!r}")
                    return
                memory[addr] = value
        if result != 'return' and result not in variables:
            self.finish('compile_error', f"no global '{result}'")
            return
        self.memory = memory
        self.done = False

    def finish(self, reason=None, detail=None) -> Dict[str, object]:
        import time
        self.out['abstain_reason'] = reason
        self.out['fault_detail'] = detail
        self.out['status'] = 'ok' if reason is None else 'abstain'
        self.out['latency_s'] = time.perf_counter() - self.t0
        self.done = True
        return self.out

    def execute(self) -> Dict[str, object]:
        """Run on the interpreter (and, for backend='transformer', check it)."""
        cfg, max_steps = self.cfg, self.max_steps
        vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, self.memory, list(self.commands), cfg.log_frac_bits,
                            checks=self.meta['checks'])
        try:
            for steps in range(max_steps):
                if vm.step():
                    break
            else:
                self.out['steps'] = max_steps
                return self.finish('step_limit', f"no HALT within {max_steps} steps")
        except ExecutionFault as fault:
            self.out['steps'] = vm.executed
            return self.finish(fault.reason, fault.detail)
        return self.complete(vm.memory, steps + 1, vm.inexact)

    def complete(self, final, steps: int, inexact: Dict[str, int]) -> Dict[str, object]:
        out, N, numbers, meta = self.out, self.N, self.numbers, self.meta
        variables, fixes, result = meta['variables'], meta['fix_variables'], self.result
        parsed, option_texts = self.parsed, self.option_texts
        final = [int(v) for v in final]
        out['steps'] = steps
        out['inexact_ops'] = {'int_div_remainder': inexact['int_div_remainder'],
                              'fix_rounding': inexact['fix_rounding'] + meta['fix_literal_rounding']}

        if self.backend in ('transformer', 'cuda-transformer'):
            hw = run_function(self.program, self.inputs, N=N, backend='transformer', s=self.s, m=self.m,
                              max_steps=self.max_steps, numbers=numbers,
                              device='cuda' if self.backend == 'cuda-transformer' else 'cpu')
            names = [k for k in variables if not k.startswith('__')]
            if any(hw[k] != ((_fix_out(final, variables[k], N, numbers) if numbers == 'rational'
                              else fix_value(final[variables[k]], final[variables[k] + 1], N)) if k in fixes
                             else final[variables[k]:variables[k] + meta['sizes'][k]] if meta['sizes'].get(k, 1) > 1
                             else final[variables[k]]) for k in names):
                raise RuntimeError("transformer and interpreter disagree")

        if result == 'return':
            value = Fraction(final[meta['retval_addr']])
        elif result in fixes:
            a = variables[result]
            out['value_raw'] = (final[a], final[a + 1])
            value = _fix_out(final, a, N, numbers)
        else:
            value = Fraction(final[variables[result]])
        out['value'] = int(value) if value.denominator == 1 else float(value)

        if self.booleans and value not in (0, 1):
            out['match'] = {key: False for key in parsed}
            return self.finish('no_match', f"result {out['value']} is not a truth value (0 or 1)")
        if any(v is None for v in parsed.values()):
            return self.finish('unparseable_options',
                               ', '.join(sorted(repr(t) for k, t in option_texts.items() if parsed[k] is None)))
        out['match'] = {key: v == value for key, v in parsed.items()}
        kind = 'exact'
        if self.rounding_slack is not None and out['inexact_ops']['fix_rounding'] > 0:
            rs = self.rounding_slack
            slack = Fraction(repr(rs) if isinstance(rs, float) else rs)
            out['match'] = {key: abs(v - value) <= slack for key, v in parsed.items()}
            kind = 'within_slack'
        hits = sum(out['match'].values())
        if hits == 0:
            return self.finish('no_match')
        if hits > 1:
            return self.finish('multi_match')
        out['match_kind'] = kind
        return self.finish()
