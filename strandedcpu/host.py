"""Feed batches into a running program through runtime.mailbox().

mb_wait() raises mb_req. Between instructions, the host fills mb[], sets
mb_len and clears mb_req; an exhausted input iterator supplies mb_len=0.
The interpreter path writes memory words, while the transformer path writes
the corresponding state rows.
"""

from typing import Dict, Iterable, List, Optional

from strandedcpu.compiler import compile_c, fix_value
from strandedcpu.interpreter import ISAInterpreter


def _word(v: int, N: int) -> int:
    v &= (1 << N) - 1
    return v - (1 << N) if v >= 1 << (N - 1) else v


def run_with_mailbox(source: str, batches: Iterable[List[int]], N: int = 16, s: int = 128,
                     m: int = 256, backend: str = "interpreter", max_steps: int = 2_000_000,
                     inputs: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    cfg, memory, commands, meta = compile_c(source, s=s, m=m, n=4096, N=N)
    var = meta['variables']
    req, length, box = var['mb_req'], var['mb_len'], var['mb']
    cap = meta['sizes']['mb']
    for name, value in (inputs or {}).items():
        memory[var[name]] = _word(value, N)
    feed = iter(batches)

    def next_batch():
        batch = next(feed, None)
        if batch is None:
            return {length: 0, req: 0}
        if len(batch) > cap:
            raise ValueError(f"batch of {len(batch)} exceeds mb[{cap}]")
        writes = {box + i: _word(v, N) for i, v in enumerate(batch)}
        writes.update({length: len(batch), req: 0})
        return writes

    if backend == "interpreter":
        vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, memory, commands, cfg.log_frac_bits)
        steps = 0
        while not vm.step():
            steps += 1
            if steps >= max_steps:
                raise RuntimeError("no HALT")
            if vm.memory[req] == 1:
                for addr, v in next_batch().items():
                    vm.memory[addr] = v
        final = vm.memory
    else:
        import torch
        from strandedcpu.transformer import Config, Computer, init_state, read_memory, get_pc
        from strandedcpu.subleq import signed_to_bipolar
        n = 128
        while n < cfg.s + cfg.m + len(commands):
            n *= 2
        tcfg = Config(s=cfg.s, m=cfg.m, n=n, N=N)
        comp = Computer(tcfg)
        X = init_state(tcfg, memory, commands)
        rows = slice(tcfg.idx_memory, tcfg.idx_memory + N)
        steps = 0
        with torch.no_grad():
            while get_pc(X, tcfg) != 0:
                X = comp.step(X)
                steps += 1
                if steps >= max_steps:
                    raise RuntimeError("no HALT")
                if X[rows, tcfg.s + req][-1] > 0 and read_memory(X, tcfg)[req] == 1:
                    for addr, v in next_batch().items():
                        X[rows, tcfg.s + addr] = signed_to_bipolar(v, N)
        final = read_memory(X, tcfg)

    out: Dict[str, object] = {}
    for name, addr in var.items():
        if name in meta['fix_variables']:
            out[name] = fix_value(final[addr], final[addr + 1], N)
        elif meta['sizes'].get(name, 1) > 1:
            out[name] = final[addr:addr + meta['sizes'][name]]
        else:
            out[name] = final[addr]
    out['__steps'] = steps
    return out
