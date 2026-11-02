"""Generate Python solutions with a chat model and check their numeric answers.

Generation expects model.generate() and tokenizer.apply_chat_template().
check_programs uses rational arithmetic through the scoring API, with CUDA
when available under backend="auto". Its results can select successful
programs for rejection sampling.
"""

import re
import time
from typing import Dict, List, Optional, Sequence

PROMPT = """Solve the problem by writing a Python function `solution()` that takes no arguments and returns the final numeric answer. Programs use finite-width integers and rational arithmetic. Python parses float literals before conversion to fractions, so long literals can lose digits. Do not add epsilons for roundoff in fraction arithmetic.

Use only: integers and decimal numbers; + - * / // %; ** with an integer exponent or an exact constant fractional exponent; comparisons; and/or/not; if/elif/else; for loops over range(...); while loops; break/continue; lists of numbers (indexing, len, sum, min, max); helper functions (no recursion); round, abs, min, max, int, float, math.floor, math.ceil, math.sqrt, math.isqrt. math.sqrt and fractional powers require an exact rational result; math.isqrt returns the floor square root of a nonnegative integer.
Do not use strings, dictionaries, sets, imports other than math, input() or print().

Keep comments brief. Reply with only the code in one ```python block.

Problem: {question}"""

_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_program(text: str) -> Optional[str]:
    """The last ```python block of a reply (or None)."""
    blocks = _CODE.findall(text)
    return blocks[-1].strip() + "\n" if blocks else None


def gsm8k_gold(answer: str) -> str:
    """GSM8K's final answer, after '####'."""
    return answer.split('####')[-1].strip()


def write_programs(model, tok, questions: Sequence[str], batch_size: int = 16, max_new_tokens: int = 1024,
                   temperature: float = 0.0, samples: int = 1, prompt: str = PROMPT) -> List[List[str]]:
    """For each question, `samples` replies (greedy when temperature is 0)."""
    import torch
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    chats = [tok.apply_chat_template([{'role': 'user', 'content': prompt.format(question=q)}],
                                     tokenize=False, add_generation_prompt=True) for q in questions]
    out: List[List[str]] = [[] for _ in questions]
    jobs = [(i, c) for i, c in enumerate(chats) for _ in range(samples)]
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        enc = tok([c for _, c in chunk], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            gen = model.generate(**enc, **kw)
        for (i, _), seq in zip(chunk, gen[:, enc['input_ids'].shape[1]:]):
            out[i].append(tok.decode(seq, skip_special_tokens=True))
    return out


def check_programs(programs: Sequence[Optional[str]], golds: Sequence[str], backend: str = 'auto',
                   compile_workers: int = 0, max_steps: int = 10_000_000) -> List[Dict[str, object]]:
    """Run each program in rational mode at N=32 against its gold answer.
    A missing program counts as a compile error."""
    if len(programs) != len(golds):
        raise ValueError("programs and golds must have equal lengths")
    from strandedcpu.python import score_python_batch
    items = [{'src': p if p is not None else "# no program\n", 'inputs': {}, 'option_texts': {'gold': g}}
             for p, g in zip(programs, golds)]
    return score_python_batch(items, N=32, numbers='rational', backend=backend, compile_workers=compile_workers,
                              max_steps=max_steps)


def summarize(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Return answer accuracy and counts of each abstention reason."""
    n = len(results)
    correct = sum(r['status'] == 'ok' for r in results)
    reasons: Dict[str, int] = {}
    for r in results:
        if r['status'] != 'ok':
            reasons[r['abstain_reason']] = reasons.get(r['abstain_reason'], 0) + 1
    return {'n': n, 'correct': correct, 'accuracy': correct / max(n, 1), 'failures': reasons}


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


# Multiple-choice formatting and exact numeric matching.

DIRECT_PROMPT = """Answer the multiple-choice question. Reason briefly, then end with a line "Answer: X" where X is the option letter.

{question}"""

_LETTER = re.compile(r"Answer:\s*\(?([A-Ea-e])\b")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def mcq_text(question: str, options: Sequence[str]) -> str:
    return question.strip() + "\nOptions:\n" + "\n".join(o.strip() for o in options)


def option_letter(option: str) -> str:
    return option.strip()[0].upper()


def option_number(option: str) -> Optional[str]:
    """The number an option states, as text parse_option_number reads, or
    None ("A)$67.40" -> "67.40"; "A)5(√3 + 1)" or "E)None of these" -> None).
    Units and currency around a single number are dropped."""
    from strandedcpu.api import parse_option_number
    text = re.sub(r"^\s*\(?[A-Ea-e]\)\s*", "", option).strip()
    if parse_option_number(text) is not None:
        return text
    if re.search(r"[√π^*×]|\d\s*[a-z]\b", text):     # roots, powers, products: not one plain number
        return None
    found = _NUMBER.findall(text.replace('$', ''))
    if len(found) == 1 and parse_option_number(found[0]) is not None:
        return found[0]
    return None


def parse_letter(reply: str) -> Optional[str]:
    found = _LETTER.findall(reply)
    return found[-1].upper() if found else None


def decisions(programs: Sequence[Optional[str]], options_list: Sequence[Sequence[str]],
              backend: str = 'auto') -> List[Optional[str]]:
    """Per question, the option letter StrandedCPU's result matches (exactly one
    option among those that state a plain number), or None."""
    if len(programs) != len(options_list):
        raise ValueError("programs and options_list must have equal lengths")
    from strandedcpu.python import score_python_batch
    items = []
    positions = []
    for index, (p, options) in enumerate(zip(programs, options_list)):
        texts = {option_letter(o): option_number(o) for o in options}
        texts = {k: v for k, v in texts.items() if v is not None}
        if not texts:
            continue
        positions.append(index)
        items.append({'src': p if p is not None else "# no program\n", 'inputs': {}, 'option_texts': texts})
    selected = [None] * len(programs)
    if not items:
        return selected
    results = score_python_batch(items, N=32, numbers='rational', backend=backend, max_steps=10_000_000)
    for index, result in zip(positions, results):
        if result['status'] == 'ok':
            selected[index] = next((key for key, matched in result['match'].items() if matched), None)
    return selected
