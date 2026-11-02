"""Build fixed-order option features from checked C or Python execution.

score_row accepts several questions and measures their combined wall time.
Each result includes per-option vectors ordered by FEATURES. Defaults are
N=32, rational arithmetic and boolean option parsing. Failed or ambiguous
runs set call_ok=0 and record an abstention reason.
"""

import math
import time
from typing import Dict, List, Optional

from strandedcpu.api import score_options_batch
from strandedcpu.python import score_python_batch

ABSTAIN_REASONS = ('compile_error', 'input_out_of_domain', 'overflow', 'runtime_error', 'step_limit',
                   'unparseable_options', 'no_match', 'multi_match')

FEATURES = (
    ['match',               # 1 for the matching option of an ok run
     'call_ok',             # 1 if the run gave exactly one matching option
     'other_matched']       # 1 for other options when the run has one valid match
    + [f'abstain_{r}' for r in ABSTAIN_REASONS]
    + ['inexact_div',       # an int / or // dropped a fraction
       'inexact_round',     # a fractional value was rounded (or cast to int dropping a fraction)
       'log2_steps']        # log2(1 + instructions executed)
)


def option_features(result: Dict[str, object], option_keys) -> Dict[str, List[float]]:
    """FEATURES for each option key of one score_options/score_python result."""
    match = result.get('match') or {}
    ok = result['status'] == 'ok'
    reason = result.get('abstain_reason')
    inexact = result.get('inexact_ops') or {}
    shared = ([float(r == reason) for r in ABSTAIN_REASONS]
              + [float(inexact.get('int_div_remainder', 0) > 0), float(inexact.get('fix_rounding', 0) > 0),
                 math.log2(1 + int(result.get('steps') or 0))])
    hit = [k for k in option_keys if match.get(k)]
    return {k: [float(bool(match.get(k)) and ok), float(ok), float(ok and k not in hit)] + shared
            for k in option_keys}


def score_item(program: str, inputs: Dict[str, object], options: Dict[str, str], *, language: str = 'python',
               **kw) -> Dict[str, object]:
    """Return one question's scoring result and per-option features."""
    return score_row([{'program': program, 'inputs': inputs, 'options': options, 'language': language}],
                     **kw)['items'][0]


def score_row(items: List[Dict[str, object]], *, N: int = 32, numbers: str = 'rational',
              boolean_options: bool = True, max_steps: int = 1_000_000, backend: str = 'interpreter',
              **kw) -> Dict[str, object]:
    """Score questions in input order and report the row's wall time.

    Each item contains program and options, with optional inputs and language
    ("python" or "c"). Python and C calls form separate batches; CUDA may split
    each batch further by execution configuration.
    """
    t0 = time.perf_counter()
    common = dict(N=N, numbers=numbers, boolean_options=boolean_options, max_steps=max_steps, backend=backend, **kw)
    py = [k for k, it in enumerate(items) if it.get('language', 'python') == 'python']
    cc = [k for k, it in enumerate(items) if it.get('language', 'python') == 'c']
    bad = [it.get('language') for it in items if it.get('language', 'python') not in ('python', 'c')]
    if bad:
        raise ValueError(f"language must be 'python' or 'c', not {bad[0]!r}")
    results: List[Optional[Dict[str, object]]] = [None] * len(items)
    for k, r in zip(py, score_python_batch([{'src': items[k]['program'], 'inputs': items[k].get('inputs') or {},
                                              'option_texts': items[k]['options']} for k in py], **common)):
        results[k] = r
    for k, r in zip(cc, score_options_batch([{'program': items[k]['program'], 'inputs': items[k].get('inputs') or {},
                                               'option_texts': items[k]['options']} for k in cc], **common)):
        results[k] = r
    out = [{'result': r, 'features': option_features(r, list(it['options']))} for r, it in zip(results, items)]
    return {'items': out, 'latency_s': time.perf_counter() - t0}
