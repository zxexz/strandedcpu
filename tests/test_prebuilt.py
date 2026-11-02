"""Check the CUDA fatbin size, architectures, and kernel-source hash.

Rebuild with `python -m strandedcpu.cuda build-fatbin` after changing the
kernel source. Building requires nvcc 12.8+; this test runs without a GPU.
"""

import json
import os

import strandedcpu.cuda


def test_prebuilt_kernel_matches_the_source():
    assert os.path.exists(strandedcpu.cuda.FATBIN), "no prebuilt kernel; run python -m strandedcpu.cuda build-fatbin"
    with open(strandedcpu.cuda.FATBIN + '.json') as f:
        meta = json.load(f)
    assert meta['source_sha256'] == strandedcpu.cuda.source_sha256(), \
        "kernel source changed since the fatbin was built; run python -m strandedcpu.cuda build-fatbin"
    assert {'sm_90', 'sm_120'} <= set(meta['archs'])          # H100 and RTX PRO 6000 at least
    assert os.path.getsize(strandedcpu.cuda.FATBIN) == meta['bytes']
