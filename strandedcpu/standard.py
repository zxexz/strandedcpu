"""Convert the analytical computer to broadcast-only FFN biases.

An extra state row is 1 at column zero and 0 elsewhere. Its weights absorb
column-zero bias differences; added FFN rows handle scratchpad bias offsets.
Attention uses hard top-two selection, including query-wise broadcast and
gather heads from the vector unit.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from strandedcpu.subleq import (
    SUBLEQConfig, to_bipolar, from_bipolar, signed_to_bipolar, signed_from_bipolar,
    _attn_head_query_wise, _top2_weights, _scores,
)

from strandedcpu.transformer import (
    Computer, Config,
    init_state, read_memory, get_pc,
    opcode_pattern, create_bipolar_value,
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC,
    OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_AND, OP_OR, OP_XOR, OP_SUB,
    OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC, OP_STORE,
)


# Config

@dataclass
class StandardConfig(Config):
    """Append a column-zero indicator row to Config."""

    def __post_init__(self):
        super().__post_init__()
        self.idx_col0 = self.d_model          # right after indicator
        self.idx_indicator = self.d_model - 1
        self.d_model += 1


# Standard Transformer Layer

class StandardTransformerLayer:
    """Transformer layer with broadcast bias vectors and hard top-two attention.

    Each head updates the running attention state; the FFN then applies a
    ReLU projection and residual update. query_wise selects per-target attention
    for heads that broadcast or gather.
    """

    def __init__(self, num_heads: int, lam: float):
        self.num_heads = num_heads
        self.lam = lam
        self.Qs: List[torch.Tensor] = []
        self.Ks: List[torch.Tensor] = []
        self.Vs: List[torch.Tensor] = []
        self.W1: torch.Tensor = None
        self.b1: torch.Tensor = None   # 1-D
        self.W2: torch.Tensor = None
        self.b2: torch.Tensor = None   # 1-D
        self.query_wise: List[bool] = []

    # forward (argmax attention)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        attn = X
        qw = self.query_wise or [False] * len(self.Qs)
        for Q, K, V, query_wise in zip(self.Qs, self.Ks, self.Vs, qw):
            attn = attn + _argmax_attention(Q, K, V, attn, query_wise)
        ff1 = F.relu(self.W1 @ attn + self.b1.unsqueeze(1))
        return attn + self.W2 @ ff1 + self.b2.unsqueeze(1)

    forward_argmax = forward  # alias for compatibility


def _argmax_attention(Q, K, V, X, query_wise=False):
    """Apply hard top-two attention with a score-gap threshold of 1.

    query_wise selects source columns per target. The default selects targets
    per source and keeps zero-key rows active, unlike subleq._attn_head.
    A head whose query and key projections are both zero contributes nothing.
    """
    n = X.shape[1]
    QX = Q @ X                         # q_rows × n
    KX = K @ X                         # q_rows × n
    VX = V @ X                         # d × n

    if QX.abs().max() < 1e-6 and KX.abs().max() < 1e-6:
        return torch.zeros_like(VX)

    if query_wise:
        return _attn_head_query_wise(QX, KX, VX)

    scores = _scores(KX, QX)           # n × n, scores[i,j] (float64)

    # A[i,j] routes source i into output column j.

    # Top-2 per row (dim=1); unlike base transformer, zero-key rows are not skipped
    return VX @ _top2_weights(scores, torch.ones(n, dtype=torch.bool))


# Conversion from base transformer layers

def _standardize_layer(old_layer, d_old: int, d_new: int,
                       col0_idx: int, indicator_idx: int,
                       n: int, s: int) -> StandardTransformerLayer:
    """Absorb per-column bias differences into weights and broadcast vectors.

    Pad Q/K/V for the new indicator row. Fold the column-zero b1 delta into W1;
    extra FFN rows reproduce b2 offsets using scratchpad and column-zero indicators.
    """
    new = StandardTransformerLayer(old_layer.num_heads, old_layer.lam)

    # 1. Expand Q, K, V
    def pad_col(M):
        return F.pad(M, (0, 1))            # append zero column

    def pad_square(M):
        return F.pad(F.pad(M, (0, 1)), (0, 0, 0, 1))  # append col then row

    heads = old_layer.num_heads
    if heads == 1:
        new.Qs = [pad_col(old_layer.Q)]
        new.Ks = [pad_col(old_layer.K)]
        new.Vs = [pad_square(old_layer.V)]
    elif heads == 2:
        new.Qs = [pad_col(old_layer.Q1), pad_col(old_layer.Q2)]
        new.Ks = [pad_col(old_layer.K1), pad_col(old_layer.K2)]
        new.Vs = [pad_square(old_layer.V1), pad_square(old_layer.V2)]
    else:
        new.Qs = [pad_col(old_layer.Q1), pad_col(old_layer.Q2),
                  pad_col(old_layer.Q3)]
        new.Ks = [pad_col(old_layer.K1), pad_col(old_layer.K2),
                  pad_col(old_layer.K3)]
        new.Vs = [pad_square(old_layer.V1), pad_square(old_layer.V2),
                  pad_square(old_layer.V3)]
    qw = getattr(old_layer, 'query_wise', set())
    new.query_wise = [h + 1 in qw for h in range(heads)]

    # 2. Convert b1 → broadcast + W1 col0 weight
    w1_rows = old_layer.W1.shape[0]
    ref_col = s                              # first memory column
    b1_broadcast = old_layer.b1[:, ref_col].clone()
    b1_col0_delta = old_layer.b1[:, 0] - b1_broadcast

    W1_padded = pad_col(old_layer.W1)        # w × d_new
    W1_padded[:, col0_idx] += b1_col0_delta  # fold col-0 bias into weight

    # 3. Convert b2 → broadcast + extra FFN rows
    b2_mem_ref = old_layer.b2[:, s].clone()            # memory-column baseline
    b2_scr_ref = old_layer.b2[:, min(1, s - 1)].clone() if s > 1 \
                 else old_layer.b2[:, 0].clone()
    b2_scr_delta = b2_scr_ref - b2_mem_ref             # scratchpad offset
    b2_col0_extra = old_layer.b2[:, 0] - b2_scr_ref   # col-0 extra offset

    has_scr = b2_scr_delta.abs().max().item() > 1e-10
    has_col0 = b2_col0_extra.abs().max().item() > 1e-10
    extra = int(has_scr) + int(has_col0)
    total_rows = w1_rows + extra

    W1_final = torch.zeros(total_rows, d_new)
    W1_final[:w1_rows] = W1_padded

    W2_padded = F.pad(old_layer.W2, (0, 0, 0, 1))  # (d_new) × w1_rows
    W2_final = torch.zeros(d_new, total_rows)
    W2_final[:, :w1_rows] = W2_padded

    b1_final = torch.zeros(total_rows)
    b1_final[:w1_rows] = b1_broadcast

    b2_final = F.pad(b2_mem_ref, (0, 1))   # d_new (broadcast)

    # Extra FFN rows for b2 per-column parts
    idx = w1_rows
    if has_scr:
        # Row that reads scratchpad indicator → 1 at scratchpad, 0 at memory
        W1_final[idx, indicator_idx] = 1.0
        b1_final[idx] = 0.0
        W2_final[:d_old, idx] = b2_scr_delta
        idx += 1
    if has_col0:
        # Row that reads col0_indicator → 1 at col 0, 0 elsewhere
        W1_final[idx, col0_idx] = 1.0
        b1_final[idx] = 0.0
        W2_final[:d_old, idx] = b2_col0_extra
        idx += 1

    new.W1 = W1_final
    new.b1 = b1_final
    new.W2 = W2_final
    new.b2 = b2_final

    return new


# Main Computer Class

class StandardComputer:
    """Convert Computer's layers to broadcast-bias form with hard attention."""

    def __init__(self, cfg: StandardConfig, use_argmax: bool = False):
        self.cfg = cfg
        self.use_argmax = use_argmax

        base_cfg = Config(s=cfg.s, m=cfg.m, n=cfg.n, N=cfg.N, lam=cfg.lam,
                            log_frac_bits=cfg.log_frac_bits, vector_unit=cfg.vector_unit)
        base = Computer(base_cfg)

        d_old = base_cfg.d_model
        d_new = cfg.d_model
        self.layers: List[StandardTransformerLayer] = []
        for layer in base.layers:
            std = _standardize_layer(
                layer, d_old, d_new,
                col0_idx=cfg.idx_col0,
                indicator_idx=cfg.idx_indicator,
                n=cfg.n, s=cfg.s,
            )
            self.layers.append(std)

    def step(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_argmax:
                X = layer.forward_argmax(X)
            else:
                X = layer.forward(X)
        return X

    def run(self, X: torch.Tensor, max_steps: int = 1000) -> Tuple[torch.Tensor, int]:
        for step in range(max_steps):
            pc = get_pc_standard(X, self.cfg)
            if pc == 0 or pc >= self.cfg.n:
                return X, step
            X = self.step(X)
        return X, max_steps


# State Init / Read helpers

def init_state_standard(cfg: StandardConfig, memory, commands):
    """Initialise state tensor with col0_indicator appended."""
    base_cfg = Config(s=cfg.s, m=cfg.m, n=cfg.n, N=cfg.N, lam=cfg.lam,
                        log_frac_bits=cfg.log_frac_bits, vector_unit=cfg.vector_unit)
    X_base = init_state(base_cfg, memory, commands)

    X = torch.zeros(cfg.d_model, cfg.n)
    X[:base_cfg.d_model] = X_base          # all base transformer rows including indicator
    X[cfg.idx_col0, 0] = 1.0           # col0_indicator
    return X


def read_memory_standard(X, cfg: StandardConfig):
    """Read memory values (same indices as base transformer)."""
    memory = []
    for i in range(cfg.m):
        col = cfg.s + i
        bipolar = X[cfg.idx_memory:cfg.idx_memory + cfg.N, col]
        memory.append(signed_from_bipolar(bipolar))
    return memory


def get_pc_standard(X, cfg: StandardConfig):
    """Get PC value (same indices as base transformer)."""
    return from_bipolar(X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0])


# Quick self-test

if __name__ == "__main__":
    cfg = StandardConfig(s=32, m=8, n=64, N=8)
    print(f"StandardConfig: d_model={cfg.d_model} "
          f"(base transformer was {cfg.d_model - 1}), idx_col0={cfg.idx_col0}")

    ext = StandardComputer(cfg)
    print(f"Layers: {len(ext.layers)}")
    for i, layer in enumerate(ext.layers):
        print(f"  L{i+1}: heads={layer.num_heads}, "
              f"W1={list(layer.W1.shape)}, "
              f"b1={list(layer.b1.shape)}, "
              f"W2={list(layer.W2.shape)}, "
              f"b2={list(layer.b2.shape)}")

    memory = [5, 0, 0, 0, 0, 0, 0, 0]
    commands = [
        (OP_INC, cfg.s + 0, 0),
        (OP_HALT, 0, 0),
    ]
    X = init_state_standard(cfg, memory, commands)
    with torch.no_grad():
        X = ext.step(X)
    result = read_memory_standard(X, cfg)
    print(f"\nINC test: mem[0]=5 -> {result[0]} (expected 6)")
    assert result[0] == 6, f"FAIL: got {result[0]}"
    print("PASS")
