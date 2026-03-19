"""
Parameter Golf - Depth Recurrent Transformer with LoRA Adapters
Optimized for DGX Spark (GB10 Blackwell) development, targeting 8xH100 submission.

Key changes from baseline:
- Depth recurrence: NUM_PHYSICAL_LAYERS shared blocks looped NUM_LOOPS times
- Optional per-loop LoRA adapters on Q/K/V/O for per-iteration specialization
- Per-effective-layer control params (attn_scales, mlp_scales, resid_mixes)
- SwiGLU MLP option (better parameter efficiency than ReLU-squared)
- Auto-detects GPU arch for optimal SDP backend (Blackwell vs Hopper)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    val_stride = int(os.environ.get("VAL_STRIDE", 64))  # sliding window eval stride (free BPB gain)

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))  # longer warmdown (winning technique)
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))  # 0 = no limit (Spark default)
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape — depth recurrence
    # Default: 3 physical layers × 5 loops = 15 effective depth at dim 640
    # dim=640 fits GB10 torch.compile (dim=768 exceeds Triton SM shared memory)
    # MLP 3x expansion is key winning technique (enabled by int6 quant saving ~4MB)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_physical_layers = int(os.environ.get("NUM_PHYSICAL_LAYERS", 3))
    num_loops = int(os.environ.get("NUM_LOOPS", 5))
    model_dim = int(os.environ.get("MODEL_DIM", 640))
    num_heads = int(os.environ.get("NUM_HEADS", 10))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 5))
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", 1920))  # 3x expansion (winning technique)
    use_swiglu = bool(int(os.environ.get("USE_SWIGLU", "1")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # LoRA for per-loop specialization (0 = no LoRA, pure weight sharing)
    lora_rank = int(os.environ.get("LORA_RANK", 0))

    # Optimizer hyperparameters (tuned to match winning submissions)
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))  # lower LR for better quant gap
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    lora_lr = float(os.environ.get("LORA_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))  # higher momentum (winning technique)
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))  # stabilizes training (winning technique)
    grad_accum_override = int(os.environ.get("GRAD_ACCUM", 0))  # 0 = auto (8 // world_size)
    disable_compile = bool(int(os.environ.get("DISABLE_COMPILE", "0")))  # 1 = skip torch.compile

    # STE Quantization-Aware Training: simulates quantization during forward pass
    # Reduces post-quant BPB gap from ~0.05 to ~0.001. Every top entry uses this.
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.5))  # enable QAT after this fraction of training

    @property
    def effective_layers(self):
        return self.num_physical_layers * self.num_loops

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Sliding window evaluation — overlapping windows at val_stride for better BPB."""
    seq_len = args.train_seq_len
    stride = min(args.val_stride, seq_len) if args.val_stride > 0 else seq_len
    total_tokens = val_tokens.numel() - 1

    # Generate window start positions, distribute across ranks
    starts = list(range(0, total_tokens - seq_len + 1, stride))
    rank_starts = starts[rank::world_size]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(rank_starts), max(1, args.val_batch_size // (seq_len * world_size))):
            batch_starts = rank_starts[i : i + max(1, args.val_batch_size // (seq_len * world_size))]
            if not batch_starts:
                break
            xs, ys = [], []
            for s in batch_starts:
                chunk = val_tokens[s : s + seq_len + 1].to(dtype=torch.int64)
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
            x = torch.stack(xs).to(device, non_blocking=True)
            y = torch.stack(ys).to(device, non_blocking=True)
            # Only score the last `stride` tokens in each window (avoid double-counting)
            score_start = seq_len - stride
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            # For loss, we use the full window loss (model returns mean over all positions)
            # but count only the scored tokens for BPB
            batch_token_count = float(y[:, score_start:].numel())
            val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            # BPB byte counting on scored portion
            prev_ids = x[:, score_start:].reshape(-1)
            tgt_ids = y[:, score_start:].reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    # For BPB, use the scored token count and byte count
    scored_tokens = val_byte_count  # bytes in scored region
    model.train()
    return float(val_loss.item()), float(bits_per_token * (val_token_count.item() / val_byte_count.item()))

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,loop_gate",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
QUANT_KEEP_FLOAT_MAX_NUMEL = 65_536
QUANT_KEEP_FLOAT_STORE_DTYPE = torch.float16
QUANT_PER_ROW_SCALE_DTYPE = torch.float16
QUANT_CLIP_PERCENTILE = 99.99984
QUANT_CLIP_Q = QUANT_CLIP_PERCENTILE / 100.0
# QUANT_BITS: 8 for int8 (baseline), 6 for int6 (saves ~25% on large matrices)
QUANT_BITS = int(os.environ.get("QUANT_BITS", "6"))

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=QUANT_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

# --- int6 packing: 4 values (6 bits each) into 3 bytes ---

def pack_int6(q: Tensor) -> Tensor:
    """Pack int6 values (range -31..31 stored as uint8 0..63) into 3/4 the bytes."""
    flat = (q.flatten().to(torch.int16) + 31).to(torch.uint8)  # shift to unsigned 0..62
    # Pad to multiple of 4
    pad = (4 - flat.numel() % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    flat = flat.reshape(-1, 4)
    a, b, c, d = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    # Pack 4 x 6-bit values into 3 bytes
    b0 = (a << 2) | (b >> 4)
    b1 = ((b & 0x0F) << 4) | (c >> 2)
    b2 = ((c & 0x03) << 6) | d
    return torch.stack([b0, b1, b2], dim=1).flatten().contiguous()

def unpack_int6(packed: Tensor, numel: int) -> Tensor:
    """Unpack 3-byte groups into 4 int6 values, return numel values in -31..31."""
    packed = packed.reshape(-1, 3)
    b0, b1, b2 = packed[:, 0].to(torch.int16), packed[:, 1].to(torch.int16), packed[:, 2].to(torch.int16)
    a = (b0 >> 2) & 0x3F
    b = ((b0 & 0x03) << 4) | (b1 >> 4)
    c = ((b1 & 0x0F) << 2) | (b2 >> 6)
    d = b2 & 0x3F
    flat = torch.stack([a, b, c, d], dim=1).flatten()[:numel]
    return (flat - 31).to(torch.int8)  # shift back to signed

def quantize_float_tensor(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor, int]:
    """Quantize to int8 or int6. Returns (quantized, scale, original_numel)."""
    t32 = t.float()
    max_val = 127 if bits == 8 else 31
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), QUANT_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(max_val)).clamp_min(1.0 / float(max_val))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val).to(torch.int8).contiguous()
        if bits == 6:
            numel = q.numel()
            q = pack_int6(q)
            return q, scale.to(dtype=QUANT_PER_ROW_SCALE_DTYPE).contiguous(), numel
        return q, scale.to(dtype=QUANT_PER_ROW_SCALE_DTYPE).contiguous(), q.numel()
    clip_abs = float(torch.quantile(t32.abs().flatten(), QUANT_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(max_val) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_val, max_val).to(torch.int8).contiguous()
    if bits == 6:
        numel = q.numel()
        q = pack_int6(q)
        return q, scale, numel
    return q, scale, q.numel()

# --- Embeddings: keep as fp16 passthrough (winning technique) ---
EMBED_NAME_PATTERNS = ("tok_emb",)

def quantize_state_dict(state_dict: dict[str, Tensor], bits: int = 8):
    """Quantize with int8 or int6, fp16 embeddings."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "quant_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["quant_payload_bytes"] += tensor_nbytes(t)
            continue
        # Embeddings: store as fp16 passthrough (better quality, winning technique)
        is_embed = any(pattern in name for pattern in EMBED_NAME_PATTERNS)
        if t.numel() <= QUANT_KEEP_FLOAT_MAX_NUMEL or is_embed:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["quant_payload_bytes"] += tensor_nbytes(kept)
            continue
        # Use int6 for large matrices, int8 for vectors
        tensor_bits = bits if t.ndim == 2 else 8
        stats["num_float_tensors"] += 1
        q, s, numel = quantize_float_tensor(t, bits=tensor_bits)
        meta: dict[str, object] = {}
        if s.ndim > 0:
            meta["scheme"] = "per_row"
            meta["axis"] = 0
        if tensor_bits == 6:
            meta["bits"] = 6
            meta["numel"] = numel
            meta["shape"] = list(t.shape)
        if meta:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["quant_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": f"int{bits}_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        # Unpack int6 if needed
        if meta.get("bits") == 6:
            numel = meta["numel"]
            shape = meta["shape"]
            q_unpacked = unpack_int6(q, numel).float()
            if meta.get("scheme") == "per_row":
                s = s.to(dtype=torch.float32)
                out[name] = (q_unpacked.reshape(shape) * s.view(shape[0], *([1] * (len(shape) - 1)))).to(dtype=dtype).contiguous()
            else:
                out[name] = (q_unpacked.reshape(shape) * float(s.item())).to(dtype=dtype).contiguous()
        elif meta.get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


# --- STE Quantization-Aware Training ---
# Simulates int6/int8 quantization during forward pass.
# Gradients pass through unchanged (straight-through estimator).
# This trains the model to be robust to quantization noise.

_qat_active = False  # global flag, toggled during training

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_val=31):
        if x.ndim == 2:
            clip_abs = torch.amax(x.abs(), dim=1)
            scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
            q = torch.clamp(torch.round(x / scale[:, None]), -max_val, max_val)
            return q * scale[:, None]
        clip_abs = torch.amax(x.abs())
        scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
        q = torch.clamp(torch.round(x / scale), -max_val, max_val)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def ste_quantize(x: Tensor, bits: int = 6) -> Tensor:
    max_val = 31 if bits == 6 else 127
    return STEQuantize.apply(x, max_val)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.training and _qat_active and self.weight.ndim == 2 and self.weight.numel() > QUANT_KEEP_FLOAT_MAX_NUMEL:
            w = ste_quantize(w, bits=QUANT_BITS)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# -----------------------------
# LoRA ADAPTER
# -----------------------------

class LoRAAdapter(nn.Module):
    """Low-rank adaptation for attention Q/K/V/O projections."""
    def __init__(self, dim: int, kv_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        # Down projections initialized with small random values
        # Up projections initialized to zero (LoRA starts as identity)
        self.q_down = CastedLinear(dim, rank, bias=False)
        self.q_up = CastedLinear(rank, dim, bias=False)
        self.k_down = CastedLinear(dim, rank, bias=False)
        self.k_up = CastedLinear(rank, kv_dim, bias=False)
        self.v_down = CastedLinear(dim, rank, bias=False)
        self.v_up = CastedLinear(rank, kv_dim, bias=False)
        self.o_down = CastedLinear(dim, rank, bias=False)
        self.o_up = CastedLinear(rank, dim, bias=False)
        self._init_lora()

    def _init_lora(self):
        for name, param in self.named_parameters():
            if "_down" in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                param.data *= 1.0 / self.rank
            elif "_up" in name:
                nn.init.zeros_(param)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, lora: LoRAAdapter | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        if lora is not None:
            q = q + lora.q_up(lora.q_down(x))
            k = k + lora.k_up(lora.k_down(x))
            v = v + lora.v_up(lora.v_down(x))
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        out = self.proj(y)
        if lora is not None:
            out = out + lora.o_up(lora.o_down(y))
        return out


# -----------------------------
# MLP VARIANTS
# -----------------------------

class MLP(nn.Module):
    """ReLU-squared MLP from modded-nanogpt."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP — better parameter efficiency than ReLU-squared."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.down = CastedLinear(hidden, dim, bias=False)
        self.down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# -----------------------------
# SHARED BLOCK (no per-layer scalars — those live in GPT)
# -----------------------------

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden: int,
        rope_base: float,
        qk_gain_init: float,
        use_swiglu: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = SwiGLUMLP(dim, mlp_hidden) if use_swiglu else MLP(dim, mlp_hidden)

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        attn_scale: Tensor,
        mlp_scale: Tensor,
        resid_mix: Tensor,
        lora: LoRAAdapter | None = None,
    ) -> Tensor:
        mix = resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), lora)
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# -----------------------------
# RECURRENT GPT
# -----------------------------

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_physical_layers: int,
        num_loops: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden: int,
        use_swiglu: bool,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        lora_rank: int,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_physical_layers = num_physical_layers
        self.num_loops = num_loops
        effective_layers = num_physical_layers * num_loops

        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # Shared physical blocks
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_hidden, rope_base, qk_gain_init, use_swiglu)
            for _ in range(num_physical_layers)
        ])

        # Per-effective-layer control parameters (NOT shared)
        self.attn_scales = nn.Parameter(torch.ones(effective_layers, model_dim, dtype=torch.float32))
        self.mlp_scales = nn.Parameter(torch.ones(effective_layers, model_dim, dtype=torch.float32))
        self.resid_mixes = nn.Parameter(
            torch.stack([
                torch.stack([torch.ones(model_dim), torch.zeros(model_dim)])
                for _ in range(effective_layers)
            ]).float()
        )

        # U-net skip connections over effective layers
        self.num_encoder_layers = effective_layers // 2
        self.num_decoder_layers = effective_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Optional LoRA adapters per effective layer
        self.lora_rank = lora_rank
        kv_dim = num_kv_heads * (model_dim // num_heads)
        if lora_rank > 0:
            self.lora_adapters = nn.ModuleList([
                LoRAAdapter(model_dim, kv_dim, lora_rank)
                for _ in range(effective_layers)
            ])
        else:
            self.lora_adapters = None

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # Encoder half — store skip connections
        for i in range(self.num_encoder_layers):
            block_idx = i % self.num_physical_layers
            lora = self.lora_adapters[i] if self.lora_adapters is not None else None
            x = self.blocks[block_idx](
                x, x0,
                self.attn_scales[i],
                self.mlp_scales[i],
                self.resid_mixes[i],
                lora,
            )
            skips.append(x)

        # Decoder half — consume skip connections in reverse
        for i in range(self.num_decoder_layers):
            eff_idx = self.num_encoder_layers + i
            block_idx = eff_idx % self.num_physical_layers
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            lora = self.lora_adapters[eff_idx] if self.lora_adapters is not None else None
            x = self.blocks[block_idx](
                x, x0,
                self.attn_scales[eff_idx],
                self.mlp_scales[eff_idx],
                self.resid_mixes[eff_idx],
                lora,
            )

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# GPU DETECTION
# -----------------------------

def configure_sdp_backends():
    """Auto-detect GPU architecture and set optimal SDP backends."""
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 10:  # Blackwell (sm_120+)
            enable_cudnn_sdp(True)
            enable_flash_sdp(False)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(False)
            return "cudnn (Blackwell)"
    # Default: Flash Attention (Hopper/Ampere)
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    return "flash (Hopper/Ampere)"


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # Inductor config for GB10 (small SM shared memory)
    if not args.disable_compile:
        import torch._inductor.config as inductor_config
        inductor_config.max_autotune = False
        inductor_config.max_autotune_gemm = False
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed + CUDA setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")

    if args.grad_accum_override > 0:
        grad_accum_steps = args.grad_accum_override
    else:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
        grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sdp_mode = configure_sdp_backends()

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # Tokenizer + validation setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # Model setup
    effective_layers = args.effective_layers
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_physical_layers=args.num_physical_layers,
        num_loops=args.num_loops,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_hidden=args.mlp_hidden,
        use_swiglu=args.use_swiglu,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        lora_rank=args.lora_rank,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False) if not args.disable_compile else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - Shared block matrices → Muon
    # - Block scalars (q_gain) → Adam (scalar_lr)
    # - Per-effective-layer control params → Adam (scalar_lr)
    # - LoRA matrices → Adam (lora_lr)
    # - Token embedding → Adam (embed_lr / tied_embed_lr)
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    # Add per-effective-layer control params
    scalar_params.extend([base_model.attn_scales, base_model.mlp_scales, base_model.resid_mixes])
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr,
        momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    # LoRA optimizer (Adam, separate LR)
    if base_model.lora_adapters is not None:
        lora_params = list(base_model.lora_adapters.parameters())
        optimizer_lora = torch.optim.Adam(
            [{"params": lora_params, "lr": args.lora_lr, "base_lr": args.lora_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.append(optimizer_lora)

    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_shared = sum(p.numel() for p in base_model.blocks.parameters())
    n_lora = sum(p.numel() for p in base_model.lora_adapters.parameters()) if base_model.lora_adapters else 0
    log0(f"model_params:{n_params} (shared_blocks:{n_shared} lora:{n_lora})")
    log0(f"architecture: {args.num_physical_layers} physical x {args.num_loops} loops = {effective_layers} effective layers")
    log0(f"model_dim:{args.model_dim} num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"mlp_hidden:{args.mlp_hidden} use_swiglu:{args.use_swiglu} lora_rank:{args.lora_rank}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"sdp_backends:{sdp_mode}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} lora_lr:{args.lora_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # Data loader + warmup
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # STE QAT: activate after qat_start_frac of training
        global _qat_active
        if args.qat_enabled:
            qat_threshold = int(args.iterations * args.qat_start_frac)
            if not _qat_active and step >= qat_threshold:
                _qat_active = True
                log0(f"QAT activated at step {step} (int{QUANT_BITS} simulation)")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # Serialization + roundtrip validation
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict(base_model.state_dict(), bits=QUANT_BITS)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    # Use zstd-22 if available (better compression), fall back to zlib
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        quant_blob = cctx.compress(quant_raw)
        compress_label = "zstd22"
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
        compress_label = "zlib9"
    quant_raw_bytes = len(quant_raw)
    quant_label = f"int{QUANT_BITS}"
    if master_process:
        ext = "ptzst" if HAS_ZSTD else "ptz"
        with open(f"final_model.{quant_label}.{ext}", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["quant_payload_bytes"], 1)
        log0(
            f"Serialized model {quant_label}+{compress_label}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['quant_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        total_bytes = quant_file_bytes + code_bytes
        log0(f"Total submission size {quant_label}+{compress_label}: {total_bytes} bytes")
        if total_bytes > 16_000_000:
            log0(f"WARNING: Total size {total_bytes} exceeds 16MB limit by {total_bytes - 16_000_000} bytes!")
        else:
            log0(f"Size budget remaining: {16_000_000 - total_bytes} bytes")

    if distributed:
        dist.barrier()
    ext = "ptzst" if HAS_ZSTD else "ptz"
    with open(f"final_model.{quant_label}.{ext}", "rb") as f:
        quant_blob_disk = f.read()
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        quant_state = torch.load(io.BytesIO(dctx.decompress(quant_blob_disk)), map_location="cpu")
    else:
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_{quant_label}_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_{quant_label}_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
