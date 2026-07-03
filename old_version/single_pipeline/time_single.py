import hashlib
import json
import math
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import (
    ScoreWriter,
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    describe_loaded_data,
    finalize_metric_sums,
    get_destination_pool,
    get_negative_sampler,
    is_run_complete,
    load_datasets,
    load_metrics,
    save_config,
    save_metrics,
    set_random_seed,
)
DEFAULT_CALENDAR_ABS_PERIODS = [1.0, 7.0, 30.0, 365.0]
DEFAULT_SECONDS_ABS_PERIODS = [86400.0, 604800.0, 2592000.0, 31536000.0]


def profile_add(profile, key, value):
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + float(value)


def sync_device(device):
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def format_seconds(value):
    return f"{float(value):.1f}s"


def reset_cuda_peak(device):
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        sync_device(device)
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_mb(device):
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    sync_device(device)
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return alloc, reserved


def profile_sync_if_needed(args, device):
    if (
        getattr(args, "profile_sync", False)
        and getattr(device, "type", None) == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(device)


def profile_now(args=None, device=None):
    if args is not None and device is not None:
        profile_sync_if_needed(args, device)
    return time.time()


def forward_tic(profile, args, device):
    if profile is not None and profile.get("_measure_model_forward", False):
        sync_device(device)
        return time.time()
    return profile_now(args, device)


def forward_toc(profile, args, device, start, *keys):
    if profile is not None and profile.get("_measure_model_forward", False):
        sync_device(device)
        elapsed = time.time() - start
    else:
        elapsed = profile_now(args, device) - start
    for key in keys:
        profile_add(profile, key, elapsed)
    profile_add(profile, "eval_model_forward_time", elapsed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    value = str(value).strip()
    if value == "":
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    value = str(value).strip()
    if value == "":
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def get_history_topk(args):
    windows = parse_int_list(getattr(args, "multi_windows", ""))
    windows = [w for w in windows if w > 0]
    if windows:
        return max(windows)
    return int(args.topk)


def get_abs_time_value(args, t_norm, t_orig):
    return float(t_orig) if getattr(args, "abs_time_use_raw", False) else float(t_norm)


class FeedForward(nn.Module):
    def __init__(self, dims, expansion_factor, dropout=0.0, use_single_layer=False):
        super().__init__()
        hidden = max(1, int(expansion_factor * dims))
        self.use_single_layer = use_single_layer
        self.linear_0 = nn.Linear(dims, dims if use_single_layer else hidden)
        self.linear_1 = None if use_single_layer else nn.Linear(hidden, dims)
        self.dropout = dropout

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.linear_1 is not None:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self,
        per_graph_size,
        dims,
        token_expansion_factor=0.5,
        channel_expansion_factor=4.0,
        dropout=0.1,
        use_single_layer=False,
    ):
        super().__init__()
        self.token_layernorm = nn.LayerNorm(dims)
        self.token_forward = FeedForward(
            per_graph_size, token_expansion_factor, dropout, use_single_layer
        )
        self.channel_layernorm = nn.LayerNorm(dims)
        self.channel_forward = FeedForward(
            dims, channel_expansion_factor, dropout, use_single_layer
        )

    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x

    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        return self.channel_forward(x)

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class MultiScaleTimeEncode(nn.Module):
    def __init__(self, dim, t_min=1.0, t_max=1.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time_dim must be even, got {dim}")
        t_min = max(float(t_min), 1e-6)
        t_max = max(float(t_max), t_min)
        freqs = torch.exp(
            torch.linspace(
                math.log(2.0 * math.pi / t_max),
                math.log(2.0 * math.pi / t_min),
                dim // 2,
            )
        )
        self.register_buffer("freqs", freqs.float())

    def forward(self, delta_t):
        angles = delta_t.unsqueeze(-1) * self.freqs
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class PeriodicTimeEncode(nn.Module):
    def __init__(self, periods, num_harmonics=1):
        super().__init__()
        if not periods:
            raise ValueError("periods must be non-empty when absolute time encoding is enabled")
        if num_harmonics <= 0:
            raise ValueError("num_harmonics must be positive")
        freqs = []
        for period in periods:
            period = max(float(period), 1e-6)
            for harmonic in range(1, int(num_harmonics) + 1):
                freqs.append(2.0 * math.pi * harmonic / period)
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
        self.output_dim = 2 * len(freqs)

    def forward(self, event_time):
        angles = event_time.unsqueeze(-1) * self.freqs
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class EventMLPMixer(nn.Module):
    def __init__(
        self,
        topk,
        event_dim,
        num_layers,
        token_expansion_factor,
        channel_expansion_factor,
        dropout,
        use_single_layer=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    per_graph_size=topk,
                    dims=event_dim,
                    token_expansion_factor=token_expansion_factor,
                    channel_expansion_factor=channel_expansion_factor,
                    dropout=dropout,
                    use_single_layer=use_single_layer,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(event_dim)
        self.out_proj = nn.Linear(event_dim, event_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward_tokens(self, x, mask=None):
        mask_f = None if mask is None else mask.unsqueeze(-1).float()
        if mask_f is not None:
            x = x * mask_f
        for block in self.blocks:
            x = block(x)
            if mask_f is not None:
                x = x * mask_f
        return x

    def pool_tokens(self, x, mask=None):
        x = self.out_norm(x)
        if mask is not None:
            weights = mask.unsqueeze(-1).float()
            denom = weights.sum(dim=1).clamp_min(1.0)
            x = (x * weights).sum(dim=1) / denom
        else:
            x = x.mean(dim=1)
        x = self.out_proj(x)
        return self.out_dropout(x)

    def forward(self, x, mask=None):
        return self.pool_tokens(self.forward_tokens(x, mask), mask)


class EventTransformerEncoder(nn.Module):
    def __init__(self, topk, event_dim, num_layers, num_heads, ff_dim, dropout):
        super().__init__()
        if event_dim % num_heads != 0:
            raise ValueError(
                f"event_dim ({event_dim}) must be divisible by transformer_heads ({num_heads})"
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, event_dim))
        self.pos_embedding = nn.Embedding(topk + 1, event_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=event_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(event_dim)
        self.out_dropout = nn.Dropout(dropout)

    def _with_cls(self, x, mask):
        batch_size, seq_len, _ = x.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)
        if mask is None:
            return x, None
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device)
        full_mask = torch.cat((cls_mask, mask), dim=1)
        return x, ~full_mask

    def forward_tokens(self, x, mask=None):
        x, key_padding_mask = self._with_cls(x, mask)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x[:, 1:]

    def pool_tokens(self, x, mask=None):
        if mask is not None:
            weights = mask.unsqueeze(-1).float()
            denom = weights.sum(dim=1).clamp_min(1.0)
            x = (x * weights).sum(dim=1) / denom
        else:
            x = x.mean(dim=1)
        return self.out_dropout(self.out_norm(x))

    def forward(self, x, mask=None):
        x, key_padding_mask = self._with_cls(x, mask)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.out_dropout(self.out_norm(x[:, 0]))


class RecentEventEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_rels,
        topk,
        time_dim,
        rel_dim,
        event_dim,
        dropout,
        t_min,
        t_max,
        use_neighbor_id=False,
        node_dim=64,
        num_layers=1,
        token_expansion_factor=0.5,
        channel_expansion_factor=4.0,
        use_single_layer=False,
        relation_embedding=None,
        use_abs_time=False,
        abs_time_periods=None,
        abs_time_harmonics=1,
        use_query_gate=False,
        query_gate_type="channel",
        use_rank_pos=False,
        encoder_backend="mixer",
        transformer_heads=2,
        transformer_ff_dim=None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.topk = topk
        self.use_neighbor_id = use_neighbor_id
        self.use_abs_time = use_abs_time
        self.use_query_gate = use_query_gate
        self.query_gate_type = query_gate_type
        self.use_rank_pos = use_rank_pos
        self.time_encoder = MultiScaleTimeEncode(time_dim, t_min=t_min, t_max=t_max)
        self.relation_embedding = relation_embedding or nn.Embedding(
            num_rels + 1, rel_dim, padding_idx=num_rels
        )
        input_dim = time_dim + rel_dim
        if use_abs_time:
            self.abs_time_encoder = PeriodicTimeEncode(
                periods=abs_time_periods or [1.0, 7.0, 30.0],
                num_harmonics=abs_time_harmonics,
            )
            input_dim += self.abs_time_encoder.output_dim
        else:
            self.abs_time_encoder = None
        if use_neighbor_id:
            self.node_embedding = nn.Embedding(
                num_nodes + 1, node_dim, padding_idx=num_nodes
            )
            input_dim += node_dim
        else:
            self.node_embedding = None

        self.event_proj = nn.Linear(input_dim, event_dim)
        self.event_norm = nn.LayerNorm(event_dim)
        self.event_dropout = nn.Dropout(dropout)

        if use_query_gate:
            if query_gate_type == "scalar":
                gate_dim = 1
            elif query_gate_type == "channel":
                gate_dim = event_dim
            else:
                raise ValueError(f"unknown query_gate_type: {query_gate_type}")
            self.query_gate = nn.Sequential(
                nn.Linear(rel_dim * 2, event_dim),
                nn.LayerNorm(event_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(event_dim, gate_dim),
                nn.Sigmoid(),
            )
        else:
            self.query_gate = None

        if use_rank_pos:
            self.rank_embedding = nn.Embedding(topk, event_dim)
        else:
            self.rank_embedding = None

        if encoder_backend == "mixer":
            self.sequence_encoder = EventMLPMixer(
                topk=topk,
                event_dim=event_dim,
                num_layers=num_layers,
                token_expansion_factor=token_expansion_factor,
                channel_expansion_factor=channel_expansion_factor,
                dropout=dropout,
                use_single_layer=use_single_layer,
            )
        elif encoder_backend == "transformer":
            self.sequence_encoder = EventTransformerEncoder(
                topk=topk,
                event_dim=event_dim,
                num_layers=num_layers,
                num_heads=transformer_heads,
                ff_dim=transformer_ff_dim or event_dim * 4,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown encoder_backend: {encoder_backend}")

    def build_event_inputs(self, delta_t, rel_ids, neighbor_ids, event_times, mask, query_rel=None):
        mask_f = mask.unsqueeze(-1).float()
        time_feat = self.time_encoder(delta_t) * mask_f
        rel_feat = self.relation_embedding(rel_ids)
        pieces = [time_feat, rel_feat]
        if self.abs_time_encoder is not None:
            pieces.append(self.abs_time_encoder(event_times) * mask_f)
        if self.node_embedding is not None:
            pieces.append(self.node_embedding(neighbor_ids))
        x = torch.cat(pieces, dim=-1)
        x = self.event_proj(x)
        x = self.event_norm(x)
        x = self.event_dropout(x)

        if self.query_gate is not None:
            if query_rel is None:
                raise ValueError("query_rel is required when query gate is enabled")
            q = self.relation_embedding(query_rel).unsqueeze(1).expand(-1, self.topk, -1)
            gate_rel = self.relation_embedding(rel_ids)
            gate = self.query_gate(torch.cat((q, gate_rel), dim=-1))
            x = x * gate

        if self.rank_embedding is not None:
            ranks = torch.arange(self.topk, device=x.device)
            x = x + self.rank_embedding(ranks).unsqueeze(0)

        x = x * mask_f
        return x

    def encode_tokens(self, delta_t, rel_ids, neighbor_ids, event_times, mask, query_rel=None):
        x = self.build_event_inputs(delta_t, rel_ids, neighbor_ids, event_times, mask, query_rel)
        tokens = self.sequence_encoder.forward_tokens(x, mask)
        return tokens * mask.unsqueeze(-1).float()

    def pool_tokens(self, tokens, mask):
        return self.sequence_encoder.pool_tokens(tokens, mask)

    def forward(self, delta_t, rel_ids, neighbor_ids, event_times, mask, query_rel=None):
        x = self.build_event_inputs(delta_t, rel_ids, neighbor_ids, event_times, mask, query_rel)
        return self.sequence_encoder(x, mask)


class MultiWindowRecentEventEncoder(nn.Module):
    def __init__(self, windows, **encoder_kwargs):
        super().__init__()
        self.windows = [int(w) for w in windows]
        self.encoders = nn.ModuleList(
            [
                RecentEventEncoder(topk=window, **encoder_kwargs)
                for window in self.windows
            ]
        )
        self.output_dim = encoder_kwargs["event_dim"] * len(self.windows)

    def _slice_inputs(self, inputs, window):
        delta_t, rel_ids, neighbor_ids, event_times, mask = inputs
        return (
            delta_t[:, :window],
            rel_ids[:, :window],
            neighbor_ids[:, :window],
            event_times[:, :window],
            mask[:, :window],
        )

    def forward(self, delta_t, rel_ids, neighbor_ids, event_times, mask, query_rel=None):
        inputs = (delta_t, rel_ids, neighbor_ids, event_times, mask)
        outputs = []
        for window, encoder in zip(self.windows, self.encoders):
            sliced = self._slice_inputs(inputs, window)
            outputs.append(encoder(*sliced, query_rel=query_rel))
        return torch.cat(outputs, dim=-1)

    def encode_tokens(self, delta_t, rel_ids, neighbor_ids, event_times, mask, query_rel=None):
        inputs = (delta_t, rel_ids, neighbor_ids, event_times, mask)
        token_chunks = []
        mask_chunks = []
        for window, encoder in zip(self.windows, self.encoders):
            sliced = self._slice_inputs(inputs, window)
            token_chunks.append(encoder.encode_tokens(*sliced, query_rel=query_rel))
            mask_chunks.append(sliced[-1])
        return torch.cat(token_chunks, dim=1), torch.cat(mask_chunks, dim=1)

    def pool_tokens(self, tokens, mask):
        outputs = []
        start = 0
        for window, encoder in zip(self.windows, self.encoders):
            end = start + window
            outputs.append(encoder.pool_tokens(tokens[:, start:end], mask[:, start:end]))
            start = end
        return torch.cat(outputs, dim=-1)


class RelationAwareEdgePredictor(nn.Module):
    def __init__(
        self,
        event_dim,
        rel_dim,
        static_node_dim,
        hidden_dim,
        dropout,
        predictor_mode,
        relation_embedding,
    ):
        super().__init__()
        self.predictor_mode = predictor_mode
        self.relation_embedding = relation_embedding
        if predictor_mode == "diag":
            self.rel_proj = nn.Sequential(
                nn.Linear(rel_dim, event_dim),
                nn.LayerNorm(event_dim),
                nn.Dropout(dropout),
            )
            input_dim = event_dim + event_dim + rel_dim
        elif predictor_mode == "concat":
            self.rel_proj = None
            input_dim = event_dim + event_dim + rel_dim
        else:
            raise ValueError(f"unknown predictor_mode: {predictor_mode}")
        input_dim += 2 * int(static_node_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_src, h_dst, query_rel, src_node_emb, dst_node_emb):
        r = self.relation_embedding(query_rel)
        if self.predictor_mode == "diag":
            h_src = h_src * self.rel_proj(r)
        x = torch.cat((h_src, h_dst, r, src_node_emb, dst_node_emb), dim=-1)
        return self.mlp(x).squeeze(-1)


class TKGTimeMixer(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_rels,
        topk=15,
        time_dim=100,
        rel_dim=64,
        node_dim=64,
        event_dim=128,
        hidden_dim=256,
        num_layers=1,
        dropout=0.1,
        t_min=1.0,
        t_max=1.0,
        use_neighbor_id=False,
        token_expansion_factor=0.5,
        channel_expansion_factor=4.0,
        use_single_layer=False,
        predictor_mode="diag",
        use_abs_time=False,
        abs_time_periods=None,
        abs_time_harmonics=1,
        use_query_gate=False,
        query_gate_type="channel",
        use_rank_pos=False,
        multi_windows=None,
        encoder_backend="mixer",
        transformer_heads=2,
        transformer_ff_dim=None,
        use_cross_history=False,
        cross_heads=2,
    ):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_rels + 1, rel_dim, padding_idx=num_rels)
        self.entity_embedding = nn.Embedding(num_nodes + 1, node_dim, padding_idx=num_nodes)
        self.num_nodes = int(num_nodes)
        self.use_cross_history = use_cross_history
        self.use_query_gate = use_query_gate
        encoder_kwargs = dict(
            num_nodes=num_nodes,
            num_rels=num_rels,
            time_dim=time_dim,
            rel_dim=rel_dim,
            event_dim=event_dim,
            dropout=dropout,
            t_min=t_min,
            t_max=t_max,
            use_neighbor_id=use_neighbor_id,
            node_dim=node_dim,
            num_layers=num_layers,
            token_expansion_factor=token_expansion_factor,
            channel_expansion_factor=channel_expansion_factor,
            use_single_layer=use_single_layer,
            relation_embedding=self.relation_embedding,
            use_abs_time=use_abs_time,
            abs_time_periods=abs_time_periods,
            abs_time_harmonics=abs_time_harmonics,
            use_query_gate=use_query_gate,
            query_gate_type=query_gate_type,
            use_rank_pos=use_rank_pos,
            encoder_backend=encoder_backend,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
        )
        multi_windows = [int(w) for w in (multi_windows or []) if int(w) > 0]
        if multi_windows:
            self.encoder = MultiWindowRecentEventEncoder(
                windows=multi_windows,
                **encoder_kwargs,
            )
            node_repr_dim = self.encoder.output_dim
        else:
            self.encoder = RecentEventEncoder(topk=topk, **encoder_kwargs)
            node_repr_dim = event_dim

        if use_cross_history:
            if event_dim % cross_heads != 0:
                raise ValueError(
                    f"event_dim ({event_dim}) must be divisible by cross_heads ({cross_heads})"
                )
            self.cross_s_to_o = nn.MultiheadAttention(
                event_dim, cross_heads, dropout=dropout, batch_first=True
            )
            self.cross_norm_s = nn.LayerNorm(event_dim)
            self.cross_dropout = nn.Dropout(dropout)
        else:
            self.cross_s_to_o = None

        self.predictor = RelationAwareEdgePredictor(
            event_dim=node_repr_dim,
            rel_dim=rel_dim,
            static_node_dim=node_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            predictor_mode=predictor_mode,
            relation_embedding=self.relation_embedding,
        )

    @staticmethod
    def _key_padding_mask(valid_mask):
        key_padding_mask = ~valid_mask.bool()
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False
        return key_padding_mask

    def _entity_embed(self, node_ids, device):
        node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=device).reshape(-1)
        node_ids = node_ids.clamp(min=0, max=self.num_nodes)
        return self.entity_embedding(node_ids)

    @staticmethod
    def _masked_token_mean(tokens, mask):
        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (tokens * weights).sum(dim=1, keepdim=True) / denom

    def encode_source(self, src_inputs, query_rel):
        src_delta, src_rel, src_neighbor, src_event_time, src_mask = src_inputs
        if self.use_cross_history:
            src_tokens = self.encoder.encode_tokens(
                src_delta, src_rel, src_neighbor, src_event_time, src_mask, query_rel=query_rel
            )
            if isinstance(src_tokens, tuple):
                src_tokens, src_token_mask = src_tokens
            else:
                src_token_mask = src_mask
            return "tokens", src_tokens, src_token_mask

        h_src = self.encoder(
            src_delta, src_rel, src_neighbor, src_event_time, src_mask, query_rel=query_rel
        )
        return "repr", h_src, None

    def score_candidates_from_source(
        self,
        source_encoded,
        dst_inputs,
        query_rel,
        num_candidates,
        source_nodes,
        candidate_nodes,
    ):
        dst_delta, dst_rel, dst_neighbor, dst_event_time, dst_mask = dst_inputs
        batch_size = query_rel.shape[0]
        query_rel_expanded = (
            query_rel.unsqueeze(1)
            .expand(batch_size, num_candidates)
            .reshape(batch_size * num_candidates)
        )

        if self.use_cross_history:
            _, src_tokens, src_token_mask = source_encoded
            dst_tokens = self.encoder.encode_tokens(
                dst_delta,
                dst_rel,
                dst_neighbor,
                dst_event_time,
                dst_mask,
                query_rel=query_rel_expanded,
            )
            if isinstance(dst_tokens, tuple):
                dst_tokens, dst_token_mask = dst_tokens
            else:
                dst_token_mask = dst_mask

            h_dst = self.encoder.pool_tokens(dst_tokens, dst_token_mask)
            dst_context = self._masked_token_mean(dst_tokens, dst_token_mask)
            src_tokens = (
                src_tokens.unsqueeze(1)
                .expand(batch_size, num_candidates, src_tokens.shape[1], src_tokens.shape[2])
                .reshape(batch_size * num_candidates, src_tokens.shape[1], src_tokens.shape[2])
            )
            src_token_mask = (
                src_token_mask.unsqueeze(1)
                .expand(batch_size, num_candidates, src_token_mask.shape[1])
                .reshape(batch_size * num_candidates, src_token_mask.shape[1])
            )

            src_cross, _ = self.cross_s_to_o(
                self.cross_norm_s(src_tokens),
                dst_context,
                dst_context,
                need_weights=False,
            )
            src_cross = src_cross * dst_token_mask.any(dim=1).view(-1, 1, 1).float()
            src_tokens = (src_tokens + self.cross_dropout(src_cross)) * src_token_mask.unsqueeze(-1).float()
            h_src = self.encoder.pool_tokens(src_tokens, src_token_mask)
        else:
            _, h_src, _ = source_encoded
            h_dst = self.encoder(
                dst_delta,
                dst_rel,
                dst_neighbor,
                dst_event_time,
                dst_mask,
                query_rel=query_rel_expanded,
            )
            h_src = (
                h_src.unsqueeze(1)
                .expand(batch_size, num_candidates, h_src.shape[-1])
                .reshape(batch_size * num_candidates, -1)
            )

        src_node_expanded = (
            np.asarray(source_nodes, dtype=np.int64)
            .reshape(batch_size, 1)
            .repeat(num_candidates, axis=1)
            .reshape(-1)
        )
        dst_node_flat = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
        src_node_emb = self._entity_embed(src_node_expanded, query_rel.device)
        dst_node_emb = self._entity_embed(dst_node_flat, query_rel.device)
        scores = self.predictor(
            h_src,
            h_dst,
            query_rel_expanded,
            src_node_emb,
            dst_node_emb,
        )
        return scores.view(batch_size, num_candidates)

    def score_representations(
        self,
        h_src,
        h_dst,
        query_rel,
        num_candidates,
        source_nodes,
        candidate_nodes,
    ):
        batch_size = query_rel.shape[0]
        query_rel_expanded = (
            query_rel.unsqueeze(1)
            .expand(batch_size, num_candidates)
            .reshape(batch_size * num_candidates)
        )
        h_src = (
            h_src.unsqueeze(1)
            .expand(batch_size, num_candidates, h_src.shape[-1])
            .reshape(batch_size * num_candidates, -1)
        )
        src_node_expanded = (
            np.asarray(source_nodes, dtype=np.int64)
            .reshape(batch_size, 1)
            .repeat(num_candidates, axis=1)
            .reshape(-1)
        )
        dst_node_flat = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
        src_node_emb = self._entity_embed(src_node_expanded, query_rel.device)
        dst_node_emb = self._entity_embed(dst_node_flat, query_rel.device)
        scores = self.predictor(
            h_src,
            h_dst,
            query_rel_expanded,
            src_node_emb,
            dst_node_emb,
        )
        return scores.view(batch_size, num_candidates)

    def score_candidates(
        self,
        src_inputs,
        dst_inputs,
        query_rel,
        num_candidates,
        source_nodes,
        candidate_nodes,
    ):
        source_encoded = self.encode_source(src_inputs, query_rel)
        return self.score_candidates_from_source(
            source_encoded,
            dst_inputs,
            query_rel,
            num_candidates,
            source_nodes,
            candidate_nodes,
        )


class RecentEventStore:
    def __init__(self, num_nodes, topk, store_neighbor=False, store_abs_time=False):
        self.num_nodes = int(num_nodes)
        self.topk = int(topk)
        self.store_neighbor = bool(store_neighbor)
        self.store_abs_time = bool(store_abs_time)
        self.version = 0
        shape = (self.num_nodes, self.topk)
        self.event_t = np.zeros(shape, dtype=np.float32)
        self.rel = np.zeros(shape, dtype=np.int32)
        self.neighbor = (
            np.zeros(shape, dtype=np.int32) if self.store_neighbor else None
        )
        self.event_abs_t = (
            np.zeros(shape, dtype=np.float32) if self.store_abs_time else None
        )
        self.count = np.zeros(self.num_nodes, dtype=np.int32)
        self.write_pos = np.zeros(self.num_nodes, dtype=np.int32)

    def update(self, events, t_norm, t_abs=None):
        if self.topk <= 0:
            return
        t_value = float(t_norm)
        abs_value = t_value if t_abs is None else float(t_abs)
        for s, r, o in events.astype(np.int64, copy=False):
            if 0 <= s < self.num_nodes:
                s = int(s)
                pos = int(self.write_pos[s])
                self.event_t[s, pos] = t_value
                self.rel[s, pos] = int(r)
                if self.neighbor is not None:
                    self.neighbor[s, pos] = int(o)
                if self.event_abs_t is not None:
                    self.event_abs_t[s, pos] = abs_value
                self.write_pos[s] = (pos + 1) % self.topk
                if self.count[s] < self.topk:
                    self.count[s] += 1
        self.version += 1

    def get_recent(self, nodes, current_t, num_rels, num_nodes):
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        n = len(nodes)
        deltas = np.zeros((n, self.topk), dtype=np.float32)
        rels = np.full((n, self.topk), int(num_rels), dtype=np.int64)
        neigh = np.full((n, self.topk), int(num_nodes), dtype=np.int64)
        event_times = np.zeros((n, self.topk), dtype=np.float32)
        mask = np.zeros((n, self.topk), dtype=np.bool_)
        if n == 0 or self.topk <= 0:
            return deltas, rels, neigh, event_times, mask

        valid_nodes = (nodes >= 0) & (nodes < self.num_nodes)
        if not np.any(valid_nodes):
            return deltas, rels, neigh, event_times, mask

        rows = nodes[valid_nodes].astype(np.int64, copy=False)
        offsets = np.arange(self.topk, dtype=np.int64)
        indices = (self.write_pos[rows].astype(np.int64)[:, None] - 1 - offsets[None, :]) % self.topk
        row_counts = self.count[rows]
        row_mask = offsets[None, :] < row_counts[:, None]

        event_t = self.event_t[rows[:, None], indices]
        rel_values = self.rel[rows[:, None], indices]
        out_rows = np.flatnonzero(valid_nodes)

        deltas[out_rows] = np.maximum(0.0, float(current_t) - event_t) * row_mask
        valid_rel = row_mask & (rel_values >= 0) & (rel_values < int(num_rels))
        rel_block = rels[out_rows]
        rel_block[valid_rel] = rel_values[valid_rel]
        rels[out_rows] = rel_block

        if self.neighbor is not None:
            neighbor_values = self.neighbor[rows[:, None], indices]
            valid_neighbor = row_mask & (neighbor_values >= 0) & (neighbor_values < int(num_nodes))
            neigh_block = neigh[out_rows]
            neigh_block[valid_neighbor] = neighbor_values[valid_neighbor]
            neigh[out_rows] = neigh_block

        if self.event_abs_t is not None:
            event_times[out_rows] = self.event_abs_t[rows[:, None], indices] * row_mask
        else:
            event_times[out_rows] = event_t * row_mask
        mask[out_rows] = row_mask
        return deltas, rels, neigh, event_times, mask


def make_recent_store(args, num_nodes):
    return RecentEventStore(
        num_nodes,
        get_history_topk(args),
        store_neighbor=getattr(args, "use_neighbor_id", False),
        store_abs_time=getattr(args, "use_abs_time", False),
    )


def infer_id_sizes(data, split_names=("train_list",)):
    max_node = -1
    max_rel = -1
    for split_name in split_names:
        for events, _, _ in data[split_name]:
            if len(events) == 0:
                continue
            max_node = max(max_node, int(events[:, [0, 2]].max()))
            max_rel = max(max_rel, int(events[:, 1].max()))

    sampler = get_negative_sampler(data)
    if hasattr(sampler, "last_dst_id"):
        max_node = max(max_node, int(sampler.last_dst_id))

    num_nodes = max(int(data["num_nodes"]), max_node + 1)
    num_rels = max(int(data["num_rels"]), max_rel + 1)
    return num_nodes, num_rels


def infer_time_bounds(data, split_names=("train_list", "val_list", "test_list"), use_raw=False):
    values = []
    for split_name in split_names:
        if use_raw:
            values.extend(float(t_orig) for _, _, t_orig in data[split_name])
        else:
            values.extend(float(t_norm) for _, t_norm, _ in data[split_name])
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def format_periods(periods):
    return ",".join(f"{float(period):g}" for period in periods)


def infer_calendar_periods_for_time_source(use_raw, norm_span, raw_span):
    raw_looks_seconds = raw_span >= 100000.0
    if use_raw:
        return DEFAULT_SECONDS_ABS_PERIODS if raw_looks_seconds else DEFAULT_CALENDAR_ABS_PERIODS
    if raw_span > 0.0 and norm_span > 0.0:
        raw_per_norm = raw_span / norm_span
        if raw_looks_seconds:
            return [period / max(raw_per_norm, 1e-6) for period in DEFAULT_SECONDS_ABS_PERIODS]
        if norm_span <= 2.0:
            return [period / max(raw_span, 1e-6) for period in DEFAULT_CALENDAR_ABS_PERIODS]
    return DEFAULT_CALENDAR_ABS_PERIODS


def resolve_abs_time_periods(args, norm_bounds, raw_bounds):
    if not getattr(args, "use_abs_time", False):
        return []

    user_periods = parse_float_list(getattr(args, "abs_time_periods", None))
    norm_span = max(float(norm_bounds[1]) - float(norm_bounds[0]), 0.0)
    raw_span = max(float(raw_bounds[1]) - float(raw_bounds[0]), 0.0)
    if user_periods:
        periods = user_periods
        source = "user"
    else:
        periods = infer_calendar_periods_for_time_source(
            getattr(args, "abs_time_use_raw", False),
            norm_span,
            raw_span,
        )
        args.abs_time_periods = format_periods(periods)
        source = "default"

    value_source = "raw" if getattr(args, "abs_time_use_raw", False) else "norm"
    value_bounds = raw_bounds if value_source == "raw" else norm_bounds
    raw_per_norm = raw_span / norm_span if norm_span > 0.0 else float("nan")
    print(
        f"[TimeTKG] abs_time source={value_source} "
        f"range=[{value_bounds[0]:g}, {value_bounds[1]:g}] "
        f"span={value_bounds[1] - value_bounds[0]:g} "
        f"periods={format_periods(periods)} ({source}); "
        f"raw_span={raw_span:g} norm_span={norm_span:g} "
        f"raw_per_norm={raw_per_norm:g}",
        flush=True,
    )
    if value_source == "norm" and norm_span <= 2.0:
        print(
            "[TimeTKG] abs_time note: normalized time span is small; "
            "default periods were scaled from raw-time span. "
            "If you pass --abs_time_periods manually, use normalized-time units.",
            flush=True,
        )
    if value_source == "norm" and source == "user" and raw_span >= 100000.0:
        print(
            "[TimeTKG] abs_time note: user periods are interpreted in normalized-time units; "
            f"one normalized unit is about {raw_per_norm:g} raw seconds.",
            flush=True,
        )
    if value_source == "raw" and raw_span >= 100000.0:
        print(
            "[TimeTKG] abs_time note: raw time looks second-scale; "
            "periods should be in seconds, e.g. 86400 for one day.",
            flush=True,
        )
    return periods


def load_time_tkg_data(args):
    return load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )


EMPTY_INT64 = np.empty(0, dtype=np.int64)


class RelationObjectHistory:
    def __init__(self, num_rels):
        self.pools = [set() for _ in range(num_rels)]
        self.arrays = [EMPTY_INT64 for _ in range(num_rels)]
        self.dirty = [False for _ in range(num_rels)]

    def get(self, relation):
        relation = int(relation)
        if relation < 0 or relation >= len(self.pools):
            return EMPTY_INT64
        if self.dirty[relation]:
            pool = self.pools[relation]
            self.arrays[relation] = (
                np.array(sorted(pool), dtype=np.int64) if pool else EMPTY_INT64
            )
            self.dirty[relation] = False
        return self.arrays[relation]

    def update(self, events):
        for _, r, o in events.astype(np.int64, copy=False):
            r = int(r)
            if r < 0 or r >= len(self.pools):
                continue
            before = len(self.pools[r])
            self.pools[r].add(int(o))
            if len(self.pools[r]) != before:
                self.dirty[r] = True


def build_timestamp_positive_objects(events):
    groups = {}
    for s, r, o in events.astype(np.int64, copy=False):
        key = (int(s), int(r))
        values = groups.get(key)
        if values is None:
            values = set()
            groups[key] = values
        values.add(int(o))
    return {key: frozenset(values) for key, values in groups.items()}


def filter_pool(pool, exclude):
    if len(pool) == 0:
        return EMPTY_INT64
    if not exclude:
        return pool
    exclude_arr = np.fromiter((int(x) for x in exclude), dtype=np.int64)
    return pool[~np.isin(pool, exclude_arr, assume_unique=True)]


def sample_from_candidates(candidates, count, rng):
    count = int(count)
    if count <= 0 or len(candidates) == 0:
        return EMPTY_INT64
    if len(candidates) <= count:
        return candidates.astype(np.int64, copy=True)
    idx = rng.choice(len(candidates), size=count, replace=False)
    return candidates[idx].astype(np.int64, copy=False)


def sample_from_pool_excluding(pool, count, exclude, rng):
    count = int(count)
    if count <= 0 or len(pool) == 0:
        return EMPTY_INT64

    if isinstance(exclude, (set, frozenset)):
        exclude_set = exclude
    else:
        exclude_set = set(int(x) for x in exclude)
    if len(pool) <= count + len(exclude_set) + 8:
        return sample_from_candidates(filter_pool(pool, exclude_set), count, rng)

    selected = []
    seen = set()
    max_attempts = max(64, count * 16)
    for _ in range(max_attempts):
        value = int(pool[rng.randint(0, len(pool))])
        if value in exclude_set or value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) == count:
            return np.array(selected, dtype=np.int64)

    remaining = filter_pool(pool, exclude_set | seen)
    extra = sample_from_candidates(remaining, count - len(selected), rng)
    if len(selected) == 0:
        return extra
    if len(extra) == 0:
        return np.array(selected, dtype=np.int64)
    return np.concatenate((np.array(selected, dtype=np.int64), extra))


def pad_negative_row(selected, num_neg, filler_pool, rng, fallback=0):
    if len(selected) >= num_neg:
        return selected[:num_neg].astype(np.int64, copy=False)

    need = int(num_neg) - len(selected)
    if len(filler_pool) > 0:
        fill = rng.choice(filler_pool, size=need, replace=True).astype(np.int64)
    elif len(selected) > 0:
        fill = rng.choice(selected, size=need, replace=True).astype(np.int64)
    else:
        fill = np.full(need, int(fallback), dtype=np.int64)
    if len(selected) == 0:
        return fill
    return np.concatenate((selected.astype(np.int64, copy=False), fill))


def sample_train_negatives(
    events,
    rel_history,
    dst_pool,
    num_neg,
    hard_ratio,
    rng,
    timestamp_positive_objects,
    candidate_cache,
):
    events = events.astype(np.int64, copy=False)
    neg = np.empty((len(events), num_neg), dtype=np.int64)
    hard_quota = min(
        int(num_neg),
        max(0, int(math.floor(float(num_neg) * float(hard_ratio) + 1e-12))),
    )
    for i, (s, r, pos_dst) in enumerate(events):
        positive_objects = timestamp_positive_objects.get(
            (int(s), int(r)), frozenset((int(pos_dst),))
        )
        hard_key = ("hard", int(r), positive_objects)
        cached = candidate_cache.get(hard_key)
        if cached is None:
            hard_candidates = filter_pool(rel_history.get(int(r)), positive_objects)
            random_exclude = set(positive_objects)
            random_exclude.update(int(x) for x in hard_candidates)
            candidate_cache[hard_key] = (hard_candidates, random_exclude)
        else:
            hard_candidates, random_exclude = cached

        hard_selected = sample_from_candidates(hard_candidates, hard_quota, rng)
        random_quota = int(num_neg) - len(hard_selected)
        random_selected = sample_from_pool_excluding(
            dst_pool, random_quota, random_exclude, rng
        )

        if len(hard_selected) == 0:
            selected = random_selected
        elif len(random_selected) == 0:
            selected = hard_selected
        else:
            selected = np.concatenate((hard_selected, random_selected))
        if len(selected) < int(num_neg):
            filler_pool = filter_pool(dst_pool, random_exclude)
        else:
            filler_pool = EMPTY_INT64
        neg[i] = pad_negative_row(
            selected,
            num_neg,
            filler_pool,
            rng,
            fallback=int(pos_dst),
        )
    return neg


def random_from_pool(pool, shape, rng, fallback):
    if len(pool) == 0:
        return np.full(shape, int(fallback), dtype=np.int64)
    idx = rng.randint(0, len(pool), size=shape)
    return pool[idx].astype(np.int64, copy=False)


def repair_own_positive_negatives(neg, pos_dst, dst_pool, rng, max_rounds=8):
    if neg.size == 0:
        return neg
    pos_dst = np.asarray(pos_dst, dtype=np.int64).reshape(-1, 1)
    invalid = neg == pos_dst
    rounds = 0
    while np.any(invalid) and len(dst_pool) > 1 and rounds < int(max_rounds):
        neg[invalid] = random_from_pool(dst_pool, int(invalid.sum()), rng, 0)
        invalid = neg == pos_dst
        rounds += 1
    if np.any(invalid) and len(dst_pool) > 0:
        rows, cols = np.where(invalid)
        for offset in range(min(len(dst_pool), 16)):
            pending = neg[rows, cols] == pos_dst.reshape(-1)[rows]
            if not np.any(pending):
                break
            candidate = dst_pool[(np.arange(int(pending.sum())) + offset) % len(dst_pool)]
            target_rows = rows[pending]
            target_cols = cols[pending]
            ok = candidate != pos_dst.reshape(-1)[target_rows]
            neg[target_rows[ok], target_cols[ok]] = candidate[ok]
    return neg


def sample_train_negatives_fast(
    events,
    rel_history,
    dst_pool,
    num_neg,
    hard_ratio,
    rng,
):
    events = events.astype(np.int64, copy=False)
    num_neg = int(num_neg)
    n = len(events)
    if n == 0 or num_neg <= 0:
        return np.empty((n, max(num_neg, 0)), dtype=np.int64)

    neg = random_from_pool(dst_pool, (n, num_neg), rng, 0)
    hard_quota = min(
        num_neg,
        max(0, int(math.floor(float(num_neg) * float(hard_ratio) + 1e-12))),
    )
    if hard_quota > 0:
        relations = events[:, 1]
        for relation in np.unique(relations):
            rows = np.flatnonzero(relations == relation)
            hard_pool = rel_history.get(int(relation))
            if len(rows) == 0 or len(hard_pool) == 0:
                continue
            neg[rows, :hard_quota] = random_from_pool(
                hard_pool,
                (len(rows), hard_quota),
                rng,
                fallback=0,
            )

    return repair_own_positive_negatives(neg, events[:, 2], dst_pool, rng)


def sample_rows_without_replacement(pool, rows, count, rng, max_matrix_mb=256.0):
    rows = int(rows)
    count = int(count)
    if rows <= 0 or count <= 0 or len(pool) == 0:
        return np.empty((rows, 0), dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) <= count:
        return np.tile(pool.reshape(1, -1), (rows, 1))
    if count == 1:
        return pool[rng.randint(0, len(pool), size=(rows, 1))]

    matrix_mb = rows * len(pool) * 8.0 / (1024.0 * 1024.0)
    if matrix_mb <= float(max_matrix_mb):
        keys = rng.random_sample((rows, len(pool)))
        picked = np.argpartition(keys, count - 1, axis=1)[:, :count]
        return pool[picked]

    return np.vstack(
        [rng.choice(pool, size=count, replace=False) for _ in range(rows)]
    ).astype(np.int64, copy=False)


def sample_train_negatives_grouped_exact(
    events,
    rel_history,
    dst_pool,
    num_neg,
    hard_ratio,
    rng,
    timestamp_positive_objects,
    candidate_cache,
    max_matrix_mb=256.0,
):
    events = events.astype(np.int64, copy=False)
    num_neg = int(num_neg)
    neg = np.empty((len(events), num_neg), dtype=np.int64)
    hard_quota = min(
        num_neg,
        max(0, int(math.floor(float(num_neg) * float(hard_ratio) + 1e-12))),
    )

    groups = {}
    for row_idx, (s, r, pos_dst) in enumerate(events):
        positive_objects = timestamp_positive_objects.get(
            (int(s), int(r)), frozenset((int(pos_dst),))
        )
        groups.setdefault((int(r), positive_objects), []).append(row_idx)

    for (relation, positive_objects), row_indices in groups.items():
        rows = np.asarray(row_indices, dtype=np.int64)
        cache_key = ("grouped_exact", relation, positive_objects)
        cached = candidate_cache.get(cache_key)
        if cached is None:
            hard_pool = filter_pool(rel_history.get(relation), positive_objects)
            random_exclude = set(positive_objects)
            random_exclude.update(int(x) for x in hard_pool)
            random_pool = filter_pool(dst_pool, random_exclude)
            cached = (hard_pool, random_pool)
            candidate_cache[cache_key] = cached
        hard_pool, random_pool = cached

        cursor = 0
        if hard_quota > 0:
            hard_block = sample_rows_without_replacement(
                hard_pool,
                len(rows),
                hard_quota,
                rng,
                max_matrix_mb=max_matrix_mb,
            )
            if hard_block.shape[1] > 0:
                neg[rows, : hard_block.shape[1]] = hard_block
                cursor = hard_block.shape[1]

        random_quota = num_neg - cursor
        if random_quota > 0:
            random_block = sample_rows_without_replacement(
                random_pool,
                len(rows),
                random_quota,
                rng,
                max_matrix_mb=max_matrix_mb,
            )
            if random_block.shape[1] > 0:
                end = cursor + random_block.shape[1]
                neg[rows, cursor:end] = random_block
                cursor = end

        if cursor < num_neg:
            need = num_neg - cursor
            if len(random_pool) > 0:
                fill = random_from_pool(random_pool, (len(rows), need), rng, 0)
            elif cursor > 0:
                fill_idx = rng.randint(0, cursor, size=(len(rows), need))
                fill = np.take_along_axis(neg[rows, :cursor], fill_idx, axis=1)
            else:
                fill = events[rows, 2:3].repeat(need, axis=1)
            neg[rows, cursor:num_neg] = fill

    return neg


def histories_to_tensors(nodes, current_t, store, args, device, num_rels, num_nodes):
    deltas, rels, neigh, event_times, mask = store.get_recent(
        nodes,
        current_t,
        num_rels,
        num_nodes,
    )

    return (
        torch.from_numpy(deltas).to(device),
        torch.from_numpy(rels).to(device),
        torch.from_numpy(neigh).to(device),
        torch.from_numpy(event_times).to(device),
        torch.from_numpy(mask).to(device),
    )


def build_candidate_inputs(
    events,
    candidate_nodes,
    current_t,
    store,
    args,
    device,
    num_rels,
    num_nodes,
):
    events = events.astype(np.int64, copy=False)
    candidate_nodes = candidate_nodes.astype(np.int64, copy=False)
    batch_size, num_candidates = candidate_nodes.shape

    src_inputs = histories_to_tensors(
        events[:, 0], current_t, store, args, device, num_rels, num_nodes
    )
    dst_inputs = histories_to_tensors(
        candidate_nodes.reshape(-1),
        current_t,
        store,
        args,
        device,
        num_rels,
        num_nodes,
    )
    query_rel = torch.from_numpy(events[:, 1].astype(np.int64)).to(device)
    return src_inputs, dst_inputs, query_rel, num_candidates


def score_candidate_nodes(
    model,
    events,
    candidate_nodes,
    current_t,
    store,
    args,
    device,
    num_rels,
    num_nodes,
):
    src_inputs, dst_inputs, query_rel, num_candidates = build_candidate_inputs(
        events=events,
        candidate_nodes=candidate_nodes,
        current_t=current_t,
        store=store,
        args=args,
        device=device,
        num_rels=num_rels,
        num_nodes=num_nodes,
    )
    return model.score_candidates(
        src_inputs,
        dst_inputs,
        query_rel,
        num_candidates,
        source_nodes=events[:, 0],
        candidate_nodes=candidate_nodes,
    )


def source_cache_signature(args):
    keys = [
        "dataset",
        "seed",
        "topk",
        "multi_windows",
        "time_dim",
        "rel_dim",
        "node_dim",
        "event_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
        "time_min",
        "token_expansion_factor",
        "channel_expansion_factor",
        "use_single_layer",
        "predictor_mode",
        "use_neighbor_id",
        "use_abs_time",
        "abs_time_periods",
        "abs_time_harmonics",
        "abs_time_use_raw",
        "use_query_gate",
        "query_gate_type",
        "use_rank_pos",
        "use_cross_history",
        "cross_heads",
        "event_encoder",
        "transformer_heads",
        "transformer_ff_dim",
    ]
    payload = {key: getattr(args, key, None) for key in keys}
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


class EvalSourceCache:
    def __init__(self, cache_root=None, mode="eval", args=None):
        self.values = {}
        self.path = None
        self.meta = {"version": 1, "mode": mode}
        if args is not None:
            self.meta["signature"] = source_cache_signature(args)
        if cache_root is not None:
            os.makedirs(cache_root, exist_ok=True)
            signature = self.meta.get("signature", "nosig")
            self.path = osp.join(cache_root, f"{mode}_source_cache_{signature}.pt")
            if osp.exists(self.path) and not getattr(args, "force", False):
                try:
                    payload = torch.load(self.path, map_location="cpu")
                    if payload.get("meta") == self.meta:
                        self.values = payload.get("values", {})
                except Exception:
                    self.values = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value

    def close(self):
        if self.path is not None:
            torch.save({"meta": self.meta, "values": self.values}, self.path)


class EvalNodeCache:
    def __init__(self, num_nodes=None, device=None, dense=True, max_mb=4096.0):
        self.values = {}
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.device = device
        self.dense = bool(dense) and self.num_nodes is not None and device is not None
        self.max_mb = float(max_mb)
        self.tensor = None
        self.present = None
        self.rel_tensors = {}
        self.rel_present = {}
        self.dense_bytes = 0

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value

    def _dense_bytes_needed(self, repr_dim, dtype):
        element_size = torch.empty((), dtype=dtype).element_size()
        bytes_needed = (self.num_nodes + 1) * int(repr_dim) * element_size
        bytes_needed += self.num_nodes + 1
        return int(bytes_needed)

    def _dense_fits(self, repr_dim, dtype):
        if not self.dense:
            return False
        return self._dense_bytes_needed(repr_dim, dtype) <= self.max_mb * 1024 * 1024

    def put_many_dense(self, nodes, encoded, relation=None):
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        if len(nodes) == 0:
            return False
        encoded = encoded.detach()
        repr_dim = int(encoded.shape[-1])
        if relation is not None:
            relation = int(relation)
            if relation not in self.rel_tensors:
                bytes_needed = self._dense_bytes_needed(repr_dim, encoded.dtype)
                if (
                    not self.dense
                    or self.dense_bytes + bytes_needed > self.max_mb * 1024 * 1024
                ):
                    self.dense = False
                    return False
                self.rel_tensors[relation] = torch.zeros(
                    (self.num_nodes + 1, repr_dim),
                    dtype=encoded.dtype,
                    device=self.device,
                )
                self.rel_present[relation] = torch.zeros(
                    self.num_nodes + 1,
                    dtype=torch.bool,
                    device=self.device,
                )
                self.dense_bytes += bytes_needed
            idx = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
            idx = idx.clamp(min=0, max=self.num_nodes)
            self.rel_tensors[relation][idx] = encoded.to(self.device)
            self.rel_present[relation][idx] = True
            return True

        if self.tensor is None:
            if not self._dense_fits(repr_dim, encoded.dtype):
                self.dense = False
                return False
            self.tensor = torch.zeros(
                (self.num_nodes + 1, repr_dim),
                dtype=encoded.dtype,
                device=self.device,
            )
            self.present = torch.zeros(
                self.num_nodes + 1,
                dtype=torch.bool,
                device=self.device,
            )
        idx = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        idx = idx.clamp(min=0, max=self.num_nodes)
        self.tensor[idx] = encoded.to(self.device)
        self.present[idx] = True
        return True

    def get_many_dense(self, nodes, relation=None):
        if relation is not None:
            relation = int(relation)
            if relation not in self.rel_tensors:
                return None
            nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
            idx = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
            idx = idx.clamp(min=0, max=self.num_nodes)
            if not bool(self.rel_present[relation][idx].all().item()):
                return None
            return self.rel_tensors[relation][idx]

        if self.tensor is None:
            return None
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        idx = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        idx = idx.clamp(min=0, max=self.num_nodes)
        if not bool(self.present[idx].all().item()):
            return None
        return self.tensor[idx]

    def missing_dense_nodes(self, nodes, relation=None):
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        valid = (nodes >= 0) & (nodes < self.num_nodes)
        if not np.any(valid):
            return EMPTY_INT64
        nodes = np.unique(nodes[valid])
        if relation is not None:
            relation = int(relation)
            if relation not in self.rel_tensors:
                return nodes.astype(np.int64, copy=False)
            idx = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
            missing = ~self.rel_present[relation][idx]
            if not bool(missing.any().item()):
                return EMPTY_INT64
            return nodes[missing.detach().cpu().numpy()].astype(np.int64, copy=False)

        if self.tensor is None:
            return nodes.astype(np.int64, copy=False)
        idx = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        missing = ~self.present[idx]
        if not bool(missing.any().item()):
            return EMPTY_INT64
        return nodes[missing.detach().cpu().numpy()].astype(np.int64, copy=False)

    def clear(self):
        self.values.clear()
        self.tensor = None
        self.present = None
        self.rel_tensors.clear()
        self.rel_present.clear()
        self.dense_bytes = 0


def can_cache_eval_nodes(model):
    return not getattr(model, "use_cross_history", False)


def source_cache_key(current_t, store_version, source, relation=None):
    if relation is None:
        return (float(current_t), int(store_version), int(source))
    return (float(current_t), int(store_version), int(source), int(relation))


def detach_source_item(source_encoded, idx):
    kind, values, mask = source_encoded
    if kind == "repr":
        return kind, values[idx].detach().cpu(), None
    return kind, values[idx].detach().cpu(), mask[idx].detach().cpu()


def stack_source_items(items, device):
    kind = items[0][0]
    values = torch.stack([item[1] for item in items]).to(device)
    if kind == "repr":
        return kind, values, None
    masks = torch.stack([item[2] for item in items]).to(device)
    return kind, values, masks


def add_metric_aliases(metrics):
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics


def prefix_metrics(prefix, metrics):
    out = {}
    for key, value in metrics.items():
        if key != "profile":
            out[f"{prefix}_{key}"] = float(value)
    return out


def get_eval_source_encoded(
    model,
    batch_data,
    current_t,
    store,
    args,
    device,
    num_rels,
    num_nodes,
    source_cache,
    profile=None,
):
    batch_data = batch_data.astype(np.int64, copy=False)
    include_relation = bool(getattr(model, "use_query_gate", False))
    keys = [
        source_cache_key(
            current_t,
            store.version,
            source,
            relation if include_relation else None,
        )
        for source, relation in batch_data[:, :2]
    ]
    items = [source_cache.get(key) if source_cache is not None else None for key in keys]
    missing = [idx for idx, item in enumerate(items) if item is None]
    profile_add(profile, "source_cache_queries", len(keys))
    profile_add(profile, "source_cache_hits", len(keys) - len(missing))
    profile_add(profile, "source_cache_misses", len(missing))

    if missing:
        missing_events = batch_data[missing]
        t_part = profile_now(args, device)
        src_inputs = histories_to_tensors(
            missing_events[:, 0], current_t, store, args, device, num_rels, num_nodes
        )
        profile_add(profile, "eval_source_history_time", profile_now(args, device) - t_part)
        query_rel = torch.from_numpy(missing_events[:, 1].astype(np.int64)).to(device)
        t_part = forward_tic(profile, args, device)
        encoded = model.encode_source(src_inputs, query_rel)
        forward_toc(profile, args, device, t_part, "eval_source_encode_time")
        for local_idx, batch_idx in enumerate(missing):
            item = detach_source_item(encoded, local_idx)
            items[batch_idx] = item
            if source_cache is not None:
                source_cache.put(keys[batch_idx], item)

    t_part = profile_now(args, device)
    stacked = stack_source_items(items, device)
    profile_add(profile, "eval_source_stack_time", profile_now(args, device) - t_part)
    return stacked


def get_eval_node_representations(
    model,
    nodes,
    current_t,
    store,
    args,
    device,
    num_rels,
    num_nodes,
    node_cache,
    profile=None,
    query_rel=None,
):
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if getattr(model, "use_query_gate", False):
        if query_rel is None:
            raise ValueError("query_rel is required for cached query-gated node encoding")
        rels = np.asarray(query_rel, dtype=np.int64).reshape(-1)
        if len(rels) != len(nodes):
            raise ValueError("query_rel must have the same length as nodes")
        out = None
        for relation in np.unique(rels):
            row_idx = np.flatnonzero(rels == relation)
            encoded = get_eval_node_representations_for_relation(
                model,
                nodes[row_idx],
                current_t,
                store,
                args,
                device,
                num_rels,
                num_nodes,
                node_cache,
                profile=profile,
                relation=int(relation),
            )
            if out is None:
                out = torch.empty(
                    (len(nodes), encoded.shape[-1]),
                    dtype=encoded.dtype,
                    device=device,
                )
            out[torch.as_tensor(row_idx, dtype=torch.long, device=device)] = encoded
        if out is None:
            return torch.empty((0, 0), dtype=torch.float32, device=device)
        return out

    return get_eval_node_representations_for_relation(
        model,
        nodes,
        current_t,
        store,
        args,
        device,
        num_rels,
        num_nodes,
        node_cache,
        profile=profile,
        relation=None,
    )


def get_eval_node_representations_for_relation(
    model,
    nodes,
    current_t,
    store,
    args,
    device,
    num_rels,
    num_nodes,
    node_cache,
    profile=None,
    relation=None,
):
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if node_cache is None:
        profile_add(profile, "node_cache_disabled_requests", len(nodes))
        t_part = profile_now(args, device)
        dst_inputs = histories_to_tensors(
            nodes, current_t, store, args, device, num_rels, num_nodes
        )
        profile_add(profile, "eval_node_history_time", profile_now(args, device) - t_part)
        if relation is None:
            query_rel = torch.zeros(len(nodes), dtype=torch.long, device=device)
        else:
            query_rel = torch.full((len(nodes),), int(relation), dtype=torch.long, device=device)
        t_part = forward_tic(profile, args, device)
        encoded = model.encoder(*dst_inputs, query_rel=query_rel)
        forward_toc(profile, args, device, t_part, "eval_node_encode_time")
        return encoded

    dense_values = node_cache.get_many_dense(nodes, relation=relation)
    if dense_values is not None:
        profile_add(profile, "node_cache_queries", len(nodes))
        profile_add(profile, "node_cache_hits", len(nodes))
        return dense_values.to(device)

    missing_dense = node_cache.missing_dense_nodes(nodes, relation=relation)
    if len(missing_dense) > 0 and node_cache.dense:
        profile_add(profile, "node_cache_unique_encodes", len(missing_dense))
        t_part = profile_now(args, device)
        dst_inputs = histories_to_tensors(
            missing_dense, current_t, store, args, device, num_rels, num_nodes
        )
        profile_add(profile, "eval_node_history_time", profile_now(args, device) - t_part)
        if relation is None:
            query_rel = torch.zeros(len(missing_dense), dtype=torch.long, device=device)
        else:
            query_rel = torch.full((len(missing_dense),), int(relation), dtype=torch.long, device=device)
        t_part = forward_tic(profile, args, device)
        encoded = model.encoder(*dst_inputs, query_rel=query_rel)
        forward_toc(profile, args, device, t_part, "eval_node_encode_time")
        if node_cache.put_many_dense(missing_dense, encoded, relation=relation):
            dense_values = node_cache.get_many_dense(nodes, relation=relation)
            if dense_values is not None:
                profile_add(profile, "node_cache_queries", len(nodes))
                profile_add(profile, "node_cache_hits", len(nodes) - len(missing_dense))
                profile_add(profile, "node_cache_misses", len(missing_dense))
                return dense_values.to(device)

    keys = [
        source_cache_key(current_t, store.version, int(node), relation)
        for node in nodes
    ]
    items = [node_cache.get(key) for key in keys]
    raw_missing = sum(1 for item in items if item is None)
    profile_add(profile, "node_cache_queries", len(keys))
    profile_add(profile, "node_cache_hits", len(keys) - raw_missing)
    profile_add(profile, "node_cache_misses", raw_missing)
    missing_keys = []
    missing_nodes = []
    seen = set()
    for key, node, item in zip(keys, nodes, items):
        if item is None and key not in seen:
            seen.add(key)
            missing_keys.append(key)
            missing_nodes.append(int(node))

    if missing_nodes:
        profile_add(profile, "node_cache_unique_encodes", len(missing_nodes))
        missing_nodes = np.asarray(missing_nodes, dtype=np.int64)
        t_part = profile_now(args, device)
        dst_inputs = histories_to_tensors(
            missing_nodes, current_t, store, args, device, num_rels, num_nodes
        )
        profile_add(profile, "eval_node_history_time", profile_now(args, device) - t_part)
        if relation is None:
            query_rel = torch.zeros(len(missing_nodes), dtype=torch.long, device=device)
        else:
            query_rel = torch.full((len(missing_nodes),), int(relation), dtype=torch.long, device=device)
        t_part = forward_tic(profile, args, device)
        encoded = model.encoder(*dst_inputs, query_rel=query_rel)
        forward_toc(profile, args, device, t_part, "eval_node_encode_time")
        for key, value in zip(missing_keys, encoded.detach().cpu()):
            node_cache.put(key, value)
        items = [node_cache.get(key) for key in keys]

    t_part = profile_now(args, device)
    stacked = torch.stack(items).to(device)
    profile_add(profile, "eval_node_stack_time", profile_now(args, device) - t_part)
    return stacked


def preload_snapshot_eval_nodes(
    model,
    events,
    t_norm,
    t_orig,
    mode,
    neg_sampler,
    store,
    args,
    device,
    num_rels,
    num_nodes,
    node_cache,
    profile=None,
):
    if (
        node_cache is None
        or not getattr(args, "preload_eval_nodes", True)
        or not can_cache_eval_nodes(model)
        or int(getattr(args, "ns_q", -1)) == -1
    ):
        return

    t0 = profile_now(args, device)
    query_gated = getattr(model, "use_query_gate", False)
    seen = {} if query_gated else np.zeros(int(num_nodes), dtype=np.bool_)
    batch_iter = collect_eval_batch(events, t_orig, neg_sampler, mode, args.eval_batch_size)
    for batch_data, neg_arr, neg_mask in batch_iter:
        if query_gated:
            for relation in np.unique(batch_data[:, 1]):
                rows = batch_data[:, 1] == relation
                rel_seen = seen.setdefault(
                    int(relation),
                    np.zeros(int(num_nodes), dtype=np.bool_),
                )
                pos_nodes = batch_data[rows, 2].astype(np.int64, copy=False)
                pos_nodes = pos_nodes[(pos_nodes >= 0) & (pos_nodes < num_nodes)]
                rel_seen[pos_nodes] = True
                if neg_arr.size > 0:
                    rel_mask = neg_mask[rows]
                    neg_nodes = neg_arr[rows][rel_mask].astype(np.int64, copy=False)
                    neg_nodes = neg_nodes[(neg_nodes >= 0) & (neg_nodes < num_nodes)]
                    rel_seen[neg_nodes] = True
        else:
            pos_nodes = batch_data[:, 2].astype(np.int64, copy=False)
            pos_nodes = pos_nodes[(pos_nodes >= 0) & (pos_nodes < num_nodes)]
            seen[pos_nodes] = True
            if neg_arr.size > 0:
                neg_nodes = neg_arr[neg_mask].astype(np.int64, copy=False)
                neg_nodes = neg_nodes[(neg_nodes >= 0) & (neg_nodes < num_nodes)]
                seen[neg_nodes] = True

    chunk_size = max(1, int(getattr(args, "eval_node_preload_chunk", 65536)))
    if query_gated:
        total_nodes = 0
        for relation, rel_seen in seen.items():
            nodes = np.flatnonzero(rel_seen).astype(np.int64, copy=False)
            total_nodes += len(nodes)
            rels = np.full(min(chunk_size, max(len(nodes), 1)), relation, dtype=np.int64)
            for start in range(0, len(nodes), chunk_size):
                chunk = nodes[start : start + chunk_size]
                if len(rels) != len(chunk):
                    rels = np.full(len(chunk), relation, dtype=np.int64)
                get_eval_node_representations(
                    model,
                    chunk,
                    t_norm,
                    store,
                    args,
                    device,
                    num_rels,
                    num_nodes,
                    node_cache,
                    profile=profile,
                    query_rel=rels,
                )
                if not node_cache.dense:
                    break
            if not node_cache.dense:
                break
        profile_add(profile, "eval_node_preload_nodes", total_nodes)
    else:
        nodes = np.flatnonzero(seen).astype(np.int64, copy=False)
        profile_add(profile, "eval_node_preload_nodes", len(nodes))
        for start in range(0, len(nodes), chunk_size):
            get_eval_node_representations(
                model,
                nodes[start : start + chunk_size],
                t_norm,
                store,
                args,
                device,
                num_rels,
                num_nodes,
                node_cache,
                profile=profile,
            )
    profile_add(profile, "eval_node_preload_time", profile_now(args, device) - t0)


def train_one_epoch(
    model,
    train_list,
    optimizer,
    dst_pool,
    args,
    device,
    num_rels,
    num_nodes,
    dataset_end_t,
    time_span,
    rng,
):
    model.train()
    store = make_recent_store(args, num_nodes)
    rel_history = RelationObjectHistory(num_rels)
    total_loss = 0.0
    total_count = 0
    profile = {
        "train_batches": 0.0,
        "train_events": 0.0,
        "train_candidate_scores": 0.0,
        "train_snapshots": 0.0,
    }
    sync_device(device)
    t0 = time.time()
    train_sampler = getattr(args, "train_sampler", "grouped_exact")
    uses_timestamp_positive_index = train_sampler in {"exact", "grouped_exact"}

    for events, t_norm, t_orig in tqdm(train_list, desc="train", leave=False):
        events = events.astype(np.int64, copy=False)
        profile_add(profile, "train_snapshots", 1)
        timestamp_positive_objects = None
        if uses_timestamp_positive_index:
            t_part = profile_now(args, device)
            timestamp_positive_objects = build_timestamp_positive_objects(events)
            profile_add(profile, "train_positive_index_time", profile_now(args, device) - t_part)
        candidate_cache = {}
        for start in range(0, len(events), args.batch_size):
            batch = events[start : start + args.batch_size]
            if len(batch) == 0:
                continue
            profile_add(profile, "train_batches", 1)
            profile_add(profile, "train_events", len(batch))
            t_part = profile_now(args, device)
            if train_sampler == "exact":
                neg_nodes = sample_train_negatives(
                    batch,
                    rel_history,
                    dst_pool,
                    args.train_num_neg,
                    args.hard_neg_ratio,
                    rng,
                    timestamp_positive_objects,
                    candidate_cache,
                )
            elif train_sampler == "grouped_exact":
                neg_nodes = sample_train_negatives_grouped_exact(
                    batch,
                    rel_history,
                    dst_pool,
                    args.train_num_neg,
                    args.hard_neg_ratio,
                    rng,
                    timestamp_positive_objects,
                    candidate_cache,
                    max_matrix_mb=getattr(args, "train_group_matrix_mb", 256.0),
                )
            else:
                neg_nodes = sample_train_negatives_fast(
                    batch,
                    rel_history,
                    dst_pool,
                    args.train_num_neg,
                    args.hard_neg_ratio,
                    rng,
                )
            profile_add(profile, "train_neg_sample_time", profile_now(args, device) - t_part)
            t_part = profile_now(args, device)
            candidates = np.concatenate((batch[:, 2:3], neg_nodes), axis=1)
            profile_add(profile, "train_candidate_build_time", profile_now(args, device) - t_part)
            profile_add(profile, "train_candidate_scores", candidates.size)
            t_part = profile_now(args, device)
            scores = score_candidate_nodes(
                model,
                batch,
                candidates,
                t_norm,
                store,
                args,
                device,
                num_rels,
                num_nodes,
            )
            profile_add(profile, "train_score_forward_time", profile_now(args, device) - t_part)
            t_part = profile_now(args, device)
            if args.train_loss == "margin":
                pos_scores = scores[:, 0]
                hardest_neg_scores = scores[:, 1:].max(dim=1).values
                target = torch.ones_like(pos_scores)
                loss_each = F.margin_ranking_loss(
                    pos_scores,
                    hardest_neg_scores,
                    target,
                    margin=args.rank_margin,
                    reduction="none",
                )
            else:
                labels = torch.zeros(len(batch), dtype=torch.long, device=device)
                loss_each = F.cross_entropy(
                    scores / args.temperature, labels, reduction="none"
                )
            if args.curriculum_decay > 0.0:
                age = max(0.0, float(dataset_end_t) - float(t_norm))
                if not args.curriculum_raw_age:
                    age = age / max(float(time_span), 1.0)
                weight = math.exp(-args.curriculum_decay * age)
                loss = (loss_each * weight).mean()
            else:
                loss = loss_each.mean()
            profile_add(profile, "train_loss_time", profile_now(args, device) - t_part)

            t_part = profile_now(args, device)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            profile_add(profile, "train_backward_step_time", profile_now(args, device) - t_part)

            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)

        t_part = profile_now(args, device)
        rel_history.update(events)
        store.update(events, t_norm, get_abs_time_value(args, t_norm, t_orig))
        profile_add(profile, "train_store_update_time", profile_now(args, device) - t_part)

    sync_device(device)
    train_time = time.time() - t0
    profile["train_total_time"] = train_time
    return total_loss / max(total_count, 1), train_time, store, profile


@torch.no_grad()
def predict_eval_batch(
    model,
    batch_data,
    neg_arr,
    current_t,
    store,
    args,
    device,
    num_rels,
    num_nodes,
    source_cache=None,
    node_cache=None,
    profile=None,
):
    batch_data = batch_data.astype(np.int64, copy=False)
    query_rel = torch.from_numpy(batch_data[:, 1].astype(np.int64)).to(device)
    profile_add(profile, "eval_batches", 1)
    profile_add(profile, "eval_events", len(batch_data))
    t_part = profile_now(args, device)
    source_encoded = get_eval_source_encoded(
        model,
        batch_data,
        current_t,
        store,
        args,
        device,
        num_rels,
        num_nodes,
        source_cache,
        profile=profile,
    )
    profile_add(profile, "eval_source_total_time", profile_now(args, device) - t_part)

    use_node_cache = node_cache is not None and can_cache_eval_nodes(model)
    if use_node_cache:
        profile_add(profile, "eval_node_cache_enabled_batches", 1)
    else:
        profile_add(profile, "eval_node_cache_disabled_batches", 1)
    pos_candidates = batch_data[:, 2:3]
    if use_node_cache:
        _, h_src, _ = source_encoded
        t_part = profile_now(args, device)
        pos_h_dst = get_eval_node_representations(
            model,
            pos_candidates.reshape(-1),
            current_t,
            store,
            args,
            device,
            num_rels,
            num_nodes,
            node_cache,
            profile=profile,
            query_rel=batch_data[:, 1],
        )
        profile_add(profile, "eval_pos_node_time", profile_now(args, device) - t_part)
        t_part = forward_tic(profile, args, device)
        pos_scores = model.score_representations(
            h_src,
            pos_h_dst,
            query_rel,
            1,
            source_nodes=batch_data[:, 0],
            candidate_nodes=pos_candidates,
        ).reshape(-1, 1)
        forward_toc(profile, args, device, t_part, "eval_pos_score_time")
        del pos_h_dst
    else:
        t_part = profile_now(args, device)
        pos_dst_inputs = histories_to_tensors(
            pos_candidates.reshape(-1),
            current_t,
            store,
            args,
            device,
            num_rels,
            num_nodes,
        )
        profile_add(profile, "eval_pos_history_time", profile_now(args, device) - t_part)
        t_part = forward_tic(profile, args, device)
        pos_scores = model.score_candidates_from_source(
            source_encoded,
            pos_dst_inputs,
            query_rel,
            1,
            source_nodes=batch_data[:, 0],
            candidate_nodes=pos_candidates,
        ).reshape(-1, 1)
        forward_toc(profile, args, device, t_part, "eval_pos_score_time")
        del pos_dst_inputs

    batch_size, max_negs = neg_arr.shape
    neg_scores = np.zeros((batch_size, max_negs), dtype=np.float32)
    start = 0
    chunk_size = int(args.eval_neg_chunk)
    while start < max_negs:
        end = min(start + chunk_size, max_negs)
        try:
            profile_add(profile, "eval_neg_chunks", 1)
            profile_add(profile, "eval_neg_candidates", batch_size * (end - start))
            t_part = profile_now(args, device)
            neg_chunk = neg_arr[:, start:end].astype(np.int64, copy=True)
            neg_chunk[neg_chunk < 0] = 0
            profile_add(profile, "eval_neg_prepare_time", profile_now(args, device) - t_part)
            if use_node_cache:
                t_part = profile_now(args, device)
                neg_h_dst = get_eval_node_representations(
                    model,
                    neg_chunk.reshape(-1),
                    current_t,
                    store,
                    args,
                    device,
                    num_rels,
                    num_nodes,
                    node_cache,
                    profile=profile,
                    query_rel=np.repeat(batch_data[:, 1], end - start),
                )
                profile_add(profile, "eval_neg_node_time", profile_now(args, device) - t_part)
                t_part = forward_tic(profile, args, device)
                scores = model.score_representations(
                    h_src,
                    neg_h_dst,
                    query_rel,
                    end - start,
                    source_nodes=batch_data[:, 0],
                    candidate_nodes=neg_chunk,
                )
                forward_toc(profile, args, device, t_part, "eval_neg_score_time")
                del neg_h_dst
            else:
                t_part = profile_now(args, device)
                neg_dst_inputs = histories_to_tensors(
                    neg_chunk.reshape(-1),
                    current_t,
                    store,
                    args,
                    device,
                    num_rels,
                    num_nodes,
                )
                profile_add(profile, "eval_neg_history_time", profile_now(args, device) - t_part)
                t_part = forward_tic(profile, args, device)
                scores = model.score_candidates_from_source(
                    source_encoded,
                    neg_dst_inputs,
                    query_rel,
                    end - start,
                    source_nodes=batch_data[:, 0],
                    candidate_nodes=neg_chunk,
                )
                forward_toc(profile, args, device, t_part, "eval_neg_score_time")
                del neg_dst_inputs
            t_part = profile_now(args, device)
            neg_scores[:, start:end] = scores.detach().cpu().numpy().astype(np.float32)
            profile_add(profile, "eval_neg_cpu_copy_time", profile_now(args, device) - t_part)
            del scores, neg_chunk
            start = end
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or chunk_size <= 1:
                raise
            profile_add(profile, "eval_oom_retries", 1)
            if "neg_dst_inputs" in locals():
                del neg_dst_inputs
            if "neg_h_dst" in locals():
                del neg_h_dst
            if "scores" in locals():
                del scores
            if "neg_chunk" in locals():
                del neg_chunk
            chunk_size = max(1, chunk_size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[TimeTKG] CUDA OOM during eval; retrying with eval_neg_chunk={chunk_size}",
                flush=True,
            )

    return pos_scores.detach().cpu().numpy().astype(np.float32), neg_scores


def validate_eval_neg_batch(args, mode, batch_data, neg_mask):
    q = int(getattr(args, "ns_q", -1))
    if q <= 0 or len(batch_data) == 0:
        return
    counts = np.sum(np.asarray(neg_mask, dtype=bool), axis=1)
    if np.all(counts == q):
        return
    bad_rows = int(np.sum(counts != q))
    raise ValueError(
        f"expected {q} negatives per {mode} event, but got "
        f"min={int(counts.min())} max={int(counts.max())} "
        f"for batch={len(batch_data)} bad_rows={bad_rows}; "
        "check that the precomputed negative cache matches the dataset, split, "
        "timestamps, ns_q, ns_seed, and train_predict_ratio"
    )


@torch.no_grad()
def evaluate_split(
    model,
    snapshot_list,
    store,
    data,
    args,
    device,
    num_rels,
    num_nodes,
    mode,
    out_dir=None,
    write_scores=False,
    max_events=None,
    measure_model_forward=False,
):
    model.eval()
    neg_sampler = get_negative_sampler(data)
    writer = ScoreWriter(out_dir, mode) if write_scores else None
    cache_root = (
        osp.join(out_dir, "source_cache")
        if out_dir is not None and getattr(args, "cache_eval_source", False)
        else None
    )
    source_cache = EvalSourceCache(cache_root, mode=mode, args=args)
    metric_sums = {}
    metric_count = 0
    profile = {
        "eval_batches": 0.0,
        "eval_events": 0.0,
        "eval_snapshots": 0.0,
    }
    if measure_model_forward:
        profile["_measure_model_forward"] = True
    t0 = profile_now(args, device)
    first_batch_logged = False
    max_events = None if max_events is None or int(max_events) <= 0 else int(max_events)
    stop_eval = False

    for events, t_norm, t_orig in tqdm(snapshot_list, desc=mode, leave=False):
        profile_add(profile, "eval_snapshots", 1)
        node_cache = (
            EvalNodeCache(
                num_nodes=num_nodes,
                device=device,
                dense=getattr(args, "dense_eval_node_cache", True),
                max_mb=getattr(args, "max_eval_node_cache_mb", 4096.0),
            )
            if can_cache_eval_nodes(model)
            else None
        )
        preload_snapshot_eval_nodes(
            model,
            events,
            t_norm,
            t_orig,
            mode,
            neg_sampler,
            store,
            args,
            device,
            num_rels,
            num_nodes,
            node_cache,
            profile=profile,
        )
        batch_iter = collect_eval_batch(events, t_orig, neg_sampler, mode, args.eval_batch_size)
        while True:
            t_part = profile_now(args, device)
            try:
                batch_data, neg_arr, neg_mask = next(batch_iter)
            except StopIteration:
                profile_add(profile, "eval_collect_batch_time", profile_now(args, device) - t_part)
                break
            profile_add(profile, "eval_collect_batch_time", profile_now(args, device) - t_part)
            validate_eval_neg_batch(args, mode, batch_data, neg_mask)
            if not first_batch_logged:
                print(
                    f"[TimeTKG] first {mode} batch: t_norm={int(t_norm)} "
                    f"t_orig={int(t_orig)} batch={batch_data.shape} "
                    f"neg_arr={neg_arr.shape} valid_negs={int(neg_mask.sum())}",
                    flush=True,
                )
                first_batch_logged = True
            pos, neg = predict_eval_batch(
                model,
                batch_data,
                neg_arr,
                t_norm,
                store,
                args,
                device,
                num_rels,
                num_nodes,
                source_cache=source_cache,
                node_cache=node_cache,
                profile=profile,
            )
            t_part = profile_now(args, device)
            batch_metrics = compute_ranking_metric_sums(pos, neg, neg_mask)
            profile_add(profile, "eval_metric_time", profile_now(args, device) - t_part)
            add_metric_sums(metric_sums, batch_metrics)
            metric_count += int(batch_metrics["count"])
            if writer is not None:
                t_part = profile_now(args, device)
                writer.write_batch(pos, neg, neg_mask)
                profile_add(profile, "eval_writer_time", profile_now(args, device) - t_part)
            if max_events is not None and metric_count >= max_events:
                stop_eval = True
                break

        if stop_eval:
            break

        t_part = profile_now(args, device)
        store.update(
            events.astype(np.int64, copy=False),
            t_norm,
            get_abs_time_value(args, t_norm, t_orig),
        )
        profile_add(profile, "eval_store_update_time", profile_now(args, device) - t_part)
        source_cache.values.clear()
        if node_cache is not None:
            node_cache.clear()

    if writer is not None:
        writer.close()
    source_cache.close()
    metrics = add_metric_aliases(finalize_metric_sums(metric_sums))
    profile["eval_total_time"] = profile_now(args, device) - t0
    profile["eval_metric_count"] = float(metric_count)
    profile.pop("_measure_model_forward", None)
    metrics["profile"] = profile
    print(
        f"[TimeTKG] {mode}_mrr_strict={metrics['mrr_strict']:.5f} "
        f"{mode}_hit10_strict={metrics['hit10']:.5f} "
        f"{mode}_mrr_loose={metrics['mrr_loose']:.5f} "
        f"time={profile['eval_total_time']:.1f}s",
        flush=True,
    )
    return metrics


def warmup_train_history(train_list, args, num_nodes):
    store = make_recent_store(args, num_nodes)
    for events, t_norm, t_orig in tqdm(train_list, desc="warmup", leave=False):
        store.update(
            events.astype(np.int64, copy=False),
            t_norm,
            get_abs_time_value(args, t_norm, t_orig),
        )
    return store


def build_model(args, num_nodes, num_rels, t_min, t_max, device):
    abs_periods = parse_float_list(getattr(args, "abs_time_periods", "1,7,30"))
    multi_windows = parse_int_list(getattr(args, "multi_windows", ""))
    model = TKGTimeMixer(
        num_nodes=num_nodes,
        num_rels=num_rels,
        topk=args.topk,
        time_dim=args.time_dim,
        rel_dim=args.rel_dim,
        node_dim=args.node_dim,
        event_dim=args.event_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        t_min=args.time_min,
        t_max=max(t_max - t_min, args.time_min),
        use_neighbor_id=args.use_neighbor_id,
        token_expansion_factor=args.token_expansion_factor,
        channel_expansion_factor=args.channel_expansion_factor,
        use_single_layer=args.use_single_layer,
        predictor_mode=args.predictor_mode,
        use_abs_time=getattr(args, "use_abs_time", False),
        abs_time_periods=abs_periods,
        abs_time_harmonics=getattr(args, "abs_time_harmonics", 1),
        use_query_gate=getattr(args, "use_query_gate", False),
        query_gate_type=getattr(args, "query_gate_type", "channel"),
        use_rank_pos=getattr(args, "use_rank_pos", False),
        multi_windows=multi_windows,
        encoder_backend=getattr(args, "event_encoder", "mixer"),
        transformer_heads=getattr(args, "transformer_heads", 2),
        transformer_ff_dim=getattr(args, "transformer_ff_dim", None),
        use_cross_history=getattr(args, "use_cross_history", False),
        cross_heads=getattr(args, "cross_heads", 2),
    ).to(device)
    return model


def score_modes(args):
    modes = ("train", "val") if float(args.train_predict_ratio) > 0.0 else ("val",)
    if getattr(args, "eval_test", True):
        modes = modes + ("test",)
    return modes


def split_train_for_prediction(data):
    train_list = data["train_list"]
    start = int(data.get("train_predict_start_idx", len(train_list)))
    start = min(max(start, 0), len(train_list))
    return train_list[:start], train_list[start:]


def snapshot_time_bounds(snapshot_list):
    if not snapshot_list:
        return 0.0, 1.0
    values = [float(t_norm) for _, t_norm, _ in snapshot_list]
    return min(values), max(values)


def count_events(snapshot_list):
    return sum(len(events) for events, _, _ in snapshot_list)


def select_quick_val_suffix(val_list, quick_val_events):
    limit = int(quick_val_events)
    total = count_events(val_list)
    if limit <= 0 or total <= limit:
        return [], val_list, {
            "enabled": False,
            "total_events": int(total),
            "eval_events": int(total),
            "eval_timestamps": int(len(val_list)),
        }

    start = len(val_list)
    selected = 0
    while start > 0 and selected <= limit:
        start -= 1
        selected += len(val_list[start][0])

    return val_list[:start], val_list[start:], {
        "enabled": True,
        "total_events": int(total),
        "limit": int(limit),
        "warmup_timestamps": int(start),
        "eval_timestamps": int(len(val_list) - start),
        "eval_events": int(selected),
    }


def update_store_history(store, snapshot_list, args):
    for events, t_norm, t_orig in snapshot_list:
        store.update(
            events.astype(np.int64, copy=False),
            t_norm,
            get_abs_time_value(args, t_norm, t_orig),
        )
    return store


def make_optimizer(model, args):
    return torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )


def train_model_phase(
    model,
    train_list,
    data,
    optimizer,
    dst_pool,
    args,
    device,
    num_rels,
    num_nodes,
    phase,
    num_epochs,
    select_with_val,
    best_path=None,
):
    if not train_list:
        raise ValueError(f"{phase} training list is empty")

    train_t_min, train_t_max = snapshot_time_bounds(train_list)
    train_time_span = max(train_t_max - train_t_min, 1.0)
    selection_metric = getattr(args, "selection_metric", "mrr")
    best_val_score = -float("inf")
    best_val_metrics = {}
    best_epoch = 0
    bad_rounds = 0
    total_train_time = 0.0
    train_peak_alloc = 0.0
    train_peak_reserved = 0.0
    rng = np.random.RandomState(args.seed)
    val_warmup_list, val_eval_list, quick_val_info = select_quick_val_suffix(
        data["val_list"],
        getattr(args, "quick_val_events", 0),
    )
    if select_with_val and quick_val_info["enabled"]:
        print(
            f"[TimeTKG][{phase}] quick val uses last {quick_val_info['eval_timestamps']} "
            f"timestamps/{quick_val_info['eval_events']} events "
            f"(limit={quick_val_info['limit']}, full_val_events={quick_val_info['total_events']}); "
            f"warmup_val_timestamps={quick_val_info['warmup_timestamps']}",
            flush=True,
        )

    for epoch in range(1, int(num_epochs) + 1):
        reset_cuda_peak(device)
        loss, train_time, train_store, train_profile = train_one_epoch(
            model,
            train_list,
            optimizer,
            dst_pool,
            args,
            device,
            num_rels,
            num_nodes,
            dataset_end_t=train_t_max,
            time_span=train_time_span,
            rng=rng,
        )
        epoch_alloc, epoch_reserved = cuda_peak_mb(device)
        train_peak_alloc = max(train_peak_alloc, epoch_alloc)
        train_peak_reserved = max(train_peak_reserved, epoch_reserved)
        total_train_time += float(train_time)

        if not select_with_val:
            best_epoch = epoch
            print(
                f"[TimeTKG][{phase}] epoch={epoch} loss={loss:.5f} "
                f"train_time={train_time:.1f}s",
                flush=True,
            )
            continue

        if val_warmup_list:
            update_store_history(train_store, val_warmup_list, args)
        val_metrics = evaluate_split(
            model,
            val_eval_list,
            train_store,
            data,
            args,
            device,
            num_rels,
            num_nodes,
            mode="val",
            write_scores=False,
        )
        val_profile = val_metrics.get("profile", {})
        val_mrr = val_metrics["mrr"]
        val_hit10 = val_metrics["hit10"]
        val_score = val_metrics[selection_metric]
        print(
            f"[TimeTKG][{phase}] epoch={epoch} loss={loss:.5f} "
            f"val_mrr={val_mrr:.5f} val_hit10={val_hit10:.5f} "
            f"select_{selection_metric}={val_score:.5f} "
            f"train_time={train_time:.1f}s",
            flush=True,
        )
        print_epoch_profile(
            epoch,
            args,
            model,
            train_profile,
            val_profile,
            device,
            peak_alloc=epoch_alloc,
            peak_reserved=epoch_reserved,
        )

        if val_score > best_val_score + args.tolerance:
            best_val_score = val_score
            best_val_metrics = {"mrr": float(val_mrr), "hit10": float(val_hit10)}
            best_epoch = epoch
            bad_rounds = 0
            if best_path is not None:
                torch.save(model.state_dict(), best_path)
            print(
                f"[TimeTKG][{phase}] saved best model at epoch {epoch} "
                f"by val_{selection_metric}={val_score:.5f}",
                flush=True,
            )
        else:
            bad_rounds += 1
            if bad_rounds >= args.patience:
                print(
                    f"[TimeTKG][{phase}] early stop at epoch {epoch}; "
                    f"best_epoch={best_epoch} "
                    f"best_val_{selection_metric}={best_val_score:.5f}",
                    flush=True,
                )
                break

    if select_with_val and best_path is not None and osp.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    return {
        "model": model,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score if select_with_val else 0.0),
        "best_val_metrics": best_val_metrics,
        "train_time_sec": float(total_train_time),
        "train_peak_alloc_mb": float(train_peak_alloc),
        "train_peak_reserved_mb": float(train_peak_reserved),
    }


def get_run_params(args):
    return {
        "data": "utils_v4_oof_qvsuffix_speed1",
        "metric": "strict_v2",
        "bs": args.batch_size,
        "ebs": args.eval_batch_size,
        "topk": args.topk,
        "neg": args.train_num_neg,
        "hard": args.hard_neg_ratio,
        "sampler": getattr(args, "train_sampler", "grouped_exact"),
        "tgmb": getattr(args, "train_group_matrix_mb", 256.0),
        "lr": args.lr,
        "wd": args.weight_decay,
        "td": args.time_dim,
        "rd": args.rel_dim,
        "hnd": args.node_dim if args.use_neighbor_id else 0,
        "snd": args.node_dim,
        "ed": args.event_dim,
        "hd": args.hidden_dim,
        "nl": args.num_layers,
        "do": args.dropout,
        "tmin": args.time_min,
        "tef": args.token_expansion_factor,
        "cef": args.channel_expansion_factor,
        "sl": int(args.use_single_layer),
        "pred": args.predictor_mode,
        "enc": getattr(args, "event_encoder", "mixer"),
        "th": getattr(args, "transformer_heads", 2),
        "tff": getattr(args, "transformer_ff_dim", None) or 0,
        "abs": int(getattr(args, "use_abs_time", False)),
        "absp": (
            str(getattr(args, "abs_time_periods", "")).replace(",", "-")
            if getattr(args, "use_abs_time", False)
            else "off"
        ),
        "absh": (
            getattr(args, "abs_time_harmonics", 1)
            if getattr(args, "use_abs_time", False)
            else 0
        ),
        "absraw": (
            int(getattr(args, "abs_time_use_raw", False))
            if getattr(args, "use_abs_time", False)
            else 0
        ),
        "gate": (
            getattr(args, "query_gate_type", "off")
            if getattr(args, "use_query_gate", False)
            else "off"
        ),
        "rank": int(getattr(args, "use_rank_pos", False)),
        "mw": str(getattr(args, "multi_windows", "")).replace(",", "-") or "off",
        "cross": int(getattr(args, "use_cross_history", False)),
        "xh": (
            getattr(args, "cross_heads", 2)
            if getattr(args, "use_cross_history", False)
            else 0
        ),
        "temp": args.temperature,
        "loss": args.train_loss,
        "margin": args.rank_margin if args.train_loss == "margin" else 0,
        "gc": args.grad_clip,
        "cd": args.curriculum_decay,
        "craw": int(args.curriculum_raw_age),
        "ep": args.num_epochs,
        "pat": args.patience,
        "tol": args.tolerance,
        "sel": getattr(args, "selection_metric", "mrr"),
        "qv": getattr(args, "quick_val_events", 0),
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "tpr": args.train_predict_ratio,
    }


def get_out_dir(args):
    params = get_run_params(args)
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "params": params,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    run_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    mw = str(params["mw"]).replace("/", "-")
    gate = str(params["gate"]).replace("/", "-")
    name = (
        f"r{run_hash}"
        f"_topk{params['topk']}"
        f"_mw{mw}"
        f"_ed{params['ed']}"
        f"_hd{params['hd']}"
        f"_bs{params['bs']}"
        f"_ebs{params['ebs']}"
        f"_neg{params['neg']}"
        f"_sam{str(params['sampler'])[:5]}"
        f"_nsq{params['ns_q']}"
        f"_nss{params['ns_seed']}"
        f"_tpr{params['tpr']:g}"
        f"_abs{params['abs']}r{params['absraw']}"
        f"_gate{gate}"
        f"_rank{params['rank']}"
        f"_loss{params['loss']}"
    )
    if len(name) > 180:
        name = (
            f"r{run_hash}_topk{params['topk']}_ed{params['ed']}"
            f"_bs{params['bs']}_ebs{params['ebs']}"
        )
    return osp.join("results_time_tkg_single", args.dataset, f"seed{args.seed}", name)


def profile_get(profile, key):
    if profile is None:
        return 0.0
    return float(profile.get(key, 0.0))


def profile_hit_rate(profile, hits_key, queries_key):
    queries = profile_get(profile, queries_key)
    if queries <= 0.0:
        return 0.0
    return profile_get(profile, hits_key) / queries


def print_epoch_profile(epoch, args, model, train_profile, val_profile, device, peak_alloc=None, peak_reserved=None):
    if peak_alloc is None or peak_reserved is None:
        peak_alloc, peak_reserved = cuda_peak_mb(device)
    train_total = profile_get(train_profile, "train_total_time")
    val_total = profile_get(val_profile, "eval_total_time")
    print(
        f"[TimeTKG][epoch {epoch} profile] total "
        f"train={format_seconds(train_total)} val={format_seconds(val_total)} "
        f"peak_alloc={peak_alloc:.0f}MB peak_reserved={peak_reserved:.0f}MB",
        flush=True,
    )
    print(
        f"[TimeTKG][epoch {epoch} train] "
        f"neg={format_seconds(profile_get(train_profile, 'train_neg_sample_time'))} "
        f"score={format_seconds(profile_get(train_profile, 'train_score_forward_time'))} "
        f"loss={format_seconds(profile_get(train_profile, 'train_loss_time'))} "
        f"backward={format_seconds(profile_get(train_profile, 'train_backward_step_time'))} "
        f"update={format_seconds(profile_get(train_profile, 'train_store_update_time'))} "
        f"batches={int(profile_get(train_profile, 'train_batches'))} "
        f"events={int(profile_get(train_profile, 'train_events'))} "
        f"cands={int(profile_get(train_profile, 'train_candidate_scores'))}",
        flush=True,
    )
    print(
        f"[TimeTKG][epoch {epoch} eval] "
        f"source={format_seconds(profile_get(val_profile, 'eval_source_total_time'))} "
        f"node={format_seconds(profile_get(val_profile, 'eval_node_encode_time'))} "
        f"pos={format_seconds(profile_get(val_profile, 'eval_pos_score_time'))} "
        f"neg_node={format_seconds(profile_get(val_profile, 'eval_neg_node_time') + profile_get(val_profile, 'eval_neg_history_time'))} "
        f"neg_score={format_seconds(profile_get(val_profile, 'eval_neg_score_time'))} "
        f"metric={format_seconds(profile_get(val_profile, 'eval_metric_time'))} "
        f"store={format_seconds(profile_get(val_profile, 'eval_store_update_time'))} "
        f"preload={format_seconds(profile_get(val_profile, 'eval_node_preload_time'))} "
        f"batches={int(profile_get(val_profile, 'eval_batches'))} "
        f"events={int(profile_get(val_profile, 'eval_metric_count'))} "
        f"neg_cands={int(profile_get(val_profile, 'eval_neg_candidates'))}",
        flush=True,
    )
    print(
        f"[TimeTKG][epoch {epoch} cache] "
        f"source_hit={profile_hit_rate(val_profile, 'source_cache_hits', 'source_cache_queries'):.3f} "
        f"node_hit={profile_hit_rate(val_profile, 'node_cache_hits', 'node_cache_queries'):.3f} "
        f"node_unique={int(profile_get(val_profile, 'node_cache_unique_encodes'))} "
        f"preload_nodes={int(profile_get(val_profile, 'eval_node_preload_nodes'))} "
        f"neg_chunks={int(profile_get(val_profile, 'eval_neg_chunks'))} "
        f"oom_retries={int(profile_get(val_profile, 'eval_oom_retries'))} "
        f"cache_nodes={int(can_cache_eval_nodes(model))}",
        flush=True,
    )
    print(
        f"[TimeTKG][epoch {epoch} config] "
        f"dataset={args.dataset} q={args.ns_q} topk={args.topk} "
        f"bs={args.batch_size} ebs={args.eval_batch_size} neg_chunk={args.eval_neg_chunk} "
        f"sampler={getattr(args, 'train_sampler', 'fast')} "
        f"tgmb={getattr(args, 'train_group_matrix_mb', 256.0):g} "
        f"preload={int(getattr(args, 'preload_eval_nodes', True))} "
        f"dense={int(getattr(args, 'dense_eval_node_cache', True))} "
        f"event_dim={args.event_dim} hidden_dim={args.hidden_dim} time_dim={args.time_dim} "
        f"abs={int(getattr(args, 'use_abs_time', False))} "
        f"rank={int(getattr(args, 'use_rank_pos', False))} "
        f"gate={int(getattr(args, 'use_query_gate', False))} "
        f"cross={int(getattr(args, 'use_cross_history', False))} "
        f"neighbor={int(getattr(args, 'use_neighbor_id', False))} "
        f"mw={getattr(args, 'multi_windows', '') or 'off'} "
        f"encoder={getattr(args, 'event_encoder', 'mixer')} "
        f"profile_sync={int(getattr(args, 'profile_sync', False))}",
        flush=True,
    )


def validate_args(args):
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if not 0.0 <= float(args.train_predict_ratio) < 1.0:
        raise ValueError("--train_predict_ratio must be in [0, 1)")
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.train_num_neg <= 0:
        raise ValueError("--train_num_neg must be positive")
    if args.eval_neg_chunk <= 0:
        raise ValueError("--eval_neg_chunk must be positive")
    if args.eval_node_preload_chunk <= 0:
        raise ValueError("--eval_node_preload_chunk must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if args.max_eval_node_cache_mb <= 0:
        raise ValueError("--max_eval_node_cache_mb must be positive")
    if args.train_group_matrix_mb <= 0:
        raise ValueError("--train_group_matrix_mb must be positive")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.time_dim % 2 != 0:
        raise ValueError("--time_dim must be even")
    if args.node_dim <= 0:
        raise ValueError("--node_dim must be positive")
    if args.abs_time_harmonics <= 0:
        raise ValueError("--abs_time_harmonics must be positive")
    if args.event_encoder == "transformer" and args.event_dim % args.transformer_heads != 0:
        raise ValueError("--event_dim must be divisible by --transformer_heads")
    if args.use_cross_history and args.event_dim % args.cross_heads != 0:
        raise ValueError("--event_dim must be divisible by --cross_heads")
    multi_windows = parse_int_list(args.multi_windows)
    if any(w <= 0 for w in multi_windows):
        raise ValueError("--multi_windows must contain positive integers")
    if args.train_loss == "margin" and args.rank_margin <= 0.0:
        raise ValueError("--rank_margin must be positive when --train_loss=margin")
    if args.num_epochs <= 0:
        raise ValueError("--num_epochs must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    args.hard_neg_ratio = min(1.0, max(0.0, args.hard_neg_ratio))


def main(args):
    validate_args(args)
    set_random_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    required_modes = score_modes(args)
    needs_data_for_out_dir = bool(args.use_abs_time and not parse_float_list(args.abs_time_periods))
    checked_complete = False
    if not needs_data_for_out_dir:
        out_dir = get_out_dir(args)
        checked_complete = True
        if is_run_complete(out_dir, modes=required_modes) and not args.force:
            print(f"[TimeTKG] already complete: {out_dir}", flush=True)
            metrics = load_metrics(out_dir)
            return metrics.get(
                f"test_{args.selection_metric}",
                metrics.get(
                    "test_mrr",
                    metrics.get(
                        "test_hit10",
                        metrics.get(f"val_{args.selection_metric}", metrics.get("val_mrr")),
                    ),
                ),
            )

    data = load_time_tkg_data(args)
    describe_loaded_data(data, prefix="[TimeTKG]")

    num_nodes, num_rels = infer_id_sizes(data, split_names=("train_list",))
    t_min, t_max = infer_time_bounds(data, split_names=("train_list",))
    raw_t_min, raw_t_max = infer_time_bounds(data, split_names=("train_list",), use_raw=True)
    train_t_min, train_t_max = t_min, t_max
    resolve_abs_time_periods(args, (t_min, t_max), (raw_t_min, raw_t_max))
    out_dir = get_out_dir(args)
    if not checked_complete and is_run_complete(out_dir, modes=required_modes) and not args.force:
        print(f"[TimeTKG] already complete: {out_dir}", flush=True)
        metrics = load_metrics(out_dir)
        return metrics.get(
            f"test_{args.selection_metric}",
            metrics.get(
                "test_mrr",
                metrics.get(
                    "test_hit10",
                    metrics.get(f"val_{args.selection_metric}", metrics.get("val_mrr")),
                ),
            ),
        )
    train_prefix, train_predict = split_train_for_prediction(data)
    if float(args.train_predict_ratio) > 0.0:
        if not train_predict:
            raise ValueError("--train_predict_ratio selected no training timestamps")
        if not train_prefix:
            raise ValueError("--train_predict_ratio leaves no prefix timestamps for OOF training")

    print(f"[TimeTKG] running -> {out_dir}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"[TimeTKG] ids: nodes={num_nodes} rels={num_rels}; "
        f"time_norm_span=[{t_min:g}, {t_max:g}] "
        f"time_raw_span=[{raw_t_min:g}, {raw_t_max:g}] "
        f"train_span=[{train_t_min:g}, {train_t_max:g}] "
        f"train_predict_ratio={float(args.train_predict_ratio):g} "
        f"train_predict_ts={len(train_predict)}",
        flush=True,
    )

    model = build_model(args, num_nodes, num_rels, t_min, t_max, device)
    print(model, flush=True)
    print(
        f"[TimeTKG] trainable parameters: {count_parameters(model) / 1_000_000:.3f}M",
        flush=True,
    )

    dst_pool = get_destination_pool(data, num_nodes)
    selection_metric = getattr(args, "selection_metric", "mrr")
    best_path = osp.join(out_dir, "best_model.pt")
    full_result = train_model_phase(
        model,
        data["train_list"],
        data,
        make_optimizer(model, args),
        dst_pool,
        args,
        device,
        num_rels,
        num_nodes,
        phase="full",
        num_epochs=args.num_epochs,
        select_with_val=True,
        best_path=best_path,
    )
    model = full_result["model"]
    best_epoch = max(1, int(full_result["best_epoch"]))
    best_val_score = float(full_result["best_val_score"])
    best_val_metrics = full_result["best_val_metrics"]

    train_metrics = None
    oof_result = None
    if train_predict:
        print(
            f"[TimeTKG] training OOF prefix model for {best_epoch} epochs "
            f"on {len(train_prefix)} prefix timestamps",
            flush=True,
        )
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_random_seed(args.seed)
        oof_model = build_model(args, num_nodes, num_rels, t_min, t_max, device)
        oof_result = train_model_phase(
            oof_model,
            train_prefix,
            data,
            make_optimizer(oof_model, args),
            dst_pool,
            args,
            device,
            num_rels,
            num_nodes,
            phase="oof",
            num_epochs=best_epoch,
            select_with_val=False,
        )
        oof_model_path = osp.join(out_dir, "oof_model.pt")
        torch.save(oof_result["model"].state_dict(), oof_model_path)
        oof_store = warmup_train_history(train_prefix, args, num_nodes)
        oof_model = oof_result["model"]
        train_metrics = evaluate_split(
            oof_model,
            train_predict,
            oof_store,
            data,
            args,
            device,
            num_rels,
            num_nodes,
            mode="train",
            out_dir=out_dir,
            write_scores=True,
        )
        oof_result["model"] = None
        del oof_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(device)

    final_store = warmup_train_history(data["train_list"], args, num_nodes)
    final_val_metrics = evaluate_split(
        model,
        data["val_list"],
        final_store,
        data,
        args,
        device,
        num_rels,
        num_nodes,
        mode="val",
        out_dir=out_dir,
        write_scores=True,
    )
    test_metrics = None
    infer_peak_alloc = 0.0
    infer_peak_reserved = 0.0
    infer_time = 0.0
    if getattr(args, "eval_test", True):
        reset_cuda_peak(device)
        test_metrics = evaluate_split(
            model,
            data["test_list"],
            final_store,
            data,
            args,
            device,
            num_rels,
            num_nodes,
            mode="test",
            out_dir=out_dir,
            write_scores=True,
            measure_model_forward=True,
        )
        infer_peak_alloc, infer_peak_reserved = cuda_peak_mb(device)
        infer_time = float(test_metrics.get("profile", {}).get("eval_model_forward_time", 0.0))
    train_time = float(full_result["train_time_sec"])
    train_peak_alloc = float(full_result["train_peak_alloc_mb"])
    train_peak_reserved = float(full_result["train_peak_reserved_mb"])
    if oof_result is not None:
        train_time += float(oof_result["train_time_sec"])
        train_peak_alloc = max(train_peak_alloc, float(oof_result["train_peak_alloc_mb"]))
        train_peak_reserved = max(train_peak_reserved, float(oof_result["train_peak_reserved_mb"]))

    msg = (
        f"[TimeTKG] final val_mrr={final_val_metrics['mrr']:.5f} "
        f"val_hit10={final_val_metrics['hit10']:.5f} "
    )
    if test_metrics is not None:
        msg += (
            f"test_mrr={test_metrics['mrr']:.5f} "
            f"test_hit10={test_metrics['hit10']:.5f} "
        )
    msg += f"best_epoch={best_epoch}"
    print(msg, flush=True)
    print(
        f"[TimeTKG] profile train_time={train_time:.1f}s "
        f"train_peak_alloc={train_peak_alloc:.0f}MB "
        f"train_peak_reserved={train_peak_reserved:.0f}MB "
        f"infer_model_forward_time={infer_time:.1f}s "
        f"infer_peak_alloc={infer_peak_alloc:.0f}MB "
        f"infer_peak_reserved={infer_peak_reserved:.0f}MB",
        flush=True,
    )

    config = vars(args).copy()
    config["out_dir"] = out_dir
    config["run_params"] = get_run_params(args)
    save_config(out_dir, config)
    metrics = {
        "format": "time_tkg_scores_v3",
        "score_protocol": "train_oof_prefix_valtest_full_train",
        "val_mrr": float(final_val_metrics["mrr"]),
        "val_hit10": float(final_val_metrics["hit10"]),
        "best_val_selection": float(best_val_score),
        "best_val_mrr": float(best_val_metrics.get("mrr", 0.0)),
        "best_val_hit10": float(best_val_metrics.get("hit10", 0.0)),
        "best_epoch": int(best_epoch),
        "best_model_path": best_path,
        "selection_metric": selection_metric,
        "train_predict_ratio": float(args.train_predict_ratio),
        "train_predict_timestamps": int(len(train_predict)),
        "train_time_sec": float(train_time),
        "train_peak_alloc_mb": float(train_peak_alloc),
        "train_peak_reserved_mb": float(train_peak_reserved),
        "infer_time_sec": float(infer_time),
        "infer_peak_alloc_mb": float(infer_peak_alloc),
        "infer_peak_reserved_mb": float(infer_peak_reserved),
        "full_train_time_sec": float(full_result["train_time_sec"]),
        "full_train_peak_alloc_mb": float(full_result["train_peak_alloc_mb"]),
        "full_train_peak_reserved_mb": float(full_result["train_peak_reserved_mb"]),
    }
    if oof_result is not None:
        metrics.update(
            {
                "oof_train_time_sec": float(oof_result["train_time_sec"]),
                "oof_train_peak_alloc_mb": float(oof_result["train_peak_alloc_mb"]),
                "oof_train_peak_reserved_mb": float(oof_result["train_peak_reserved_mb"]),
                "oof_model_path": osp.join(out_dir, "oof_model.pt"),
            }
        )
    if train_metrics is not None:
        metrics["train_mrr"] = float(train_metrics["mrr"])
        metrics["train_hit10"] = float(train_metrics["hit10"])
        metrics.update(prefix_metrics("train", train_metrics))
    metrics.update(prefix_metrics("val", final_val_metrics))
    if test_metrics is not None:
        metrics["test_mrr"] = float(test_metrics["mrr"])
        metrics["test_hit10"] = float(test_metrics["hit10"])
        metrics.update(prefix_metrics("test", test_metrics))
    save_metrics(out_dir, metrics)
    if test_metrics is not None:
        return test_metrics[selection_metric]
    return final_val_metrics[selection_metric]

