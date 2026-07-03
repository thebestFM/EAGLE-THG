import heapq
import os.path as osp
import sys
import math
import time
from collections import defaultdict

import numpy as np
import torch

from numba import njit, prange, get_num_threads, set_num_threads
NUMBA_AVAILABLE = True

from utils import (
    SUPPORTED_DATASETS,
    TGB_DATASETS,
    ScoreWriter,
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    describe_loaded_data,
    finalize_metric_sums,
    inverse_aug,
    is_run_complete,
    load_datasets,
    load_metrics,
    make_dir_name,
    save_config,
    save_metrics,
    set_random_seed,
)


def _cuda_device(device):
    if not torch.cuda.is_available():
        return None
    dev = torch.device(device)
    if dev.type != 'cuda':
        return None
    if dev.index is None:
        return torch.device('cuda', torch.cuda.current_device())
    return dev


def _cuda_synchronize(device):
    dev = _cuda_device(device)
    if dev is not None:
        torch.cuda.synchronize(dev)


def _reset_cuda_peak(device):
    dev = _cuda_device(device)
    if dev is None:
        return False
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    return True


def _cuda_peak_allocated(device):
    dev = _cuda_device(device)
    if dev is None:
        return None
    torch.cuda.synchronize(dev)
    return int(torch.cuda.max_memory_allocated(dev))


def _format_bytes(num_bytes):
    if num_bytes is None:
        return 'n/a'
    return f'{num_bytes / (1024 ** 2):.1f}MB'


class LogicMatrixUpdater:
    """Logic Matrix Updater for M_trans"""
    def __init__(self, num_nodes, num_rels, window_size=10.0, decay_factor=0.1, device='cuda:0'):
        self.num_rels = num_rels
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.device = device

        self.M_trans_counts = torch.zeros((num_rels, num_rels), dtype=torch.float32, device=device)

        self.history_r = [[] for _ in range(num_nodes)]
        self.history_t = [[] for _ in range(num_nodes)]

    def update_M_trans_step(self, events, current_t):
        """
        Update M_trans counts based on new events-match-history
        events: (B, 3) numpy array
        """
        num_events = len(events)
        threshold = current_t - self.window_size

        for i in range(num_events):
            r = int(events[i, 1])
            o = int(events[i, 2])

            hist_r = self.history_r[o]
            hist_t = self.history_t[o]

            if len(hist_t) > 0 and hist_t[0] <= threshold:
                k = 0
                while k < len(hist_t) and hist_t[k] <= threshold:
                    k += 1

                del hist_t[:k]
                del hist_r[:k]

            n_hist = len(hist_r)
            if n_hist > 0:
                r_old_tensor = torch.tensor(hist_r, dtype=torch.long, device=self.device)
                t_prev_tensor = torch.tensor(hist_t, dtype=torch.float32, device=self.device)

                delta = current_t - t_prev_tensor
                weights = torch.exp(-self.decay_factor * delta)

                self.M_trans_counts[r].scatter_add_(0, r_old_tensor, weights)

    def update_history_step(self, events, current_t):
        """
        Add current events to node history
        events: (B, 3) numpy array
        """
        num_events = len(events)
        for i in range(num_events):
            s = int(events[i, 0])
            r = int(events[i, 1])

            self.history_r[s].append(r)
            self.history_t[s].append(current_t)

    def get_normalized_M_trans(self):
        """
        Return M_trans scaled to range [0.2, 1.0] per row.
        """
        target_min = 0.2
        target_max = 1.0

        M_trans_out = torch.empty_like(self.M_trans_counts)
        target_range = target_max - target_min

        r_min = self.M_trans_counts.min(dim=1, keepdim=True)[0]
        r_max = self.M_trans_counts.max(dim=1, keepdim=True)[0]

        diff = r_max - r_min
        mask_zero_diff = (diff < 1e-9)

        diff = torch.where(mask_zero_diff, torch.ones_like(diff), diff)

        norm_01 = (self.M_trans_counts - r_min) / diff

        M_trans_out = target_min + norm_01 * target_range

        M_trans_out = torch.where(mask_zero_diff.expand_as(M_trans_out),
                                   torch.full_like(M_trans_out, target_min),
                                   M_trans_out)

        return M_trans_out


class SemanticMatrixUpdater:
    """Semantic Matrix Updater for M_sim"""
    def __init__(self, num_nodes, num_rels, window_size, device='cuda:0'):
        self.num_rels = num_rels
        self.window_size = window_size
        self.device = device

        self.node_shift = 10 ** int(math.ceil(math.log10(num_nodes + 1)))

        self.M_sim_counts = torch.zeros((num_rels, num_rels), dtype=torch.float32, device=device)

        self.pair_history_t = {}
        self.pair_history_r = {}

    def _get_key(self, s, o):
        return s * self.node_shift + o

    def update_M_sim_step(self, events, current_t):
        num_events = len(events)
        threshold = current_t - self.window_size

        for i in range(num_events):
            s = int(events[i, 0])
            r_new = int(events[i, 1])
            o = int(events[i, 2])

            key = self._get_key(s, o)

            if key not in self.pair_history_t:
                self.pair_history_t[key] = []
                self.pair_history_r[key] = []

            hist_t = self.pair_history_t[key]
            hist_r = self.pair_history_r[key]

            if len(hist_t) > 0 and hist_t[0] <= threshold:
                k = 0
                while k < len(hist_t) and hist_t[k] <= threshold:
                    k += 1

                del hist_t[:k]
                del hist_r[:k]

            n_hist = len(hist_r)
            if n_hist > 0:
                r_old_tensor = torch.tensor(hist_r, dtype=torch.long, device=self.device)
                ones_tensor = torch.ones(n_hist, dtype=torch.float32, device=self.device)

                self.M_sim_counts[:, r_new].scatter_add_(0, r_old_tensor, ones_tensor)
                self.M_sim_counts[r_new, :].scatter_add_(0, r_old_tensor, ones_tensor)

            hist_t.append(current_t)
            hist_r.append(r_new)

    def get_probability_M_sim(self):
        """
        PPMI + Sigmoid -> Mapped to [0.5, 1.0]
        """
        target_min = 0.5
        target_max = 1.0

        epsilon = 1e-8
        alpha = 1.0

        total_count = self.M_sim_counts.sum() + (alpha * self.num_rels * self.num_rels)
        row_sums = self.M_sim_counts.sum(dim=1) + (alpha * self.num_rels)
        col_sums = self.M_sim_counts.sum(dim=0) + (alpha * self.num_rels)

        target_range = target_max - target_min

        c_ij = self.M_sim_counts + alpha

        p_ij = c_ij / total_count
        p_i = row_sums.unsqueeze(1) / total_count
        p_j = col_sums.unsqueeze(0) / total_count

        lift = p_ij / (p_i * p_j + epsilon)
        pmi = torch.log(lift + epsilon)

        sigmoid_val = torch.sigmoid(pmi)

        M_sim_out = target_min + sigmoid_val * target_range

        return M_sim_out


def _shared_w_code(shared_w):
    if shared_w == 'unweighted':
        return 0
    if shared_w == 'dual_msim':
        return 1
    if shared_w == 'cross_msim':
        return 2
    raise ValueError(f'unsupported shared_w={shared_w}')


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _fast_tagmax_update_one(keys, scores, tags, lens, norms, m_trans,
                                source, targets, rels, ppr_k, ppr_alpha, ppr_beta,
                                temp_keys, temp_scores, temp_tags,
                                stamp, pos, epoch):
        if lens[source] == 0:
            keys[source, 0] = source
            scores[source, 0] = 1.0
            tags[source, 0] = -1
            lens[source] = 1
            norms[source] = 0.0

        for i in range(targets.shape[0]):
            target = targets[i]
            if lens[target] == 0:
                keys[target, 0] = target
                scores[target, 0] = 1.0
                tags[target, 0] = -1
                lens[target] = 1
                norms[target] = 0.0

        last_norm = norms[source]
        new_norm = last_norm * ppr_beta + ppr_beta
        s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
        s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
        s2_per_event = s2_total / targets.shape[0]

        count = 0
        slen = lens[source]
        for j in range(slen):
            k = keys[source, j]
            temp_keys[count] = k
            temp_scores[count] = scores[source, j] * s1
            temp_tags[count] = tags[source, j]
            stamp[k] = epoch
            pos[k] = count
            count += 1

        for i in range(targets.shape[0]):
            target = targets[i]
            r_update = rels[i]
            tlen = lens[target]
            sum_logic_weighted = 0.0
            for j in range(tlen):
                r_k = tags[target, j]
                factor = 1.0
                if r_k != -1:
                    factor = m_trans[r_update, r_k]
                c = scores[target, j] * factor
                if c > 0.0:
                    sum_logic_weighted += c
            if sum_logic_weighted <= 0.0:
                continue

            scale = s2_per_event / sum_logic_weighted
            for j in range(tlen):
                k = keys[target, j]
                r_k = tags[target, j]
                factor = 1.0
                if r_k != -1:
                    factor = m_trans[r_update, r_k]
                c = scores[target, j] * factor
                if c <= 0.0:
                    continue
                contrib = c * scale
                if stamp[k] == epoch:
                    p = pos[k]
                    if contrib > temp_scores[p]:
                        temp_scores[p] = contrib
                        temp_tags[p] = r_update
                else:
                    temp_keys[count] = k
                    temp_scores[count] = contrib
                    temp_tags[count] = r_update
                    stamp[k] = epoch
                    pos[k] = count
                    count += 1

        if stamp[source] == epoch:
            p = pos[source]
            temp_scores[p] += ppr_alpha
            temp_tags[p] = -1
        else:
            temp_keys[count] = source
            temp_scores[count] = ppr_alpha
            temp_tags[count] = -1
            stamp[source] = epoch
            pos[source] = count
            count += 1

        if count > ppr_k:
            order = np.argsort(temp_scores[:count])
            sum_top = 0.0
            for out_i in range(ppr_k):
                idx = order[count - 1 - out_i]
                sum_top += temp_scores[idx]
            if sum_top > 0.0:
                inv = 1.0 / sum_top
                for out_i in range(ppr_k):
                    idx = order[count - 1 - out_i]
                    keys[source, out_i] = temp_keys[idx]
                    scores[source, out_i] = temp_scores[idx] * inv
                    tags[source, out_i] = temp_tags[idx]
                lens[source] = ppr_k
            else:
                keys[source, 0] = source
                scores[source, 0] = 1.0
                tags[source, 0] = -1
                lens[source] = 1
        else:
            order = np.argsort(temp_scores[:count])
            for out_i in range(count):
                idx = order[count - 1 - out_i]
                keys[source, out_i] = temp_keys[idx]
                scores[source, out_i] = temp_scores[idx]
                tags[source, out_i] = temp_tags[idx]
            lens[source] = count

        norms[source] = new_norm


    @njit(parallel=True, cache=True)
    def _fast_tagmax_update_source_join_batch(keys, scores, tags, lens, norms, m_trans,
                                              source_ids, group_offsets, flat_targets, flat_rels,
                                              ppr_k, ppr_alpha, ppr_beta,
                                              out_keys, out_scores, out_tags, out_lens, out_norms):
        num_groups = source_ids.shape[0]
        for g in prange(num_groups):
            source = source_ids[g]
            start = group_offsets[g]
            end = group_offsets[g + 1]
            num_events = end - start

            last_norm = norms[source]
            new_norm = last_norm * ppr_beta + ppr_beta
            s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
            s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
            s2_per_event = s2_total / num_events

            capacity = ppr_k * (num_events + 1) + 2
            table_size = 1
            while table_size < capacity * 2:
                table_size *= 2
            table_mask = table_size - 1
            table_keys = np.empty(table_size, dtype=np.int32)
            table_pos = np.empty(table_size, dtype=np.int32)
            for h0 in range(table_size):
                table_keys[h0] = -1

            uniq_keys = np.empty(capacity, dtype=np.int32)
            uniq_scores = np.empty(capacity, dtype=np.float32)
            uniq_tags = np.empty(capacity, dtype=np.int32)
            uniq_count = 0

            slen = lens[source]
            for j in range(slen):
                k = keys[source, j]
                sc = scores[source, j] * s1
                tg = tags[source, j]
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        table_keys[h] = k
                        table_pos[h] = uniq_count
                        uniq_keys[uniq_count] = k
                        uniq_scores[uniq_count] = sc
                        uniq_tags[uniq_count] = tg
                        uniq_count += 1
                        break
                    if old_key == k:
                        p = table_pos[h]
                        if sc > uniq_scores[p]:
                            uniq_scores[p] = sc
                            uniq_tags[p] = tg
                        break
                    h = (h + 1) & table_mask

            for i in range(start, end):
                target = flat_targets[i]
                r_update = flat_rels[i]
                tlen = lens[target]
                sum_logic_weighted = 0.0
                for j in range(tlen):
                    r_k = tags[target, j]
                    factor = 1.0
                    if r_k != -1:
                        factor = m_trans[r_update, r_k]
                    c = scores[target, j] * factor
                    if c > 0.0:
                        sum_logic_weighted += c
                if sum_logic_weighted <= 0.0:
                    continue

                scale = s2_per_event / sum_logic_weighted
                for j in range(tlen):
                    k = keys[target, j]
                    r_k = tags[target, j]
                    factor = 1.0
                    if r_k != -1:
                        factor = m_trans[r_update, r_k]
                    c = scores[target, j] * factor
                    if c <= 0.0:
                        continue
                    sc = c * scale
                    h = (np.int64(k) * 2654435761) & table_mask
                    while True:
                        old_key = table_keys[h]
                        if old_key == -1:
                            table_keys[h] = k
                            table_pos[h] = uniq_count
                            uniq_keys[uniq_count] = k
                            uniq_scores[uniq_count] = sc
                            uniq_tags[uniq_count] = r_update
                            uniq_count += 1
                            break
                        if old_key == k:
                            p = table_pos[h]
                            if sc > uniq_scores[p]:
                                uniq_scores[p] = sc
                                uniq_tags[p] = r_update
                            break
                        h = (h + 1) & table_mask

            h = (np.int64(source) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    table_keys[h] = source
                    table_pos[h] = uniq_count
                    uniq_keys[uniq_count] = source
                    uniq_scores[uniq_count] = ppr_alpha
                    uniq_tags[uniq_count] = -1
                    uniq_count += 1
                    break
                if old_key == source:
                    p = table_pos[h]
                    uniq_scores[p] += ppr_alpha
                    uniq_tags[p] = -1
                    break
                h = (h + 1) & table_mask

            if uniq_count <= 0:
                uniq_keys[uniq_count] = source
                uniq_scores[uniq_count] = 1.0
                uniq_tags[uniq_count] = -1
                uniq_count += 1

            if uniq_count > ppr_k:
                score_order = np.argsort(uniq_scores[:uniq_count])
                sum_top = 0.0
                for out_i in range(ppr_k):
                    idx = score_order[uniq_count - 1 - out_i]
                    sum_top += uniq_scores[idx]
                if sum_top > 0.0:
                    inv = 1.0 / sum_top
                    for out_i in range(ppr_k):
                        idx = score_order[uniq_count - 1 - out_i]
                        out_keys[g, out_i] = uniq_keys[idx]
                        out_scores[g, out_i] = uniq_scores[idx] * inv
                        out_tags[g, out_i] = uniq_tags[idx]
                    out_lens[g] = ppr_k
                else:
                    out_keys[g, 0] = source
                    out_scores[g, 0] = 1.0
                    out_tags[g, 0] = -1
                    out_lens[g] = 1
            else:
                score_order = np.argsort(uniq_scores[:uniq_count])
                for out_i in range(uniq_count):
                    idx = score_order[uniq_count - 1 - out_i]
                    out_keys[g, out_i] = uniq_keys[idx]
                    out_scores[g, out_i] = uniq_scores[idx]
                    out_tags[g, out_i] = uniq_tags[idx]
                out_lens[g] = uniq_count

            out_norms[g] = new_norm


    @njit(parallel=True, cache=True)
    def _fast_tagmax_commit_source_join(keys, scores, tags, lens, norms,
                                        source_ids, out_keys, out_scores, out_tags, out_lens, out_norms):
        num_groups = source_ids.shape[0]
        for g in prange(num_groups):
            source = source_ids[g]
            n = out_lens[g]
            for j in range(n):
                keys[source, j] = out_keys[g, j]
                scores[source, j] = out_scores[g, j]
                tags[source, j] = out_tags[g, j]
            lens[source] = n
            norms[source] = out_norms[g]


    @njit(parallel=True, cache=True)
    def _fast_tagsum_update_source_join_batch(keys, scores, tags, tag_scores, lens, norms, m_trans,
                                              source_ids, group_offsets, flat_targets, flat_rels,
                                              ppr_k, ppr_alpha, ppr_beta,
                                              out_keys, out_scores, out_tags, out_tag_scores,
                                              out_lens, out_norms):
        num_groups = source_ids.shape[0]
        for g in prange(num_groups):
            source = source_ids[g]
            start = group_offsets[g]
            end = group_offsets[g + 1]
            num_events = end - start

            last_norm = norms[source]
            new_norm = last_norm * ppr_beta + ppr_beta
            s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
            s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
            s2_per_event = s2_total / num_events

            capacity = ppr_k * (num_events + 1) + 2
            table_size = 1
            while table_size < capacity * 2:
                table_size *= 2
            table_mask = table_size - 1
            table_keys = np.empty(table_size, dtype=np.int32)
            table_pos = np.empty(table_size, dtype=np.int32)
            for h0 in range(table_size):
                table_keys[h0] = -1

            uniq_keys = np.empty(capacity, dtype=np.int32)
            uniq_scores = np.empty(capacity, dtype=np.float32)
            uniq_tags = np.empty(capacity, dtype=np.int32)
            uniq_tag_scores = np.empty(capacity, dtype=np.float32)
            uniq_count = 0

            slen = lens[source]
            for j in range(slen):
                k = keys[source, j]
                sc = scores[source, j] * s1
                tg = tags[source, j]
                tsc = tag_scores[source, j] * s1
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        table_keys[h] = k
                        table_pos[h] = uniq_count
                        uniq_keys[uniq_count] = k
                        uniq_scores[uniq_count] = sc
                        uniq_tags[uniq_count] = tg
                        uniq_tag_scores[uniq_count] = tsc
                        uniq_count += 1
                        break
                    if old_key == k:
                        p = table_pos[h]
                        uniq_scores[p] += sc
                        if tsc > uniq_tag_scores[p]:
                            uniq_tag_scores[p] = tsc
                            uniq_tags[p] = tg
                        break
                    h = (h + 1) & table_mask

            for i in range(start, end):
                target = flat_targets[i]
                r_update = flat_rels[i]
                tlen = lens[target]
                sum_logic_weighted = 0.0
                for j in range(tlen):
                    r_k = tags[target, j]
                    factor = 1.0
                    if r_k != -1:
                        factor = m_trans[r_update, r_k]
                    c = scores[target, j] * factor
                    if c > 0.0:
                        sum_logic_weighted += c
                if sum_logic_weighted <= 0.0:
                    continue

                scale = s2_per_event / sum_logic_weighted
                for j in range(tlen):
                    k = keys[target, j]
                    r_k = tags[target, j]
                    factor = 1.0
                    if r_k != -1:
                        factor = m_trans[r_update, r_k]
                    c = scores[target, j] * factor
                    if c <= 0.0:
                        continue
                    contrib = c * scale
                    h = (np.int64(k) * 2654435761) & table_mask
                    while True:
                        old_key = table_keys[h]
                        if old_key == -1:
                            table_keys[h] = k
                            table_pos[h] = uniq_count
                            uniq_keys[uniq_count] = k
                            uniq_scores[uniq_count] = contrib
                            if k == source:
                                uniq_tags[uniq_count] = -1
                                uniq_tag_scores[uniq_count] = 0.0
                            else:
                                uniq_tags[uniq_count] = r_update
                                uniq_tag_scores[uniq_count] = contrib
                            uniq_count += 1
                            break
                        if old_key == k:
                            p = table_pos[h]
                            uniq_scores[p] += contrib
                            if k != source and contrib > uniq_tag_scores[p]:
                                uniq_tag_scores[p] = contrib
                                uniq_tags[p] = r_update
                            break
                        h = (h + 1) & table_mask

            h = (np.int64(source) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    table_keys[h] = source
                    table_pos[h] = uniq_count
                    uniq_keys[uniq_count] = source
                    uniq_scores[uniq_count] = ppr_alpha
                    uniq_tags[uniq_count] = -1
                    uniq_tag_scores[uniq_count] = ppr_alpha
                    uniq_count += 1
                    break
                if old_key == source:
                    p = table_pos[h]
                    uniq_scores[p] += ppr_alpha
                    uniq_tags[p] = -1
                    uniq_tag_scores[p] += ppr_alpha
                    break
                h = (h + 1) & table_mask

            if uniq_count > ppr_k:
                score_order = np.argsort(uniq_scores[:uniq_count])
                sum_top = 0.0
                for out_i in range(ppr_k):
                    idx = score_order[uniq_count - 1 - out_i]
                    sum_top += uniq_scores[idx]
                if sum_top > 0.0:
                    inv = 1.0 / sum_top
                    for out_i in range(ppr_k):
                        idx = score_order[uniq_count - 1 - out_i]
                        out_keys[g, out_i] = uniq_keys[idx]
                        out_scores[g, out_i] = uniq_scores[idx] * inv
                        out_tags[g, out_i] = uniq_tags[idx]
                        out_tag_scores[g, out_i] = uniq_tag_scores[idx] * inv
                    out_lens[g] = ppr_k
                else:
                    out_keys[g, 0] = source
                    out_scores[g, 0] = 1.0
                    out_tags[g, 0] = -1
                    out_tag_scores[g, 0] = 1.0
                    out_lens[g] = 1
            else:
                score_order = np.argsort(uniq_scores[:uniq_count])
                for out_i in range(uniq_count):
                    idx = score_order[uniq_count - 1 - out_i]
                    out_keys[g, out_i] = uniq_keys[idx]
                    out_scores[g, out_i] = uniq_scores[idx]
                    out_tags[g, out_i] = uniq_tags[idx]
                    out_tag_scores[g, out_i] = uniq_tag_scores[idx]
                out_lens[g] = uniq_count

            out_norms[g] = new_norm


    @njit(parallel=True, cache=True)
    def _fast_tagsum_commit_source_join(keys, scores, tags, tag_scores, lens, norms,
                                        source_ids, out_keys, out_scores, out_tags,
                                        out_tag_scores, out_lens, out_norms):
        num_groups = source_ids.shape[0]
        for g in prange(num_groups):
            source = source_ids[g]
            n = out_lens[g]
            for j in range(n):
                keys[source, j] = out_keys[g, j]
                scores[source, j] = out_scores[g, j]
                tags[source, j] = out_tags[g, j]
                tag_scores[source, j] = out_tag_scores[g, j]
            lens[source] = n
            norms[source] = out_norms[g]


    @njit(parallel=True, cache=True)
    def _fast_perrel_update_source_join_batch(entry_keys, rel_keys, rel_scores, entry_lens, rel_lens,
                                              norms, m_trans, source_ids, group_offsets,
                                              flat_targets, flat_rels, ppr_k, top_k_relation,
                                              num_rels, ppr_alpha, ppr_beta, use_mtrans,
                                              out_entry_keys, out_rel_keys, out_rel_scores,
                                              out_entry_lens, out_rel_lens, out_norms):
        num_groups = source_ids.shape[0]
        rel_base = num_rels + 1
        self_code = num_rels
        for g in prange(num_groups):
            source = source_ids[g]
            start = group_offsets[g]
            end = group_offsets[g + 1]
            num_events = end - start

            last_norm = norms[source]
            new_norm = last_norm * ppr_beta + ppr_beta
            s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
            s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
            s2_per_event = s2_total / num_events

            cand_cap = ppr_k * top_k_relation + ppr_k * num_events + 2
            cand_combo = np.empty(cand_cap, dtype=np.int64)
            cand_scores = np.empty(cand_cap, dtype=np.float32)
            count = 0

            slen = entry_lens[source]
            for eidx in range(slen):
                entry = entry_keys[source, eidx]
                rlen = rel_lens[source, eidx]
                for ridx in range(rlen):
                    r = rel_keys[source, eidx, ridx]
                    rel_code = self_code if r == -1 else r
                    cand_combo[count] = np.int64(entry) * rel_base + rel_code
                    cand_scores[count] = rel_scores[source, eidx, ridx] * s1
                    count += 1

            for i in range(start, end):
                target = flat_targets[i]
                r_update = flat_rels[i]
                tlen = entry_lens[target]
                total_w = 0.0
                for eidx in range(tlen):
                    rlen = rel_lens[target, eidx]
                    m = 0.0
                    for ridx in range(rlen):
                        r = rel_keys[target, eidx, ridx]
                        w = rel_scores[target, eidx, ridx]
                        if use_mtrans:
                            factor = 1.0
                            if r != -1:
                                factor = m_trans[r_update, r]
                            m += w * factor
                        else:
                            m += w
                    if m > 0.0:
                        total_w += m
                if total_w <= 0.0:
                    continue
                scale = s2_per_event / total_w
                for eidx in range(tlen):
                    entry = entry_keys[target, eidx]
                    rlen = rel_lens[target, eidx]
                    m = 0.0
                    for ridx in range(rlen):
                        r = rel_keys[target, eidx, ridx]
                        w = rel_scores[target, eidx, ridx]
                        if use_mtrans:
                            factor = 1.0
                            if r != -1:
                                factor = m_trans[r_update, r]
                            m += w * factor
                        else:
                            m += w
                    if m <= 0.0:
                        continue
                    cand_combo[count] = np.int64(entry) * rel_base + r_update
                    cand_scores[count] = m * scale
                    count += 1

            cand_combo[count] = np.int64(source) * rel_base + self_code
            cand_scores[count] = ppr_alpha
            count += 1

            if count <= 0:
                out_entry_keys[g, 0] = source
                out_rel_keys[g, 0, 0] = -1
                out_rel_scores[g, 0, 0] = 1.0
                out_entry_lens[g] = 1
                out_rel_lens[g, 0] = 1
                out_norms[g] = new_norm
                continue

            order = np.argsort(cand_combo[:count])
            tmp_entry_keys = np.empty(count, dtype=np.int32)
            tmp_totals = np.zeros(count, dtype=np.float32)
            tmp_rel_keys = np.empty((count, top_k_relation), dtype=np.int32)
            tmp_rel_scores = np.zeros((count, top_k_relation), dtype=np.float32)
            tmp_rel_lens = np.zeros(count, dtype=np.int32)
            entry_count = 0

            first = order[0]
            cur_combo = cand_combo[first]
            cur_score = cand_scores[first]

            cur_entry = np.int32(cur_combo // rel_base)
            cur_rel_code = np.int32(cur_combo - np.int64(cur_entry) * rel_base)
            cur_rel = -1 if cur_rel_code == self_code else cur_rel_code
            cur_rel_count = 0
            cur_rel_keys = np.empty(top_k_relation, dtype=np.int32)
            cur_rel_scores = np.zeros(top_k_relation, dtype=np.float32)

            for oi in range(1, count + 1):
                flush_pair = False
                if oi == count:
                    flush_pair = True
                else:
                    p = order[oi]
                    combo = cand_combo[p]
                    if combo == cur_combo:
                        cur_score += cand_scores[p]
                    else:
                        flush_pair = True

                if flush_pair:
                    if cur_rel_count < top_k_relation:
                        cur_rel_keys[cur_rel_count] = cur_rel
                        cur_rel_scores[cur_rel_count] = cur_score
                        cur_rel_count += 1
                    else:
                        min_idx = 0
                        min_score = cur_rel_scores[0]
                        for ridx in range(1, top_k_relation):
                            if cur_rel_scores[ridx] < min_score:
                                min_score = cur_rel_scores[ridx]
                                min_idx = ridx
                        if cur_score > min_score:
                            cur_rel_keys[min_idx] = cur_rel
                            cur_rel_scores[min_idx] = cur_score

                    if oi < count:
                        p = order[oi]
                        next_combo = cand_combo[p]
                        next_entry = np.int32(next_combo // rel_base)
                        next_rel_code = np.int32(next_combo - np.int64(next_entry) * rel_base)
                        if next_entry != cur_entry:
                            tmp_entry_keys[entry_count] = cur_entry
                            tmp_rel_lens[entry_count] = cur_rel_count
                            total = 0.0
                            for ridx in range(cur_rel_count):
                                tmp_rel_keys[entry_count, ridx] = cur_rel_keys[ridx]
                                tmp_rel_scores[entry_count, ridx] = cur_rel_scores[ridx]
                                total += cur_rel_scores[ridx]
                            tmp_totals[entry_count] = total
                            entry_count += 1

                            cur_entry = next_entry
                            cur_rel_count = 0
                            for ridx in range(top_k_relation):
                                cur_rel_scores[ridx] = 0.0

                        cur_combo = next_combo
                        cur_score = cand_scores[p]
                        cur_rel = -1 if next_rel_code == self_code else next_rel_code

            tmp_entry_keys[entry_count] = cur_entry
            tmp_rel_lens[entry_count] = cur_rel_count
            total = 0.0
            for ridx in range(cur_rel_count):
                tmp_rel_keys[entry_count, ridx] = cur_rel_keys[ridx]
                tmp_rel_scores[entry_count, ridx] = cur_rel_scores[ridx]
                total += cur_rel_scores[ridx]
            tmp_totals[entry_count] = total
            entry_count += 1

            keep_count = entry_count
            if keep_count > ppr_k:
                keep_count = ppr_k
            entry_order = np.argsort(tmp_totals[:entry_count])
            total_keep = 0.0
            for out_i in range(keep_count):
                idx = entry_order[entry_count - 1 - out_i]
                total_keep += tmp_totals[idx]

            if total_keep > 0.0:
                inv = 1.0 / total_keep
                for out_i in range(keep_count):
                    idx = entry_order[entry_count - 1 - out_i]
                    out_entry_keys[g, out_i] = tmp_entry_keys[idx]
                    out_rel_lens[g, out_i] = tmp_rel_lens[idx]
                    for ridx in range(tmp_rel_lens[idx]):
                        out_rel_keys[g, out_i, ridx] = tmp_rel_keys[idx, ridx]
                        out_rel_scores[g, out_i, ridx] = tmp_rel_scores[idx, ridx] * inv
                out_entry_lens[g] = keep_count
            else:
                out_entry_keys[g, 0] = source
                out_rel_keys[g, 0, 0] = -1
                out_rel_scores[g, 0, 0] = 1.0
                out_entry_lens[g] = 1
                out_rel_lens[g, 0] = 1

            out_norms[g] = new_norm


    @njit(parallel=True, cache=True)
    def _fast_perrel_commit_source_join(entry_keys, rel_keys, rel_scores, entry_lens,
                                        rel_lens, norms, source_ids, out_entry_keys,
                                        out_rel_keys, out_rel_scores, out_entry_lens,
                                        out_rel_lens, out_norms, top_k_relation):
        num_groups = source_ids.shape[0]
        for g in prange(num_groups):
            source = source_ids[g]
            n = out_entry_lens[g]
            for eidx in range(n):
                entry_keys[source, eidx] = out_entry_keys[g, eidx]
                rlen = out_rel_lens[g, eidx]
                rel_lens[source, eidx] = rlen
                for ridx in range(rlen):
                    rel_keys[source, eidx, ridx] = out_rel_keys[g, eidx, ridx]
                    rel_scores[source, eidx, ridx] = out_rel_scores[g, eidx, ridx]
                for ridx in range(rlen, top_k_relation):
                    rel_scores[source, eidx, ridx] = 0.0
            entry_lens[source] = n
            norms[source] = out_norms[g]


    @njit(cache=True)
    def _fast_perrel_predict_batch(entry_keys, rel_keys, rel_scores, entry_lens,
                                   rel_lens, m_sim, batch_data, neg_samples,
                                   pos_out, neg_out, stamp, pos, epoch):
        num_queries = batch_data.shape[0]
        num_negs = neg_samples.shape[1]
        num_rels = m_sim.shape[0]

        for i in range(num_queries):
            s = batch_data[i, 0]
            r = batch_data[i, 1]
            slen = entry_lens[s]

            for j in range(slen):
                k = entry_keys[s, j]
                stamp[k] = epoch
                pos[k] = j

            o = batch_data[i, 2]
            if stamp[o] == epoch:
                eidx = pos[o]
                total = 0.0
                rlen = rel_lens[s, eidx]
                for ridx in range(rlen):
                    rel = rel_keys[s, eidx, ridx]
                    w = 1.0
                    if rel != -1:
                        w = m_sim[rel, r]
                    total += rel_scores[s, eidx, ridx] * w
                pos_out[i, 0] = total

            for j in range(num_negs):
                ng = neg_samples[i, j]
                if ng < 0:
                    continue
                if stamp[ng] == epoch:
                    eidx = pos[ng]
                    total = 0.0
                    rlen = rel_lens[s, eidx]
                    for ridx in range(rlen):
                        rel = rel_keys[s, eidx, ridx]
                        w = 1.0
                        if rel != -1:
                            w = m_sim[rel, r]
                        total += rel_scores[s, eidx, ridx] * w
                    neg_out[i, j] = total

            epoch += 1
            if epoch > 9000000000000000000:
                for z in range(stamp.shape[0]):
                    stamp[z] = 0
                epoch = 1
        return epoch


    @njit(cache=True)
    def _fast_tagmax_predict_batch(keys, scores, tags, lens, m_sim,
                                   batch_data, neg_samples,
                                   pos_out, neg_out,
                                   stamp, pos, epoch):
        num_queries = batch_data.shape[0]
        num_negs = neg_samples.shape[1]
        num_rels = m_sim.shape[0]
        orig_rels = num_rels // 2

        for i in range(num_queries):
            s = batch_data[i, 0]
            r = batch_data[i, 1]
            o = batch_data[i, 2]
            slen = lens[s]

            for j in range(slen):
                k = keys[s, j]
                stamp[k] = epoch
                pos[k] = j

            if stamp[o] == epoch:
                p = pos[o]
                tag = tags[s, p]
                w = 1.0
                if tag != -1:
                    w = m_sim[tag, r]
                pos_out[i, 0] = scores[s, p] * w

            for j in range(num_negs):
                ng = neg_samples[i, j]
                if ng == -1:
                    continue
                if stamp[ng] == epoch:
                    p = pos[ng]
                    tag = tags[s, p]
                    w = 1.0
                    if tag != -1:
                        w = m_sim[tag, r]
                    neg_out[i, j] = scores[s, p] * w

            epoch += 1
            if epoch > 9000000000000000000:
                stamp[:] = 0
                epoch = 1

        return epoch


    @njit(parallel=True, cache=True)
    def _fast_tag_predict_parts_batch(keys, scores, tags, lens, m_sim,
                                      batch_data, neg_samples, shared_w_code,
                                      top_share, pos_direct, neg_direct,
                                      pos_shared, neg_shared):
        num_queries = batch_data.shape[0]
        num_negs = neg_samples.shape[1]
        num_rels = m_sim.shape[0]
        orig_rels = num_rels // 2

        for i in prange(num_queries):
            s = batch_data[i, 0]
            r = batch_data[i, 1]
            o = batch_data[i, 2]
            r_inv = (r + orig_rels) % num_rels
            slen = lens[s]

            src_share_len = slen
            if top_share > 0 and top_share < src_share_len:
                src_share_len = top_share

            table_size = 1
            while table_size < slen * 2 + 1:
                table_size *= 2
            table_mask = table_size - 1
            table_keys = np.empty(table_size, dtype=np.int32)
            table_pos = np.empty(table_size, dtype=np.int32)
            for h0 in range(table_size):
                table_keys[h0] = -1

            for j in range(slen):
                k = keys[s, j]
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        table_keys[h] = k
                        table_pos[h] = j
                        break
                    if old_key == k:
                        table_pos[h] = j
                        break
                    h = (h + 1) & table_mask

            h = (np.int64(o) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    break
                if old_key == o:
                    p = table_pos[h]
                    tag = tags[s, p]
                    w = 1.0
                    if tag != -1:
                        w = m_sim[tag, r]
                    pos_direct[i, 0] = scores[s, p] * w
                    break
                h = (h + 1) & table_mask

            tlen = lens[o]
            target_share_len = tlen
            if top_share > 0 and top_share < target_share_len:
                target_share_len = top_share
            total = 0.0
            for tj in range(target_share_len):
                k = keys[o, tj]
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == k:
                        sp = table_pos[h]
                        if sp < src_share_len:
                            if shared_w_code == 0:
                                total += scores[s, sp] * scores[o, tj]
                            elif shared_w_code == 1:
                                st = tags[s, sp]
                                tt = tags[o, tj]
                                ws = 1.0
                                wo = 1.0
                                if st != -1:
                                    ws = m_sim[st, r]
                                if tt != -1:
                                    wo = m_sim[tt, r_inv]
                                total += scores[s, sp] * scores[o, tj] * ws * wo
                            else:
                                st = tags[s, sp]
                                tt = tags[o, tj]
                                wm = 1.0
                                if st != -1 and tt != -1:
                                    wm = m_sim[st, tt]
                                total += scores[s, sp] * scores[o, tj] * wm
                        break
                    h = (h + 1) & table_mask
            pos_shared[i, 0] = total

            for j in range(num_negs):
                ng = neg_samples[i, j]
                if ng < 0:
                    continue

                h = (np.int64(ng) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == ng:
                        p = table_pos[h]
                        tag = tags[s, p]
                        w = 1.0
                        if tag != -1:
                            w = m_sim[tag, r]
                        neg_direct[i, j] = scores[s, p] * w
                        break
                    h = (h + 1) & table_mask

                tlen = lens[ng]
                target_share_len = tlen
                if top_share > 0 and top_share < target_share_len:
                    target_share_len = top_share
                total = 0.0
                for tj in range(target_share_len):
                    k = keys[ng, tj]
                    h = (np.int64(k) * 2654435761) & table_mask
                    while True:
                        old_key = table_keys[h]
                        if old_key == -1:
                            break
                        if old_key == k:
                            sp = table_pos[h]
                            if sp < src_share_len:
                                if shared_w_code == 0:
                                    total += scores[s, sp] * scores[ng, tj]
                                elif shared_w_code == 1:
                                    st = tags[s, sp]
                                    tt = tags[ng, tj]
                                    ws = 1.0
                                    wo = 1.0
                                    if st != -1:
                                        ws = m_sim[st, r]
                                    if tt != -1:
                                        wo = m_sim[tt, r_inv]
                                    total += scores[s, sp] * scores[ng, tj] * ws * wo
                                else:
                                    st = tags[s, sp]
                                    tt = tags[ng, tj]
                                    wm = 1.0
                                    if st != -1 and tt != -1:
                                        wm = m_sim[st, tt]
                                    total += scores[s, sp] * scores[ng, tj] * wm
                            break
                        h = (h + 1) & table_mask
                neg_shared[i, j] = total


    @njit(parallel=True, cache=True)
    def _fast_perrel_predict_parts_batch(entry_keys, rel_keys, rel_scores,
                                         entry_lens, rel_lens, m_sim,
                                         batch_data, neg_samples, shared_w_code,
                                         top_share, pos_direct, neg_direct,
                                         pos_shared, neg_shared):
        num_queries = batch_data.shape[0]
        num_negs = neg_samples.shape[1]
        num_rels = m_sim.shape[0]
        orig_rels = num_rels // 2

        for i in prange(num_queries):
            s = batch_data[i, 0]
            r = batch_data[i, 1]
            o = batch_data[i, 2]
            r_inv = (r + orig_rels) % num_rels
            slen = entry_lens[s]

            src_share_len = slen
            if top_share > 0 and top_share < src_share_len:
                src_share_len = top_share

            table_size = 1
            while table_size < slen * 2 + 1:
                table_size *= 2
            table_mask = table_size - 1
            table_keys = np.empty(table_size, dtype=np.int32)
            table_pos = np.empty(table_size, dtype=np.int32)
            for h0 in range(table_size):
                table_keys[h0] = -1

            for eidx in range(slen):
                k = entry_keys[s, eidx]
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        table_keys[h] = k
                        table_pos[h] = eidx
                        break
                    if old_key == k:
                        table_pos[h] = eidx
                        break
                    h = (h + 1) & table_mask

            h = (np.int64(o) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    break
                if old_key == o:
                    eidx = table_pos[h]
                    total = 0.0
                    rlen = rel_lens[s, eidx]
                    for ridx in range(rlen):
                        rel = rel_keys[s, eidx, ridx]
                        w = 1.0
                        if rel != -1:
                            w = m_sim[rel, r]
                        total += rel_scores[s, eidx, ridx] * w
                    pos_direct[i, 0] = total
                    break
                h = (h + 1) & table_mask

            tlen = entry_lens[o]
            target_share_len = tlen
            if top_share > 0 and top_share < target_share_len:
                target_share_len = top_share
            total_shared = 0.0
            for teidx in range(target_share_len):
                entry = entry_keys[o, teidx]
                h = (np.int64(entry) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == entry:
                        seidx = table_pos[h]
                        if seidx < src_share_len:
                            srlen = rel_lens[s, seidx]
                            trlen = rel_lens[o, teidx]
                            if shared_w_code == 0:
                                sw = 0.0
                                tw = 0.0
                                for ridx in range(srlen):
                                    sw += rel_scores[s, seidx, ridx]
                                for ridx in range(trlen):
                                    tw += rel_scores[o, teidx, ridx]
                                total_shared += sw * tw
                            elif shared_w_code == 1:
                                sw = 0.0
                                tw = 0.0
                                for ridx in range(srlen):
                                    rel = rel_keys[s, seidx, ridx]
                                    w = 1.0
                                    if rel != -1:
                                        w = m_sim[rel, r]
                                    sw += rel_scores[s, seidx, ridx] * w
                                for ridx in range(trlen):
                                    rel = rel_keys[o, teidx, ridx]
                                    w = 1.0
                                    if rel != -1:
                                        w = m_sim[rel, r_inv]
                                    tw += rel_scores[o, teidx, ridx] * w
                                total_shared += sw * tw
                            else:
                                for sridx in range(srlen):
                                    rs = rel_keys[s, seidx, sridx]
                                    ws0 = rel_scores[s, seidx, sridx]
                                    for tridx in range(trlen):
                                        rt = rel_keys[o, teidx, tridx]
                                        wm = 1.0
                                        if rs != -1 and rt != -1:
                                            wm = m_sim[rs, rt]
                                        total_shared += ws0 * rel_scores[o, teidx, tridx] * wm
                        break
                    h = (h + 1) & table_mask
            pos_shared[i, 0] = total_shared

            for j in range(num_negs):
                ng = neg_samples[i, j]
                if ng < 0:
                    continue

                h = (np.int64(ng) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == ng:
                        eidx = table_pos[h]
                        total = 0.0
                        rlen = rel_lens[s, eidx]
                        for ridx in range(rlen):
                            rel = rel_keys[s, eidx, ridx]
                            w = 1.0
                            if rel != -1:
                                w = m_sim[rel, r]
                            total += rel_scores[s, eidx, ridx] * w
                        neg_direct[i, j] = total
                        break
                    h = (h + 1) & table_mask

                tlen = entry_lens[ng]
                target_share_len = tlen
                if top_share > 0 and top_share < target_share_len:
                    target_share_len = top_share
                total_shared = 0.0
                for teidx in range(target_share_len):
                    entry = entry_keys[ng, teidx]
                    h = (np.int64(entry) * 2654435761) & table_mask
                    while True:
                        old_key = table_keys[h]
                        if old_key == -1:
                            break
                        if old_key == entry:
                            seidx = table_pos[h]
                            if seidx < src_share_len:
                                srlen = rel_lens[s, seidx]
                                trlen = rel_lens[ng, teidx]
                                if shared_w_code == 0:
                                    sw = 0.0
                                    tw = 0.0
                                    for ridx in range(srlen):
                                        sw += rel_scores[s, seidx, ridx]
                                    for ridx in range(trlen):
                                        tw += rel_scores[ng, teidx, ridx]
                                    total_shared += sw * tw
                                elif shared_w_code == 1:
                                    sw = 0.0
                                    tw = 0.0
                                    for ridx in range(srlen):
                                        rel = rel_keys[s, seidx, ridx]
                                        w = 1.0
                                        if rel != -1:
                                            w = m_sim[rel, r]
                                        sw += rel_scores[s, seidx, ridx] * w
                                    for ridx in range(trlen):
                                        rel = rel_keys[ng, teidx, ridx]
                                        w = 1.0
                                        if rel != -1:
                                            w = m_sim[rel, r_inv]
                                        tw += rel_scores[ng, teidx, ridx] * w
                                    total_shared += sw * tw
                                else:
                                    for sridx in range(srlen):
                                        rs = rel_keys[s, seidx, sridx]
                                        ws0 = rel_scores[s, seidx, sridx]
                                        for tridx in range(trlen):
                                            rt = rel_keys[ng, teidx, tridx]
                                            wm = 1.0
                                            if rs != -1 and rt != -1:
                                                wm = m_sim[rs, rt]
                                            total_shared += ws0 * rel_scores[ng, teidx, tridx] * wm
                            break
                        h = (h + 1) & table_mask
                neg_shared[i, j] = total_shared
class FastPerRelSourceJoinPredictor:
    R_SELF = -1

    def __init__(self, num_nodes, num_rels, *,
                 c_storage='per_rel', shared_w='dual_msim',
                 decay_level='timestamp', per_rel_use_mtrans=False,
                 ppr_k=1000, top_k_relation=4,
                 ppr_alpha=0.1, ppr_beta=0.8, device='cuda:0',
                 top_share=-1,
                 max_events_in_single_batch=20000,
                 source_join_log_batches=True):
        if not NUMBA_AVAILABLE:
            raise RuntimeError('FastPerRelSourceJoinPredictor requires numba')
        if c_storage != 'per_rel':
            raise ValueError('FastPerRelSourceJoinPredictor requires c_storage=per_rel')
        if decay_level != 'timestamp':
            raise ValueError('FastPerRelSourceJoinPredictor only supports decay_level=timestamp')
        if int(top_k_relation) <= 0:
            raise ValueError('FastPerRelSourceJoinPredictor requires top_k_relation > 0')

        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.device = device
        self.c_storage = c_storage
        self.shared_w = shared_w
        self.shared_w_code = _shared_w_code(shared_w)
        self.top_share = int(top_share)
        self.decay_level = decay_level
        self.per_rel_use_mtrans = bool(per_rel_use_mtrans)
        self.ppr_k = int(ppr_k)
        self.top_k_relation = int(top_k_relation)
        self.ppr_alpha = float(ppr_alpha)
        self.ppr_beta = float(ppr_beta)
        self.max_events_in_single_batch = max(1, int(max_events_in_single_batch))
        self.source_join_log_batches = bool(source_join_log_batches)

        self.M_trans = None
        self.M_sim = None
        self.M_trans_np = None
        self.M_sim_np = None
        self.M_sim_dirty = True

        self.entry_keys = np.empty((self.num_nodes, self.ppr_k), dtype=np.int32)
        self.rel_keys = np.empty((self.num_nodes, self.ppr_k, self.top_k_relation), dtype=np.int32)
        self.rel_scores = np.zeros((self.num_nodes, self.ppr_k, self.top_k_relation), dtype=np.float32)
        self.entry_lens = np.zeros(self.num_nodes, dtype=np.int32)
        self.rel_lens = np.zeros((self.num_nodes, self.ppr_k), dtype=np.int32)
        self.ppr_norms = np.zeros(self.num_nodes, dtype=np.float32)

        self._pred_stamp = np.zeros(self.num_nodes, dtype=np.int64)
        self._pred_pos = np.zeros(self.num_nodes, dtype=np.int32)
        self._pred_epoch = np.int64(1)
        self.source_join_stats = {
            'batches': 0,
            'events': 0,
            'sources': 0,
            'kernel_time_s': 0.0,
            'commit_time_s': 0.0,
            'max_batch_events': 0,
            'max_batch_sources': 0,
            'max_group_events': 0,
            'max_output_mem_mb': 0.0,
            'max_scratch_est_mb': 0.0,
        }

    @property
    def needs_m_trans(self):
        return self.per_rel_use_mtrans

    def sync_M_sim(self, M):
        self.M_sim = M
        if isinstance(M, torch.Tensor):
            self.M_sim_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_sim_np = np.ascontiguousarray(M, dtype=np.float32)
        self.M_sim_dirty = False

    def sync_M_trans(self, M):
        self.M_trans = M
        if isinstance(M, torch.Tensor):
            self.M_trans_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_trans_np = np.ascontiguousarray(M, dtype=np.float32)

    def mark_M_sim_dirty(self):
        self.M_sim_dirty = True

    def ensure_M_sim(self, semantic_updater):
        if self.M_sim_dirty or self.M_sim_np is None:
            self.sync_M_sim(semantic_updater.get_probability_M_sim())

    @staticmethod
    def _max_rss_mb():
        return FastTagMaxSourceJoinPredictor._max_rss_mb()

    def _ensure_nodes_initialized(self, node_ids):
        if node_ids.size == 0:
            return
        unique_nodes = np.unique(np.ascontiguousarray(node_ids, dtype=np.int32))
        missing = unique_nodes[self.entry_lens[unique_nodes] == 0]
        if missing.size == 0:
            return
        self.entry_keys[missing, 0] = missing
        self.rel_keys[missing, 0, 0] = self.R_SELF
        self.rel_scores[missing, 0, 0] = 1.0
        self.entry_lens[missing] = 1
        self.rel_lens[missing, 0] = 1
        self.ppr_norms[missing] = 0.0

    def _make_source_batches(self, order, groups):
        return FastTagMaxSourceJoinPredictor._make_source_batches(self, order, groups)

    def _materialize_batch(self, batch_chunks, groups):
        return FastTagMaxSourceJoinPredictor._materialize_batch(self, batch_chunks, groups)

    def update_state(self, events, t_norm=None):
        if self.per_rel_use_mtrans and self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastPerRelSourceJoinPredictor.update_state')
        if self.M_trans_np is None:
            self.M_trans_np = np.ones((self.num_rels, self.num_rels), dtype=np.float32)

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))
        if not order:
            return

        batches = self._make_source_batches(order, groups)
        threads = get_num_threads() if get_num_threads is not None else 1
        t_label = 'n/a' if t_norm is None else str(int(t_norm))

        for batch_idx, batch_chunks in enumerate(batches, start=1):
            source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events = (
                self._materialize_batch(batch_chunks, groups)
            )
            self._ensure_nodes_initialized(np.concatenate((source_ids, flat_targets)))

            n_groups = len(batch_chunks)
            out_entry_keys = np.empty((n_groups, self.ppr_k), dtype=np.int32)
            out_rel_keys = np.empty((n_groups, self.ppr_k, self.top_k_relation), dtype=np.int32)
            out_rel_scores = np.zeros((n_groups, self.ppr_k, self.top_k_relation), dtype=np.float32)
            out_entry_lens = np.empty(n_groups, dtype=np.int32)
            out_rel_lens = np.zeros((n_groups, self.ppr_k), dtype=np.int32)
            out_norms = np.empty(n_groups, dtype=np.float32)

            active_workers = max(1, min(int(n_groups), int(threads)))
            cand_cap = self.ppr_k * self.top_k_relation + self.ppr_k * int(max_group_events) + 2
            output_mem_mb = (
                out_entry_keys.nbytes + out_rel_keys.nbytes + out_rel_scores.nbytes +
                out_entry_lens.nbytes + out_rel_lens.nbytes + out_norms.nbytes
            ) / (1024.0 ** 2)
            scratch_est_mb = cand_cap * 32.0 * active_workers / (1024.0 ** 2)

            t0 = time.time()
            _fast_perrel_update_source_join_batch(
                self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
                self.rel_lens, self.ppr_norms, self.M_trans_np, source_ids,
                group_offsets, flat_targets, flat_rels, self.ppr_k,
                self.top_k_relation, self.num_rels, self.ppr_alpha, self.ppr_beta,
                self.per_rel_use_mtrans, out_entry_keys, out_rel_keys, out_rel_scores,
                out_entry_lens, out_rel_lens, out_norms,
            )
            kernel_s = time.time() - t0

            t1 = time.time()
            _fast_perrel_commit_source_join(
                self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
                self.rel_lens, self.ppr_norms, source_ids, out_entry_keys,
                out_rel_keys, out_rel_scores, out_entry_lens, out_rel_lens,
                out_norms, self.top_k_relation,
            )
            commit_s = time.time() - t1

            st = self.source_join_stats
            st['batches'] += 1
            st['events'] += int(total_events)
            st['sources'] += int(n_groups)
            st['kernel_time_s'] += float(kernel_s)
            st['commit_time_s'] += float(commit_s)
            st['max_batch_events'] = max(st['max_batch_events'], int(total_events))
            st['max_batch_sources'] = max(st['max_batch_sources'], int(n_groups))
            st['max_group_events'] = max(st['max_group_events'], int(max_group_events))
            st['max_output_mem_mb'] = max(st['max_output_mem_mb'], float(output_mem_mb))
            st['max_scratch_est_mb'] = max(st['max_scratch_est_mb'], float(scratch_est_mb))

            if self.source_join_log_batches:
                rss_mb = self._max_rss_mb()
                rss_text = 'n/a' if rss_mb is None else f'{rss_mb:.1f}MB'
                print(
                    f'[C-v5-source-join] mode=per_rel t_norm={t_label} '
                    f'batch={batch_idx}/{len(batches)} events={int(total_events)} '
                    f'sources={n_groups} max_group_events={int(max_group_events)} '
                    f'threads={int(threads)} out_mem={output_mem_mb:.1f}MB '
                    f'scratch_est={scratch_est_mb:.1f}MB max_rss={rss_text} '
                    f'kernel={kernel_s:.4f}s commit={commit_s:.4f}s',
                    flush=True,
                )

    def predict_batch(self, batch_data, neg_samples_arr, gamma_list):
        if self.M_sim_np is None:
            raise RuntimeError('M_sim must be synced before prediction')
        batch_i64 = np.ascontiguousarray(batch_data.astype(np.int64, copy=False))
        neg_i64 = np.ascontiguousarray(neg_samples_arr.astype(np.int64, copy=False))
        need_shared = any(abs(float(g)) > 1e-12 for g in gamma_list)
        pos_direct = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_direct = np.zeros(neg_i64.shape, dtype=np.float32)

        if not need_shared:
            self._pred_epoch = _fast_perrel_predict_batch(
                self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
                self.rel_lens, self.M_sim_np, batch_i64, neg_i64, pos_direct, neg_direct,
                self._pred_stamp, self._pred_pos, self._pred_epoch,
            )
            return [(pos_direct, neg_direct) for _ in gamma_list]

        pos_shared = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_shared = np.zeros(neg_i64.shape, dtype=np.float32)
        _fast_perrel_predict_parts_batch(
            self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
            self.rel_lens, self.M_sim_np, batch_i64, neg_i64, self.shared_w_code,
            self.top_share, pos_direct, neg_direct, pos_shared, neg_shared,
        )
        return [
            (pos_direct + float(g) * pos_shared, neg_direct + float(g) * neg_shared)
            for g in gamma_list
        ]

    def get_source_join_stats(self):
        return dict(self.source_join_stats)



class CPredictor:
    R_SELF = -1

    def __init__(self, num_nodes, num_rels, *,
                 c_storage='per_rel', shared_w='dual_msim',
                 decay_level='timestamp', per_rel_use_mtrans=False,
                 ppr_k=500, top_k_relation=8,
                 ppr_alpha=0.1, ppr_beta=0.8, device='cuda:0',
                 top_share=-1):
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.device = device

        self.c_storage = c_storage
        self.shared_w = shared_w
        self.top_share = int(top_share)

        self.decay_level = decay_level
        self.per_rel_use_mtrans = per_rel_use_mtrans

        self.ppr_k = ppr_k
        self.top_k_relation = top_k_relation
        self.ppr_alpha = ppr_alpha
        self.ppr_beta = ppr_beta

        self.M_trans = None
        self.M_sim = None
        self.M_trans_np = None
        self.M_sim_np = None
        self.M_sim_dirty = True

        self.ppr_norms = np.zeros(num_nodes, dtype=np.float32)
        if c_storage in ('tag_max', 'tag_sum'):
            self.ppr_scores = [{} for _ in range(num_nodes)]
            self.ppr_tags = [{} for _ in range(num_nodes)]
            if c_storage == 'tag_sum':
                self.ppr_tag_scores = [{} for _ in range(num_nodes)]
        elif c_storage == 'per_rel':
            self.ppr_breakdown = [{} for _ in range(num_nodes)]
        else:
            raise ValueError(c_storage)

    @property
    def needs_m_trans(self):
        """是否需要 M_trans (用于决定是否跑 LogicMatrixUpdater)."""
        return self.c_storage in ('tag_max', 'tag_sum') or self.per_rel_use_mtrans

    def sync_M_sim(self, M):
        self.M_sim = M
        if isinstance(M, torch.Tensor):
            self.M_sim_np = M.detach().cpu().numpy()
        else:
            self.M_sim_np = np.asarray(M, dtype=np.float32)
        self.M_sim_dirty = False

    def sync_M_trans(self, M):
        self.M_trans = M
        if isinstance(M, torch.Tensor):
            self.M_trans_np = M.detach().cpu().numpy()
        else:
            self.M_trans_np = np.asarray(M, dtype=np.float32)

    def mark_M_sim_dirty(self):
        self.M_sim_dirty = True

    def ensure_M_sim(self, semantic_updater):
        if self.M_sim_dirty or self.M_sim_np is None:
            self.sync_M_sim(semantic_updater.get_probability_M_sim())

    def _tag_share_keys(self, score_dict):
        if self.top_share <= 0 or len(score_dict) <= self.top_share:
            return score_dict.keys()
        return set(k for k, _ in heapq.nlargest(self.top_share, score_dict.items(), key=lambda x: x[1]))

    def _perrel_share_keys(self, breakdown):
        if self.top_share <= 0 or len(breakdown) <= self.top_share:
            return breakdown.keys()
        return set(k for k, _ in heapq.nlargest(
            self.top_share, breakdown.items(), key=lambda x: sum(x[1].values())
        ))


    def update_state(self, events):
        """events: (N, 3) ndarray (s, r, o). 按 decay_level 选择合批粒度."""
        if self.decay_level == 'timestamp':
            by_source = defaultdict(list)
            for i in range(len(events)):
                by_source[int(events[i, 0])].append(
                    (int(events[i, 2]), int(events[i, 1]))
                )
            for source, src_events in by_source.items():
                self._update_batched(source, src_events)
        else:  # event-level: 每个事件单独走 K=1 的 batched
            for i in range(len(events)):
                self._update_batched(
                    int(events[i, 0]),
                    [(int(events[i, 2]), int(events[i, 1]))],
                )

    def _update_batched(self, source, src_events):
        """同 source 的 K 个事件合并为一次更新.

        Z_new = β·Z_old + β (与 K 无关, 一个时间戳一次 β 衰减)
        target_prob_s2_total = (1-α)·β/Z_new, K 个事件平分.
        """
        if self.c_storage == 'tag_max':
            self._update_tag_max(source, src_events)
        elif self.c_storage == 'tag_sum':
            self._update_tag_sum(source, src_events)
        else:
            self._update_per_rel(source, src_events)

    def _ppr_coeffs(self, source):
        """返回 (target_prob_s1, target_prob_s2_total, target_prob_restart, new_norm)."""
        last_norm = float(self.ppr_norms[source])
        new_norm = last_norm * self.ppr_beta + self.ppr_beta
        s1 = (1.0 - self.ppr_alpha) * (last_norm * self.ppr_beta) / new_norm
        s2 = (1.0 - self.ppr_alpha) * self.ppr_beta / new_norm
        return s1, s2, self.ppr_alpha, new_norm


    def _ensure_tag_max_init(self, node):
        if not self.ppr_scores[node]:
            self.ppr_scores[node][node] = 1.0
            self.ppr_tags[node][node] = self.R_SELF
            self.ppr_norms[node] = 0.0

    def _update_tag_max(self, source, src_events):
        K = len(src_events)
        self._ensure_tag_max_init(source)
        for target, _ in src_events:
            self._ensure_tag_max_init(target)

        s1, s2_total, restart, new_norm = self._ppr_coeffs(source)
        s2_per_event = s2_total / K
        M_trans = self.M_trans_np if self.M_trans_np is not None else self.M_trans

        s_scores = self.ppr_scores[source]
        s_tags = self.ppr_tags[source]

        temp_scores = {k: w * s1 for k, w in s_scores.items()}
        temp_tags = dict(s_tags)

        for target, r_update in src_events:
            o_scores = self.ppr_scores[target]
            o_tags = self.ppr_tags[target]

            candidates = []
            sum_logic_weighted = 0.0
            for k, w_raw in o_scores.items():
                r_k = o_tags[k]
                factor = 1.0 if r_k == self.R_SELF else float(M_trans[r_update, r_k])
                c = w_raw * factor
                if c > 0:
                    candidates.append((k, c))
                    sum_logic_weighted += c
            if sum_logic_weighted <= 0:
                continue

            scale = s2_per_event / sum_logic_weighted
            for k, c in candidates:
                contrib = c * scale
                if k not in temp_scores or contrib > temp_scores[k]:
                    temp_scores[k] = contrib
                    temp_tags[k] = r_update

        temp_scores[source] = temp_scores.get(source, 0.0) + restart
        temp_tags[source] = self.R_SELF

        if len(temp_scores) > self.ppr_k:
            top = heapq.nlargest(self.ppr_k, temp_scores.items(), key=lambda x: x[1])
            sum_top = sum(v for _, v in top) or 1.0
            self.ppr_scores[source] = {k: v / sum_top for k, v in top}
            self.ppr_tags[source] = {k: temp_tags[k] for k, _ in top}
        else:
            self.ppr_scores[source] = temp_scores
            self.ppr_tags[source] = temp_tags

        self.ppr_norms[source] = new_norm


    def _ensure_tag_sum_init(self, node):
        if not self.ppr_scores[node]:
            self.ppr_scores[node][node] = 1.0
            self.ppr_tags[node][node] = self.R_SELF
            self.ppr_tag_scores[node][node] = 1.0
            self.ppr_norms[node] = 0.0

    def _update_tag_sum(self, source, src_events):
        K = len(src_events)
        self._ensure_tag_sum_init(source)
        for target, _ in src_events:
            self._ensure_tag_sum_init(target)

        s1, s2_total, restart, new_norm = self._ppr_coeffs(source)
        s2_per_event = s2_total / K
        M_trans = self.M_trans_np if self.M_trans_np is not None else self.M_trans

        s_scores = self.ppr_scores[source]            # sum
        s_tags = self.ppr_tags[source]
        s_tag_scores = self.ppr_tag_scores[source]    # 单次最大贡献

        temp_scores = {k: w * s1 for k, w in s_scores.items()}
        temp_tag_scores = {k: w * s1 for k, w in s_tag_scores.items()}
        temp_tags = dict(s_tags)

        for target, r_update in src_events:
            o_scores = self.ppr_scores[target]   # 拉的是 target 的 sum
            o_tags = self.ppr_tags[target]

            candidates = []
            sum_logic_weighted = 0.0
            for k, w_raw in o_scores.items():
                r_k = o_tags[k]
                factor = 1.0 if r_k == self.R_SELF else float(M_trans[r_update, r_k])
                c = w_raw * factor
                if c > 0:
                    candidates.append((k, c))
                    sum_logic_weighted += c
            if sum_logic_weighted <= 0:
                continue

            scale = s2_per_event / sum_logic_weighted
            for k, c in candidates:
                contrib = c * scale
                temp_scores[k] = temp_scores.get(k, 0.0) + contrib   # sum
                if k == source:
                    continue   # source 的 tag 始终 R_SELF (在 restart 步处理)
                if k not in temp_tag_scores or contrib > temp_tag_scores[k]:
                    temp_tags[k] = r_update
                    temp_tag_scores[k] = contrib

        temp_scores[source] = temp_scores.get(source, 0.0) + restart
        temp_tags[source] = self.R_SELF
        temp_tag_scores[source] = temp_tag_scores.get(source, 0.0) + restart

        if len(temp_scores) > self.ppr_k:
            top = heapq.nlargest(self.ppr_k, temp_scores.items(), key=lambda x: x[1])
            sum_top = sum(v for _, v in top) or 1.0
            inv = 1.0 / sum_top
            kept = [k for k, _ in top]
            self.ppr_scores[source] = {k: temp_scores[k] * inv for k in kept}
            self.ppr_tags[source] = {k: temp_tags[k] for k in kept}
            self.ppr_tag_scores[source] = {k: temp_tag_scores[k] * inv for k in kept}
        else:
            self.ppr_scores[source] = temp_scores
            self.ppr_tags[source] = temp_tags
            self.ppr_tag_scores[source] = temp_tag_scores

        self.ppr_norms[source] = new_norm


    def _ensure_per_rel_init(self, node):
        if not self.ppr_breakdown[node]:
            self.ppr_breakdown[node] = {node: {self.R_SELF: 1.0}}
            self.ppr_norms[node] = 0.0

    def _update_per_rel(self, source, src_events):
        K = len(src_events)
        self._ensure_per_rel_init(source)
        for target, _ in src_events:
            self._ensure_per_rel_init(target)

        s1, s2_total, restart, new_norm = self._ppr_coeffs(source)
        s2_per_event = s2_total / K
        M_trans = self.M_trans_np if self.M_trans_np is not None else self.M_trans

        s_bd = self.ppr_breakdown[source]

        new_bd = {entry: {r: w * s1 for r, w in rd.items()}
                  for entry, rd in s_bd.items()}

        for target, r_update in src_events:
            t_bd = self.ppr_breakdown[target]

            if self.per_rel_use_mtrans:
                weighted = {}
                total_w = 0.0
                for entry, t_rd in t_bd.items():
                    m = sum(
                        w * (1.0 if r == self.R_SELF else float(M_trans[r_update, r]))
                        for r, w in t_rd.items()
                    )
                    if m > 0:
                        weighted[entry] = m
                        total_w += m
            else:
                weighted = {entry: sum(t_rd.values()) for entry, t_rd in t_bd.items()}
                total_w = sum(weighted.values())

            if total_w <= 0:
                continue
            scale = s2_per_event / total_w
            for entry, m in weighted.items():
                contrib = m * scale
                if entry not in new_bd:
                    new_bd[entry] = {}
                new_bd[entry][r_update] = new_bd[entry].get(r_update, 0.0) + contrib

        if source not in new_bd:
            new_bd[source] = {}
        new_bd[source][self.R_SELF] = new_bd[source].get(self.R_SELF, 0.0) + restart

        if self.top_k_relation > 0:
            for entry in list(new_bd.keys()):
                rd = new_bd[entry]
                if len(rd) > self.top_k_relation:
                    top_rel = heapq.nlargest(self.top_k_relation, rd.items(), key=lambda x: x[1])
                    new_bd[entry] = dict(top_rel)

        if len(new_bd) > self.ppr_k:
            totals = [(e, sum(rd.values())) for e, rd in new_bd.items()]
            top = heapq.nlargest(self.ppr_k, totals, key=lambda x: x[1])
            kept = set(e for e, _ in top)
            new_bd = {e: new_bd[e] for e in kept}

        total = sum(sum(rd.values()) for rd in new_bd.values())
        if total > 0:
            inv = 1.0 / total
            for entry in new_bd:
                new_bd[entry] = {r: w * inv for r, w in new_bd[entry].items()}
        else:
            new_bd = {source: {self.R_SELF: 1.0}}

        self.ppr_breakdown[source] = new_bd
        self.ppr_norms[source] = new_norm


    def predict_batch(self, batch_data, neg_samples_arr, gamma_list):
        """返回 [(pos, neg), ...] 每个 gamma 一份."""
        need_shared = any(abs(float(g)) > 1e-12 for g in gamma_list)
        d_pos, d_neg, s_pos, s_neg = self._predict_parts(batch_data, neg_samples_arr, need_shared)
        return [(d_pos + g * s_pos, d_neg + g * s_neg) for g in gamma_list]

    def _predict_parts(self, batch_data, neg_samples_arr, need_shared):
        if self.c_storage in ('tag_max', 'tag_sum'):
            return self._predict_tag_parts(batch_data, neg_samples_arr, need_shared)
        return self._predict_per_rel_parts(batch_data, neg_samples_arr, need_shared)


    def _predict_tag_parts(self, batch_data, neg_samples_arr, need_shared):
        M_sim = self.M_sim_np if self.M_sim_np is not None else self.M_sim.cpu().numpy()
        orig_rels = self.num_rels // 2

        B = len(batch_data)
        N = neg_samples_arr.shape[1]
        d_pos = np.zeros((B, 1), dtype=np.float32)
        d_neg = np.zeros((B, N), dtype=np.float32)
        s_pos = np.zeros((B, 1), dtype=np.float32)
        s_neg = np.zeros((B, N), dtype=np.float32)

        for i in range(B):
            s, r, o = int(batch_data[i, 0]), int(batch_data[i, 1]), int(batch_data[i, 2])
            r_inv = (r + orig_rels) % self.num_rels
            s_scores = self.ppr_scores[s]
            s_tags = self.ppr_tags[s]
            s_share_keys = self._tag_share_keys(s_scores) if need_shared else None

            def direct(target):
                if target not in s_scores:
                    return 0.0
                tag = s_tags[target]
                w = 1.0 if tag == self.R_SELF else float(M_sim[tag, r])
                return s_scores[target] * w

            def shared(target):
                t_scores = self.ppr_scores[target]
                t_tags = self.ppr_tags[target]
                common = s_share_keys & self._tag_share_keys(t_scores)
                if not common:
                    return 0.0
                tot = 0.0
                if self.shared_w == 'unweighted':
                    for z in common:
                        tot += s_scores[z] * t_scores[z]
                elif self.shared_w == 'dual_msim':
                    for z in common:
                        ws = 1.0 if s_tags[z] == self.R_SELF else float(M_sim[s_tags[z], r])
                        wo = 1.0 if t_tags[z] == self.R_SELF else float(M_sim[t_tags[z], r_inv])
                        tot += s_scores[z] * t_scores[z] * ws * wo
                elif self.shared_w == 'cross_msim':
                    for z in common:
                        if s_tags[z] == self.R_SELF or t_tags[z] == self.R_SELF:
                            wm = 1.0
                        else:
                            wm = float(M_sim[s_tags[z], t_tags[z]])
                        tot += s_scores[z] * t_scores[z] * wm
                return tot

            d_pos[i, 0] = direct(o)
            if need_shared:
                s_pos[i, 0] = shared(o)
            for j in np.flatnonzero(neg_samples_arr[i] != -1):
                ng = int(neg_samples_arr[i, j])
                d_neg[i, j] = direct(ng)
                if need_shared:
                    s_neg[i, j] = shared(ng)
        return d_pos, d_neg, s_pos, s_neg


    def _predict_per_rel_parts(self, batch_data, neg_samples_arr, need_shared):
        M_sim = self.M_sim_np if self.M_sim_np is not None else self.M_sim.cpu().numpy()
        orig_rels = self.num_rels // 2

        B = len(batch_data)
        N = neg_samples_arr.shape[1]
        d_pos = np.zeros((B, 1), dtype=np.float32)
        d_neg = np.zeros((B, N), dtype=np.float32)
        s_pos = np.zeros((B, 1), dtype=np.float32)
        s_neg = np.zeros((B, N), dtype=np.float32)

        def entry_w(rd, q_r):
            """sum_r rd[r] * M_sim[r, q_r]."""
            tot = 0.0
            for r, w in rd.items():
                tot += w * (1.0 if r == self.R_SELF else float(M_sim[r, q_r]))
            return tot

        for i in range(B):
            s, r, o = int(batch_data[i, 0]), int(batch_data[i, 1]), int(batch_data[i, 2])
            r_inv = (r + orig_rels) % self.num_rels
            s_bd = self.ppr_breakdown[s]
            if not s_bd:
                continue
            s_share_keys = self._perrel_share_keys(s_bd) if need_shared else None

            s_direct_w = {z: entry_w(rd, r) for z, rd in s_bd.items()}
            s_total_w = None
            if need_shared and self.shared_w == 'unweighted':
                s_total_w = {z: sum(rd.values()) for z, rd in s_bd.items()}

            def shared(target):
                t_bd = self.ppr_breakdown[target]
                if not t_bd:
                    return 0.0
                common = s_share_keys & self._perrel_share_keys(t_bd)
                if not common:
                    return 0.0
                tot = 0.0
                if self.shared_w == 'unweighted':
                    for z in common:
                        tot += s_total_w[z] * sum(t_bd[z].values())
                elif self.shared_w == 'dual_msim':
                    for z in common:
                        tot += s_direct_w[z] * entry_w(t_bd[z], r_inv)
                elif self.shared_w == 'cross_msim':
                    for z in common:
                        for r_s, w_s in s_bd[z].items():
                            for r_o, w_o in t_bd[z].items():
                                if r_s == self.R_SELF or r_o == self.R_SELF:
                                    wm = 1.0
                                else:
                                    wm = float(M_sim[r_s, r_o])
                                tot += w_s * w_o * wm
                return tot

            d_pos[i, 0] = s_direct_w.get(o, 0.0)
            if need_shared:
                s_pos[i, 0] = shared(o)
            for j in np.flatnonzero(neg_samples_arr[i] != -1):
                ng = int(neg_samples_arr[i, j])
                d_neg[i, j] = s_direct_w.get(ng, 0.0)
                if need_shared:
                    s_neg[i, j] = shared(ng)
        return d_pos, d_neg, s_pos, s_neg


class FastTagMaxArrayCPredictor:
    R_SELF = -1

    def __init__(self, num_nodes, num_rels, *,
                 c_storage='tag_max', shared_w='dual_msim',
                 decay_level='timestamp', per_rel_use_mtrans=False,
                 ppr_k=1000, top_k_relation=8,
                 ppr_alpha=0.1, ppr_beta=0.8, device='cuda:0',
                 top_share=-1):
        if not NUMBA_AVAILABLE:
            raise RuntimeError('FastTagMaxArrayCPredictor requires numba')
        if c_storage not in ('tag_max', 'tag_sum'):
            raise ValueError('FastTagMaxArrayCPredictor only supports c_storage=tag_max/tag_sum')
        if decay_level != 'timestamp':
            raise ValueError('FastTagMaxArrayCPredictor only supports decay_level=timestamp')

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.device = device
        self.c_storage = c_storage
        self.shared_w = shared_w
        self.shared_w_code = _shared_w_code(shared_w)
        self.top_share = int(top_share)
        self.decay_level = decay_level
        self.per_rel_use_mtrans = per_rel_use_mtrans
        self.ppr_k = int(ppr_k)
        self.top_k_relation = top_k_relation
        self.ppr_alpha = float(ppr_alpha)
        self.ppr_beta = float(ppr_beta)

        self.M_trans = None
        self.M_sim = None
        self.M_trans_np = None
        self.M_sim_np = None
        self.M_sim_dirty = True

        self.keys = np.empty((num_nodes, self.ppr_k), dtype=np.int32)
        self.scores = np.empty((num_nodes, self.ppr_k), dtype=np.float32)
        self.tags = np.empty((num_nodes, self.ppr_k), dtype=np.int32)
        self.lens = np.zeros(num_nodes, dtype=np.int32)
        self.ppr_norms = np.zeros(num_nodes, dtype=np.float32)

        self._stamp = np.zeros(num_nodes, dtype=np.int64)
        self._pos = np.zeros(num_nodes, dtype=np.int32)
        self._epoch = np.int64(1)
        self._pred_stamp = np.zeros(num_nodes, dtype=np.int64)
        self._pred_pos = np.zeros(num_nodes, dtype=np.int32)
        self._pred_epoch = np.int64(1)

        self._temp_capacity = 0
        self._temp_keys = np.empty(1, dtype=np.int32)
        self._temp_scores = np.empty(1, dtype=np.float32)
        self._temp_tags = np.empty(1, dtype=np.int32)

    @property
    def needs_m_trans(self):
        return True

    def sync_M_sim(self, M):
        self.M_sim = M
        if isinstance(M, torch.Tensor):
            self.M_sim_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_sim_np = np.ascontiguousarray(M, dtype=np.float32)
        self.M_sim_dirty = False

    def sync_M_trans(self, M):
        self.M_trans = M
        if isinstance(M, torch.Tensor):
            self.M_trans_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_trans_np = np.ascontiguousarray(M, dtype=np.float32)

    def mark_M_sim_dirty(self):
        self.M_sim_dirty = True

    def ensure_M_sim(self, semantic_updater):
        if self.M_sim_dirty or self.M_sim_np is None:
            self.sync_M_sim(semantic_updater.get_probability_M_sim())

    def _ensure_temp_capacity(self, num_events):
        capacity = self.ppr_k * (int(num_events) + 1) + 1
        if capacity <= self._temp_capacity:
            return
        self._temp_capacity = capacity
        self._temp_keys = np.empty(capacity, dtype=np.int32)
        self._temp_scores = np.empty(capacity, dtype=np.float32)
        self._temp_tags = np.empty(capacity, dtype=np.int32)

    def update_state(self, events):
        if self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastTagMaxArrayCPredictor.update_state')

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))

        max_events = 1
        for s in order:
            n = len(groups[s][0])
            if n > max_events:
                max_events = n
        self._ensure_temp_capacity(max_events)

        for source in order:
            target_list, rel_list = groups[source]
            targets = np.asarray(target_list, dtype=np.int32)
            rels = np.asarray(rel_list, dtype=np.int32)
            if self._epoch > 9000000000000000000:
                self._stamp.fill(0)
                self._epoch = np.int64(1)
            _fast_tagmax_update_one(
                self.keys, self.scores, self.tags, self.lens, self.ppr_norms,
                self.M_trans_np, int(source), targets, rels,
                self.ppr_k, self.ppr_alpha, self.ppr_beta,
                self._temp_keys, self._temp_scores, self._temp_tags,
                self._stamp, self._pos, self._epoch,
            )
            self._epoch += 1

    def predict_batch(self, batch_data, neg_samples_arr, gamma_list):
        if self.M_sim_np is None:
            raise RuntimeError('M_sim must be synced before prediction')

        batch_i64 = np.ascontiguousarray(batch_data.astype(np.int64, copy=False))
        neg_i64 = np.ascontiguousarray(neg_samples_arr.astype(np.int64, copy=False))
        need_shared = any(abs(float(g)) > 1e-12 for g in gamma_list)
        pos_direct = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_direct = np.zeros(neg_i64.shape, dtype=np.float32)

        if not need_shared:
            self._pred_epoch = _fast_tagmax_predict_batch(
                self.keys, self.scores, self.tags, self.lens, self.M_sim_np,
                batch_i64, neg_i64, pos_direct, neg_direct,
                self._pred_stamp, self._pred_pos, self._pred_epoch,
            )
            return [(pos_direct, neg_direct) for _ in gamma_list]

        pos_shared = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_shared = np.zeros(neg_i64.shape, dtype=np.float32)
        _fast_tag_predict_parts_batch(
            self.keys, self.scores, self.tags, self.lens, self.M_sim_np,
            batch_i64, neg_i64, self.shared_w_code, self.top_share,
            pos_direct, neg_direct, pos_shared, neg_shared,
        )
        return [
            (pos_direct + float(g) * pos_shared, neg_direct + float(g) * neg_shared)
            for g in gamma_list
        ]


class FastTagMaxSourceJoinPredictor(FastTagMaxArrayCPredictor):
    def __init__(self, *args, max_events_in_single_batch=20000,
                 source_join_log_batches=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_events_in_single_batch = max(1, int(max_events_in_single_batch))
        self.source_join_log_batches = bool(source_join_log_batches)
        self.source_join_stats = {
            'batches': 0,
            'events': 0,
            'sources': 0,
            'kernel_time_s': 0.0,
            'commit_time_s': 0.0,
            'max_batch_events': 0,
            'max_batch_sources': 0,
            'max_group_events': 0,
            'max_output_mem_mb': 0.0,
            'max_scratch_est_mb': 0.0,
        }

    def _ensure_nodes_initialized(self, node_ids):
        if node_ids.size == 0:
            return
        unique_nodes = np.unique(np.ascontiguousarray(node_ids, dtype=np.int32))
        missing = unique_nodes[self.lens[unique_nodes] == 0]
        if missing.size == 0:
            return
        self.keys[missing, 0] = missing
        self.scores[missing, 0] = 1.0
        self.tags[missing, 0] = -1
        self.lens[missing] = 1
        self.ppr_norms[missing] = 0.0

    @staticmethod
    def _max_rss_mb():
        try:
            import resource
        except Exception:
            return None
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == 'darwin':
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0

    def _make_source_batches(self, order, groups):
        batches = []
        current = []
        cur_events = 0
        limit = self.max_events_in_single_batch
        for source in order:
            n_total = len(groups[source][0])
            start = 0
            while start < n_total:
                end = min(start + limit, n_total)
                n = end - start
                if cur_events > 0 and cur_events + n > limit:
                    batches.append(current)
                    current = []
                    cur_events = 0
                current.append((source, start, end))
                cur_events += n
                start = end
                if start < n_total:
                    batches.append(current)
                    current = []
                    cur_events = 0
        if current:
            batches.append(current)
        return batches

    def _materialize_batch(self, batch_chunks, groups):
        source_ids = np.asarray([chunk[0] for chunk in batch_chunks], dtype=np.int32)
        group_offsets = np.empty(len(batch_chunks) + 1, dtype=np.int32)
        group_offsets[0] = 0
        total_events = 0
        max_group_events = 0
        for i, (_, start, end) in enumerate(batch_chunks):
            n = end - start
            total_events += n
            if n > max_group_events:
                max_group_events = n
            group_offsets[i + 1] = total_events

        flat_targets = np.empty(total_events, dtype=np.int32)
        flat_rels = np.empty(total_events, dtype=np.int32)
        pos = 0
        for source, start, end in batch_chunks:
            target_list, rel_list = groups[source]
            n = end - start
            flat_targets[pos:pos + n] = np.asarray(target_list[start:end], dtype=np.int32)
            flat_rels[pos:pos + n] = np.asarray(rel_list[start:end], dtype=np.int32)
            pos += n
        return source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events

    def update_state(self, events, t_norm=None):
        if self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastTagMaxSourceJoinPredictor.update_state')

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))
        if not order:
            return

        batches = self._make_source_batches(order, groups)
        threads = get_num_threads() if get_num_threads is not None else 1
        t_label = 'n/a' if t_norm is None else str(int(t_norm))

        for batch_idx, batch_sources in enumerate(batches, start=1):
            source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events = (
                self._materialize_batch(batch_sources, groups)
            )
            self._ensure_nodes_initialized(np.concatenate((source_ids, flat_targets)))

            out_keys = np.empty((len(batch_sources), self.ppr_k), dtype=np.int32)
            out_scores = np.empty((len(batch_sources), self.ppr_k), dtype=np.float32)
            out_tags = np.empty((len(batch_sources), self.ppr_k), dtype=np.int32)
            out_lens = np.empty(len(batch_sources), dtype=np.int32)
            out_norms = np.empty(len(batch_sources), dtype=np.float32)

            active_workers = max(1, min(int(len(batch_sources)), int(threads)))
            candidate_capacity = self.ppr_k * (int(max_group_events) + 1) + 1
            output_mem_mb = (
                out_keys.nbytes + out_scores.nbytes + out_tags.nbytes +
                out_lens.nbytes + out_norms.nbytes
            ) / (1024.0 ** 2)
            scratch_est_mb = candidate_capacity * 40.0 * active_workers / (1024.0 ** 2)

            t0 = time.time()
            _fast_tagmax_update_source_join_batch(
                self.keys, self.scores, self.tags, self.lens, self.ppr_norms,
                self.M_trans_np, source_ids, group_offsets, flat_targets, flat_rels,
                self.ppr_k, self.ppr_alpha, self.ppr_beta,
                out_keys, out_scores, out_tags, out_lens, out_norms,
            )
            kernel_s = time.time() - t0

            t1 = time.time()
            _fast_tagmax_commit_source_join(
                self.keys, self.scores, self.tags, self.lens, self.ppr_norms,
                source_ids, out_keys, out_scores, out_tags, out_lens, out_norms,
            )
            commit_s = time.time() - t1

            st = self.source_join_stats
            st['batches'] += 1
            st['events'] += int(total_events)
            st['sources'] += int(len(batch_sources))
            st['kernel_time_s'] += float(kernel_s)
            st['commit_time_s'] += float(commit_s)
            st['max_batch_events'] = max(st['max_batch_events'], int(total_events))
            st['max_batch_sources'] = max(st['max_batch_sources'], int(len(batch_sources)))
            st['max_group_events'] = max(st['max_group_events'], int(max_group_events))
            st['max_output_mem_mb'] = max(st['max_output_mem_mb'], float(output_mem_mb))
            st['max_scratch_est_mb'] = max(st['max_scratch_est_mb'], float(scratch_est_mb))

            if self.source_join_log_batches:
                rss_mb = self._max_rss_mb()
                rss_text = 'n/a' if rss_mb is None else f'{rss_mb:.1f}MB'
                print(
                    f'[C-source-join] t_norm={t_label} '
                    f'batch={batch_idx}/{len(batches)} events={int(total_events)} '
                    f'sources={len(batch_sources)} max_group_events={int(max_group_events)} '
                    f'threads={int(threads)} out_mem={output_mem_mb:.1f}MB '
                    f'scratch_est={scratch_est_mb:.1f}MB max_rss={rss_text} '
                    f'kernel={kernel_s:.4f}s commit={commit_s:.4f}s',
                    flush=True,
                )

    def get_source_join_stats(self):
        return dict(self.source_join_stats)


class FastTagSumSourceJoinPredictor(FastTagMaxSourceJoinPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.c_storage != 'tag_sum':
            raise ValueError('FastTagSumSourceJoinPredictor requires c_storage=tag_sum')
        self.tag_scores = np.empty((self.num_nodes, self.ppr_k), dtype=np.float32)

    def _ensure_nodes_initialized(self, node_ids):
        if node_ids.size == 0:
            return
        unique_nodes = np.unique(np.ascontiguousarray(node_ids, dtype=np.int32))
        missing = unique_nodes[self.lens[unique_nodes] == 0]
        if missing.size == 0:
            return
        self.keys[missing, 0] = missing
        self.scores[missing, 0] = 1.0
        self.tags[missing, 0] = -1
        self.tag_scores[missing, 0] = 1.0
        self.lens[missing] = 1
        self.ppr_norms[missing] = 0.0

    def update_state(self, events, t_norm=None):
        if self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastTagSumSourceJoinPredictor.update_state')

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))
        if not order:
            return

        batches = self._make_source_batches(order, groups)
        threads = get_num_threads() if get_num_threads is not None else 1
        t_label = 'n/a' if t_norm is None else str(int(t_norm))

        for batch_idx, batch_chunks in enumerate(batches, start=1):
            source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events = (
                self._materialize_batch(batch_chunks, groups)
            )
            self._ensure_nodes_initialized(np.concatenate((source_ids, flat_targets)))

            n_groups = len(batch_chunks)
            out_keys = np.empty((n_groups, self.ppr_k), dtype=np.int32)
            out_scores = np.empty((n_groups, self.ppr_k), dtype=np.float32)
            out_tags = np.empty((n_groups, self.ppr_k), dtype=np.int32)
            out_tag_scores = np.empty((n_groups, self.ppr_k), dtype=np.float32)
            out_lens = np.empty(n_groups, dtype=np.int32)
            out_norms = np.empty(n_groups, dtype=np.float32)

            active_workers = max(1, min(int(n_groups), int(threads)))
            candidate_capacity = self.ppr_k * (int(max_group_events) + 1) + 2
            output_mem_mb = (
                out_keys.nbytes + out_scores.nbytes + out_tags.nbytes +
                out_tag_scores.nbytes + out_lens.nbytes + out_norms.nbytes
            ) / (1024.0 ** 2)
            scratch_est_mb = candidate_capacity * 48.0 * active_workers / (1024.0 ** 2)

            t0 = time.time()
            _fast_tagsum_update_source_join_batch(
                self.keys, self.scores, self.tags, self.tag_scores, self.lens,
                self.ppr_norms, self.M_trans_np, source_ids, group_offsets,
                flat_targets, flat_rels, self.ppr_k, self.ppr_alpha, self.ppr_beta,
                out_keys, out_scores, out_tags, out_tag_scores, out_lens, out_norms,
            )
            kernel_s = time.time() - t0

            t1 = time.time()
            _fast_tagsum_commit_source_join(
                self.keys, self.scores, self.tags, self.tag_scores, self.lens,
                self.ppr_norms, source_ids, out_keys, out_scores, out_tags,
                out_tag_scores, out_lens, out_norms,
            )
            commit_s = time.time() - t1

            st = self.source_join_stats
            st['batches'] += 1
            st['events'] += int(total_events)
            st['sources'] += int(n_groups)
            st['kernel_time_s'] += float(kernel_s)
            st['commit_time_s'] += float(commit_s)
            st['max_batch_events'] = max(st['max_batch_events'], int(total_events))
            st['max_batch_sources'] = max(st['max_batch_sources'], int(n_groups))
            st['max_group_events'] = max(st['max_group_events'], int(max_group_events))
            st['max_output_mem_mb'] = max(st['max_output_mem_mb'], float(output_mem_mb))
            st['max_scratch_est_mb'] = max(st['max_scratch_est_mb'], float(scratch_est_mb))

            if self.source_join_log_batches:
                rss_mb = self._max_rss_mb()
                rss_text = 'n/a' if rss_mb is None else f'{rss_mb:.1f}MB'
                print(
                    f'[C-v5-source-join] mode=tag_sum t_norm={t_label} '
                    f'batch={batch_idx}/{len(batches)} events={int(total_events)} '
                    f'sources={n_groups} max_group_events={int(max_group_events)} '
                    f'threads={int(threads)} out_mem={output_mem_mb:.1f}MB '
                    f'scratch_est={scratch_est_mb:.1f}MB max_rss={rss_text} '
                    f'kernel={kernel_s:.4f}s commit={commit_s:.4f}s',
                    flush=True,
                )



def events_for_update(events, data, args, is_train=False):
    events_f64 = np.ascontiguousarray(events, dtype=np.float64)
    if not is_train and data['is_tgb'] and not bool(getattr(args, 'close_update_backward', False)):
        return inverse_aug(events_f64, data['num_rels_raw'], data['num_rels'])
    return events_f64


def update_runtime(predictor, events_f64, t_norm, semantic_updater, logic_updater):
    """统一推进 M 矩阵和 predictor 状态. logic_updater=None 时跳过 M_trans."""
    semantic_updater.update_M_sim_step(events_f64, t_norm)
    predictor.mark_M_sim_dirty()
    if logic_updater is not None:
        logic_updater.update_M_trans_step(events_f64, t_norm)
        logic_updater.update_history_step(events_f64, t_norm)
        predictor.sync_M_trans(logic_updater.get_normalized_M_trans())
    if hasattr(predictor, 'get_source_join_stats'):
        predictor.update_state(events_f64, t_norm=t_norm)
    else:
        predictor.update_state(events_f64)


def warm_up(predictor, snapshots, data, args, semantic_updater, logic_updater):
    total = len(snapshots)
    print(f'[C] warm-up on {total} train snapshots '
          f'(c_storage={args.c_storage}, max_events_in_single_batch={args.max_events_in_single_batch})',
          flush=True)
    t0 = time.time()
    peak_reset = False
    for idx, (events, t_norm, t_orig) in enumerate(snapshots, start=1):
        events_f64 = events_for_update(events, data, args, is_train=True)
        if not peak_reset:
            peak_reset = _reset_cuda_peak(predictor.device)
            print(
                f'[C] train peak allocated reset before first train update '
                f'(cuda_active={peak_reset})',
                flush=True,
            )
        update_runtime(predictor, events_f64, t_norm, semantic_updater, logic_updater)
        elapsed = time.time() - t0
        print(
            f'[C] train snapshot {idx}/{total} '
            f't_norm={int(t_norm)} t_orig={int(t_orig)} '
            f'events={len(events)} update_events={len(events_f64)} '
            f'elapsed={elapsed:.1f}s',
            flush=True,
        )
    train_time = time.time() - t0
    train_peak = _cuda_peak_allocated(predictor.device) if peak_reset else None
    print(f'[C] train warmup: {train_time:.1f}s', flush=True)
    print(
        f'[C] train peak allocated: {_format_bytes(train_peak)} '
        f'({0 if train_peak is None else train_peak} bytes)',
        flush=True,
    )
    return {
        'train_time_s': train_time,
        'peak_memory_allocated_bytes': train_peak,
    }


def run_inference(predictor, data, args, gamma_list, semantic_updater, logic_updater):
    train_start = data['train_predict_start_idx']
    train_warmup = data['train_list'][:train_start]
    train_predict = data['train_list'][train_start:]
    train_stats = warm_up(predictor, train_warmup, data, args, semantic_updater, logic_updater)

    neg_sampler = data['negative_sampler']
    out_dirs = [make_c_result_dir(args, gamma=g) for g in gamma_list]

    def eval_split(snapshot_list, mode, is_train=False):
        writers = [ScoreWriter(d, mode) for d in out_dirs]
        metric_sums = [{} for _ in gamma_list]
        first_batch_logged = False
        peak_reset_logged = False
        peak_reset_before_score = True
        cuda_peak_active = False
        inference_time = 0.0
        inference_pos_count = 0
        inference_neg_count = 0
        inference_peak = None
        t0 = time.time()
        total = len(snapshot_list)
        for idx, (events, t_norm, t_orig) in enumerate(snapshot_list, start=1):
            rows_this_ts = 0
            batches_this_ts = 0
            batches = collect_eval_batch(events, t_orig, neg_sampler, mode, args.batch_size)
            for batch_data, neg_arr, neg_mask in batches:
                predictor.ensure_M_sim(semantic_updater)
                if not first_batch_logged:
                    print(
                        f'[C] first {mode} batch: t_norm={int(t_norm)} '
                        f't_orig={int(t_orig)} batch={batch_data.shape} '
                        f'neg_arr={neg_arr.shape} valid_negs={int(neg_mask.sum())}',
                        flush=True,
                    )
                    first_batch_logged = True
                batches_this_ts += 1
                rows_this_ts += len(batch_data)
                if peak_reset_before_score:
                    cuda_peak_active = _reset_cuda_peak(predictor.device)
                    if not peak_reset_logged:
                        print(
                            f'[C] {mode} inference peak allocated reset before first score batch '
                            f'(cuda_active={cuda_peak_active})',
                            flush=True,
                        )
                        peak_reset_logged = True
                    peak_reset_before_score = False
                _cuda_synchronize(predictor.device)
                pred_t0 = time.perf_counter()
                scored = predictor.predict_batch(batch_data, neg_arr, gamma_list)
                _cuda_synchronize(predictor.device)
                inference_time += time.perf_counter() - pred_t0
                inference_pos_count += int(len(batch_data))
                inference_neg_count += int(neg_mask.sum())
                peak_now = _cuda_peak_allocated(predictor.device) if cuda_peak_active else None
                if peak_now is not None:
                    inference_peak = peak_now if inference_peak is None else max(inference_peak, peak_now)
                for i, (pos, neg) in enumerate(scored):
                    batch_sums = compute_ranking_metric_sums(pos, neg, neg_mask)
                    add_metric_sums(metric_sums[i], batch_sums)
                    writers[i].write_batch(pos, neg, neg_mask)
            events_f64 = events_for_update(events, data, args, is_train=is_train)
            update_runtime(predictor, events_f64, t_norm, semantic_updater, logic_updater)
            peak_reset_before_score = True
            elapsed = time.time() - t0
            mrr_so_far = ','.join(
                f'{s.get("mrr_loose", 0.0) / max(int(s.get("count", 0)), 1):.5f}' for s in metric_sums
            )
            print(
                f'[C] {mode} snapshot {idx}/{total} '
                f't_norm={int(t_norm)} t_orig={int(t_orig)} '
                f'events={len(events)} batches={batches_this_ts} rows={rows_this_ts} '
                f'update_events={len(events_f64)} mrr_so_far=[{mrr_so_far}] '
                f'elapsed={elapsed:.1f}s',
                flush=True,
            )
        for w in writers:
            w.close()
        eval_time = time.time() - t0
        inference_total_count = inference_pos_count + inference_neg_count
        avg_inference_time = inference_time / max(inference_total_count, 1)
        print(
            f'[C] {mode} inference_time={inference_time:.6f}s '
            f'positive_scores={inference_pos_count} '
            f'negative_scores={inference_neg_count} '
            f'total_scores={inference_total_count} '
            f'avg_per_score={avg_inference_time:.9f}s '
            f'({avg_inference_time * 1e6:.3f}us) '
            f'peak_allocated={_format_bytes(inference_peak)} '
            f'({0 if inference_peak is None else inference_peak} bytes)',
            flush=True,
        )
        print(f'[C] {mode} eval: {eval_time:.1f}s', flush=True)
        split_stats = {
            'inference_time_s': inference_time,
            'positive_score_count': inference_pos_count,
            'negative_score_count': inference_neg_count,
            'total_score_count': inference_total_count,
            'avg_inference_time_s_per_score': avg_inference_time,
            'peak_memory_allocated_bytes': inference_peak,
            'eval_time_s': eval_time,
        }
        return [finalize_metric_sums(s) for s in metric_sums], split_stats

    train_metrics = None
    train_predict_stats = None
    if train_predict:
        print(f'[C] predict-then-train snapshots: {len(train_predict)}', flush=True)
        train_metrics, train_predict_stats = eval_split(train_predict, 'train', is_train=True)

    val_metrics, val_stats = eval_split(data['val_list'], 'val')
    test_metrics = None
    test_stats = None
    if getattr(args, 'eval_test', True):
        test_metrics, test_stats = eval_split(data['test_list'], 'test')
    runtime_stats = {
        'train': train_stats,
        'val': val_stats,
    }
    if test_stats is not None:
        runtime_stats['test'] = test_stats
    if train_predict_stats is not None:
        runtime_stats['train_predict'] = train_predict_stats
    if hasattr(predictor, 'get_source_join_stats'):
        runtime_stats['source_join'] = predictor.get_source_join_stats()
    return train_metrics, val_metrics, test_metrics, runtime_stats



def make_c_result_dir(args, gamma):
    common = dict(
        impl='v5',
        c_storage=args.c_storage,
        shared_w=args.shared_w,
        ppr_k=args.ppr_k,
        ppr_a=args.ppr_alpha,
        ppr_b=args.ppr_beta,
        gamma=float(gamma),
        ns_q=args.ns_q,
        ns_seed=args.ns_seed,
        train_predict_ratio=args.train_predict_ratio,
        ws=args.window_semantic_sim,
        meb=args.max_events_in_single_batch,
        ts=args.top_share,
    )
    if args.dataset in TGB_DATASETS:
        common['close_update_backward'] = bool(getattr(args, 'close_update_backward', False))
    if args.c_storage in ('tag_max', 'tag_sum') or args.per_rel_use_mtrans:
        common['decay_rt'] = args.decay_rel_trans
        common['wt'] = args.window_trans
    if args.c_storage == 'per_rel':
        common['top_kr'] = args.top_k_relation
        common['prtm'] = int(args.per_rel_use_mtrans)
    return make_dir_name('results_c_single', args.dataset, args.seed, **common)


def build_runtime(args, num_nodes, num_rels, device):
    """根据 args 构造 predictor + M-矩阵更新器. 不需要 M_trans 时返回 logic_updater=None."""
    if args.c_storage == 'per_rel' and int(args.top_k_relation) <= 0:
        raise ValueError('per_rel mode requires top_k_relation > 0')

    use_fast_array = NUMBA_AVAILABLE
    if use_fast_array and int(args.source_join_threads) > 0 and set_num_threads is not None:
        set_num_threads(int(args.source_join_threads))
    threads = get_num_threads() if (use_fast_array and get_num_threads is not None) else 1

    if use_fast_array and args.c_storage == 'tag_max':
        print(
            '[C-v5] using FastTagMaxSourceJoinPredictor '
            f'(max_events_in_single_batch={int(args.max_events_in_single_batch)}, '
            f'top_share={int(args.top_share)}, threads={int(threads)}, '
            f'log_batches={bool(args.source_join_log_batches)})',
            flush=True,
        )
        predictor = FastTagMaxSourceJoinPredictor(
            num_nodes=num_nodes, num_rels=num_rels,
            c_storage=args.c_storage, shared_w=args.shared_w,
            decay_level='timestamp', per_rel_use_mtrans=args.per_rel_use_mtrans,
            ppr_k=args.ppr_k, top_k_relation=args.top_k_relation,
            ppr_alpha=args.ppr_alpha, ppr_beta=args.ppr_beta, device=device,
            top_share=args.top_share,
            max_events_in_single_batch=args.max_events_in_single_batch,
            source_join_log_batches=args.source_join_log_batches,
        )
    elif use_fast_array and args.c_storage == 'tag_sum':
        print(
            '[C-v5] using FastTagSumSourceJoinPredictor '
            f'(max_events_in_single_batch={int(args.max_events_in_single_batch)}, '
            f'top_share={int(args.top_share)}, threads={int(threads)}, '
            f'log_batches={bool(args.source_join_log_batches)})',
            flush=True,
        )
        predictor = FastTagSumSourceJoinPredictor(
            num_nodes=num_nodes, num_rels=num_rels,
            c_storage=args.c_storage, shared_w=args.shared_w,
            decay_level='timestamp', per_rel_use_mtrans=args.per_rel_use_mtrans,
            ppr_k=args.ppr_k, top_k_relation=args.top_k_relation,
            ppr_alpha=args.ppr_alpha, ppr_beta=args.ppr_beta, device=device,
            top_share=args.top_share,
            max_events_in_single_batch=args.max_events_in_single_batch,
            source_join_log_batches=args.source_join_log_batches,
        )
    elif use_fast_array and args.c_storage == 'per_rel':
        print(
            '[C-v5] using FastPerRelSourceJoinPredictor '
            f'(max_events_in_single_batch={int(args.max_events_in_single_batch)}, '
            f'top_share={int(args.top_share)}, '
            f'top_k_relation={int(args.top_k_relation)}, '
            f'per_rel_use_mtrans={bool(args.per_rel_use_mtrans)}, '
            f'threads={int(threads)}, log_batches={bool(args.source_join_log_batches)})',
            flush=True,
        )
        predictor = FastPerRelSourceJoinPredictor(
            num_nodes=num_nodes, num_rels=num_rels,
            c_storage=args.c_storage, shared_w=args.shared_w,
            decay_level='timestamp', per_rel_use_mtrans=args.per_rel_use_mtrans,
            ppr_k=args.ppr_k, top_k_relation=args.top_k_relation,
            ppr_alpha=args.ppr_alpha, ppr_beta=args.ppr_beta, device=device,
            top_share=args.top_share,
            max_events_in_single_batch=args.max_events_in_single_batch,
            source_join_log_batches=args.source_join_log_batches,
        )
    else:
        print(
            '[C-v5] fast array source-join path disabled '
            f'(numba={NUMBA_AVAILABLE}, c_storage={args.c_storage}, '
            f'top_k_relation={int(args.top_k_relation)}); falling back to CPredictor',
            flush=True,
        )
        predictor = CPredictor(
            num_nodes=num_nodes, num_rels=num_rels,
            c_storage=args.c_storage, shared_w=args.shared_w,
            decay_level='timestamp', per_rel_use_mtrans=args.per_rel_use_mtrans,
            ppr_k=args.ppr_k, top_k_relation=args.top_k_relation,
            ppr_alpha=args.ppr_alpha, ppr_beta=args.ppr_beta, device=device,
            top_share=args.top_share,
        )

    semantic_updater = SemanticMatrixUpdater(
        num_nodes=num_nodes, num_rels=num_rels,
        window_size=args.window_semantic_sim, device=device,
    )
    predictor.sync_M_sim(semantic_updater.get_probability_M_sim())

    logic_updater = None
    if predictor.needs_m_trans:
        logic_updater = LogicMatrixUpdater(
            num_nodes=num_nodes, num_rels=num_rels,
            window_size=args.window_trans,
            decay_factor=args.decay_rel_trans, device=device,
        )
        predictor.sync_M_trans(logic_updater.get_normalized_M_trans())
    else:
        print('[C] M_trans not needed, skipping LogicMatrixUpdater', flush=True)

    return predictor, semantic_updater, logic_updater


def prefix_metrics(prefix, metrics):
    return {f'{prefix}_{key}': value for key, value in metrics.items()}


def save_result(args, gamma, val_metrics, test_metrics=None, runtime_stats=None, train_metrics=None):
    cfg = vars(args).copy()
    cfg['gamma'] = float(gamma)
    cfg['score_formula'] = 'direct + gamma * shared'

    metrics = {
        'format': 'c_scores_v2',
        'val_mrr': float(val_metrics['mrr_strict']),
    }
    if test_metrics is not None:
        metrics['test_mrr'] = float(test_metrics['mrr_strict'])
    if train_metrics is not None:
        metrics.update(prefix_metrics('train', train_metrics))
    metrics.update(prefix_metrics('val', val_metrics))
    if test_metrics is not None:
        metrics.update(prefix_metrics('test', test_metrics))
    if runtime_stats is not None:
        metrics['runtime_stats'] = runtime_stats

    out_dir = make_c_result_dir(args, gamma=gamma)
    save_config(out_dir, cfg)
    save_metrics(out_dir, metrics)
    return metrics



def load_c_data(args):
    return load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )


def validate_args(args):
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f'unsupported dataset: {args.dataset}')
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError('--ns_q must be -1 or a positive integer')
    if not 0.0 <= float(args.train_predict_ratio) <= 1.0:
        raise ValueError('--train_predict_ratio must be in [0, 1]')
    if args.c_storage == 'per_rel' and int(args.top_k_relation) <= 0:
        raise ValueError('--top_k_relation must be > 0 when --c_storage per_rel')
    if int(args.ppr_k) <= 0:
        raise ValueError('--ppr_k must be > 0')
    if int(args.batch_size) <= 0:
        raise ValueError('--batch_size must be > 0')
    if int(args.max_events_in_single_batch) <= 0:
        raise ValueError('--max_events_in_single_batch must be > 0')
    if int(args.source_join_threads) < 0:
        raise ValueError('--source_join_threads must be >= 0')


def main(args):
    """单组超参数 + 单个 gamma."""
    validate_args(args)
    set_random_seed(args.seed)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.gamma = float(args.gamma)

    out_dir = make_c_result_dir(args, gamma=args.gamma)
    print(f'[C] loading dataset {args.dataset} with ns_q={args.ns_q} ns_seed={args.ns_seed}', flush=True)
    data = load_c_data(args)
    describe_loaded_data(data, prefix='[C]')

    expected_modes = ('train', 'val') if data['train_predict_count'] else ('val',)
    if getattr(args, 'eval_test', True):
        expected_modes = expected_modes + ('test',)
    if is_run_complete(out_dir, expected_modes):
        metrics = load_metrics(out_dir)
        has_required = metrics.get('format') == 'c_scores_v2' and 'val_mrr_avg' in metrics
        if getattr(args, 'eval_test', True):
            has_required = has_required and 'test_mrr_loose' in metrics and 'test_mrr_avg' in metrics
        if has_required:
            metrics['val_mrr'] = metrics['val_mrr_strict']
            if 'test_mrr_strict' in metrics:
                metrics['test_mrr'] = metrics['test_mrr_strict']
            print(f'[C] already complete: {out_dir}', flush=True)
            msg = (
                f"[C] val_mrr_loose={metrics['val_mrr_loose']:.5f} "
                f"val_mrr_strict={metrics['val_mrr_strict']:.5f} "
                f"val_mrr_avg={metrics['val_mrr_avg']:.5f}"
            )
            if 'test_mrr_strict' in metrics:
                msg += (
                    f" test_mrr_loose={metrics['test_mrr_loose']:.5f} "
                    f"test_mrr_strict={metrics['test_mrr_strict']:.5f} "
                    f"test_mrr_avg={metrics['test_mrr_avg']:.5f}"
                )
            print(msg, flush=True)
            return metrics
        print(f'[C] stale metrics found, recomputing: {out_dir}', flush=True)

    print(f'[C] single run: gamma={args.gamma:g} -> {out_dir}', flush=True)
    predictor, semantic_updater, logic_updater = build_runtime(args, data['num_nodes'], data['num_rels'], device)

    train_metrics_list, val_metrics_list, test_metrics_list, runtime_stats = run_inference(
        predictor, data, args, [args.gamma], semantic_updater, logic_updater
    )
    train_metrics = train_metrics_list[0] if train_metrics_list is not None else None
    val_metrics = val_metrics_list[0]
    test_metrics = test_metrics_list[0] if test_metrics_list is not None else None
    msg = (
        f"[C] gamma={args.gamma:g} "
        f"val_mrr_loose={val_metrics['mrr_loose']:.5f} "
        f"val_mrr_strict={val_metrics['mrr_strict']:.5f} "
        f"val_mrr_avg={val_metrics['mrr_avg']:.5f}"
    )
    if test_metrics is not None:
        msg += (
            f" test_mrr_loose={test_metrics['mrr_loose']:.5f} "
            f"test_mrr_strict={test_metrics['mrr_strict']:.5f} "
            f"test_mrr_avg={test_metrics['mrr_avg']:.5f}"
        )
    print(msg, flush=True)
    return save_result(
        args,
        args.gamma,
        val_metrics,
        test_metrics,
        runtime_stats=runtime_stats,
        train_metrics=train_metrics,
    )


def tune_c(args, all_trials=10, top_k=10, metric='val_mrr_strict'):
    """Tune C with three small Optuna studies, ranked by validation only."""
    from copy import deepcopy

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError('Optuna is required by tune_c') from exc

    fixed = {
        'shared_w': 'dual_msim',
        'per_rel_use_mtrans': False,
        'ppr_k': 1000,
        'top_k_relation': 0,
        'top_share': 100,
        'decay_rel_trans': 0.05,
        'window_semantic_sim': 5.0,
        'window_trans': 5.0,
        'max_events_in_single_batch': getattr(args, 'max_events_in_single_batch', 20000),
        'source_join_threads': getattr(args, 'source_join_threads', 0),
        'source_join_log_batches': getattr(args, 'source_join_log_batches', 0),
    }
    storages = ('tag_sum', 'tag_max') # 'per_rel'
    trials_per_storage = max(1, int(all_trials) // len(storages))
    records = []

    def make_args(storage, params):
        run_args = deepcopy(args)
        for key, value in fixed.items():
            setattr(run_args, key, value)
        for key, value in params.items():
            setattr(run_args, key, value)
        run_args.c_storage = storage
        run_args.eval_test = False
        if storage == 'per_rel':
            run_args.top_k_relation = int(params.get('top_k_relation', 100))
            run_args.per_rel_use_mtrans = bool(params.get('per_rel_use_mtrans', False))
        else:
            run_args.top_k_relation = 0
            run_args.per_rel_use_mtrans = False
        return run_args

    for storage in storages:
        sampler = optuna.samplers.TPESampler(seed=int(args.seed) + len(records) + 17)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        def objective(trial):
            params = {
                'ppr_alpha': trial.suggest_float('ppr_alpha', 0.01, 0.08, log=True),
                'ppr_beta': trial.suggest_float('ppr_beta', 0.85, 0.98),
                # 'gamma': trial.suggest_float('gamma', 1e-4, 0.1, log=True),
            }
            if storage == 'per_rel':
                params['top_k_relation'] = trial.suggest_categorical('top_k_relation', [50, 100, 200])
                params['per_rel_use_mtrans'] = trial.suggest_categorical('per_rel_use_mtrans', [False, True])
            run_args = make_args(storage, params)
            metrics = main(run_args)
            score = float(metrics[metric])
            record = {
                'rank_source': 'validation',
                'score': score,
                'metric': metric,
                'storage': storage,
                'trial': int(trial.number),
                'params': {
                    key: getattr(run_args, key)
                    for key in (
                        'c_storage',
                        'shared_w',
                        'per_rel_use_mtrans',
                        'ppr_k',
                        'top_k_relation',
                        'ppr_alpha',
                        'ppr_beta',
                        'gamma',
                        'top_share',
                        'decay_rel_trans',
                        'window_semantic_sim',
                        'window_trans',
                    )
                },
                'out_dir': make_c_result_dir(run_args, run_args.gamma),
                'args': vars(run_args).copy(),
            }
            records.append(record)
            trial.set_user_attr('record', record)
            print(
                f"[C-tune] storage={storage} trial={trial.number}/{trials_per_storage} "
                f"val {metric}={score:.5f} params={record['params']}",
                flush=True,
            )
            return score

        study.optimize(objective, n_trials=trials_per_storage)

    records.sort(key=lambda item: item['score'], reverse=True)
    return records[: int(top_k)]
