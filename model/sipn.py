import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mmcv.cnn import build_activation_layer, build_norm_layer
except ImportError:  # minimal fallback so the script runs without mmcv
    def build_norm_layer(cfg, num_features):
        cfg = dict(cfg)
        layer_type = cfg.pop('type', 'BN')
        if layer_type in ('BN', 'BN2d'):
            return 'bn', nn.BatchNorm2d(num_features, **cfg)
        if layer_type == 'BN1d':
            return 'bn', nn.BatchNorm1d(num_features, **cfg)
        raise NotImplementedError(f'Unsupported norm type: {layer_type}')

    def build_activation_layer(cfg):
        cfg = dict(cfg)
        layer_type = cfg.pop('type', 'ReLU')
        return getattr(nn, layer_type)(**cfg)

try:
    from utils import Graph
except ImportError:
    Graph = None

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

EPS = 1e-4


# =====================================================================================
# Skeleton definition (SHREC'17 / DHG-14/28, 22 joints)
# =====================================================================================
# 0 wrist, 1 palm, 2-5 thumb, 6-9 index, 10-13 middle, 14-17 ring, 18-21 pinky.
# Parent of every joint (the palm is the root and is its own parent).
SHREC_PARENTS = [1, 1, 0, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20]

MODALITIES = ('joint', 'bone', 'joint_motion', 'bone_motion')


def hand_adjacency(parents=SHREC_PARENTS):
    """Spatial-partition adjacency (identity, inward, outward), shape 3 x V x V.
    Used only when `utils.Graph` is not available."""
    V = len(parents)
    identity = np.eye(V, dtype=np.float32)
    inward = np.zeros((V, V), dtype=np.float32)
    for child, parent in enumerate(parents):
        if child != parent:
            inward[child, parent] = 1.0
    outward = inward.T.copy()

    def normalize(a):
        deg = a.sum(0)
        deg_inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
        return a * deg_inv[None, :]

    return np.stack([identity, normalize(inward), normalize(outward)])


def to_modality(seq, modality='joint', parents=SHREC_PARENTS):
    """Convert a raw joint sequence (T, V, 3) into one of the four input modalities."""
    seq = np.asarray(seq, dtype=np.float32)
    if modality == 'joint':
        return seq
    bone = seq - seq[:, parents]
    if modality == 'bone':
        return bone
    src = seq if modality == 'joint_motion' else bone
    if modality not in ('joint_motion', 'bone_motion'):
        raise ValueError(f'Unknown modality: {modality}')
    motion = np.zeros_like(src)
    motion[1:] = src[1:] - src[:-1]
    return motion


# =====================================================================================
# SIPN stage 1: DTW / DBA phase alignment (numpy, runs in the data loader)
# =====================================================================================
def resample_linear(seq, out_len):
    """Uniform linear interpolation of a (T, ...) array to out_len frames."""
    seq = np.asarray(seq, dtype=np.float64)
    T = seq.shape[0]
    if T == out_len:
        return seq.copy()
    if T == 1:
        return np.repeat(seq, out_len, axis=0)
    pos = np.linspace(0, T - 1, out_len)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (pos - lo).reshape((-1,) + (1,) * (seq.ndim - 1))
    return seq[lo] * (1 - frac) + seq[hi] * frac


def sakoe_chiba_mask(n, m, band_ratio=0.1):
    """Sakoe-Chiba band around the (scaled) diagonal, radius = r * max(n, m) frames."""
    if band_ratio is None or band_ratio <= 0:
        return np.ones((n, m), dtype=np.bool_)
    longest = max(n, m)
    radius = max(1, int(math.ceil(band_ratio * longest)))
    i = np.arange(n)[:, None] / max(n - 1, 1)
    j = np.arange(m)[None, :] / max(m - 1, 1)
    return np.abs(i - j) * longest <= radius


@njit(cache=False)
def _dtw_accumulate(cost, mask):
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if not mask[i - 1, j - 1]:
                continue
            best = D[i - 1, j - 1]
            if D[i - 1, j] < best:
                best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            D[i, j] = cost[i - 1, j - 1] + best
    return D


def _dtw_matrix(a, b, band_ratio):
    a2 = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b2 = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    cost = ((a2[:, None, :] - b2[None, :, :]) ** 2).sum(-1)
    D = _dtw_accumulate(cost, sakoe_chiba_mask(len(a2), len(b2), band_ratio))
    if not np.isfinite(D[-1, -1]):
        # the band is infeasible for extreme length ratios: fall back to unconstrained DTW
        D = _dtw_accumulate(cost, np.ones_like(cost, dtype=np.bool_))
    return D


def dtw_distance(a, b, band_ratio=0.1):
    """DTW distance (sqrt of the accumulated squared Euclidean cost), eq. (1)."""
    return float(np.sqrt(_dtw_matrix(a, b, band_ratio)[-1, -1]))


def dtw_path(a, b, band_ratio=0.1):
    """DTW alignment path pi(a, b) = [(i_1, j_1), ..., (i_L, j_L)], eq. (2),
    satisfying the monotonicity, continuity and boundary conditions."""
    D = _dtw_matrix(a, b, band_ratio)
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        candidates = ((D[i - 1, j - 1], i - 1, j - 1),   # diagonal first on ties
                      (D[i - 1, j], i - 1, j),
                      (D[i, j - 1], i, j - 1))
        _, i, j = min(candidates, key=lambda c: c[0])
        path.append((i - 1, j - 1))
    path.reverse()
    return path, float(np.sqrt(D[-1, -1]))


def warp_to_prototype(seq, path, proto_len):
    """Phase-aligned sequence S' (eq. 3): every prototype frame j takes the mean of
    all input frames mapped onto it. Prototype frames that receive no input frame
    (edge case) are linearly interpolated from the neighbouring assigned frames."""
    seq = np.asarray(seq, dtype=np.float64)
    idx_in = np.fromiter((p[0] for p in path), dtype=np.int64)
    idx_pr = np.fromiter((p[1] for p in path), dtype=np.int64)
    feat = seq.reshape(len(seq), -1)
    sums = np.zeros((proto_len, feat.shape[1]))
    counts = np.zeros(proto_len)
    np.add.at(sums, idx_pr, feat[idx_in])
    np.add.at(counts, idx_pr, 1.0)
    assigned = counts > 0
    out = np.zeros_like(sums)
    out[assigned] = sums[assigned] / counts[assigned, None]
    if not assigned.all():
        known = np.nonzero(assigned)[0]
        missing = np.nonzero(~assigned)[0]
        for d in range(out.shape[1]):
            out[missing, d] = np.interp(missing, known, out[known, d])
    return out.reshape((proto_len,) + seq.shape[1:])


def dba_prototype(sequences, proto_len, band_ratio=0.1, max_iter=10, tol=1e-4, verbose=False):
    """DTW Barycenter Averaging (Petitjean et al.) for one class (Sec. 3.2.1).

    Initialisation: mean of the class instances uniformly resampled to proto_len
    (the class-median length). Each iteration aligns every instance to the current
    prototype and replaces each prototype frame with the average of the aligned
    frames. Stops when the mean DBA objective changes by less than `tol`.
    """
    seqs = [np.asarray(s, dtype=np.float64) for s in sequences]
    feat_shape = seqs[0].shape[1:]
    proto = np.mean([resample_linear(s, proto_len) for s in seqs], axis=0)
    prev_obj = None
    for it in range(max_iter):
        sums = np.zeros((proto_len, int(np.prod(feat_shape))))
        counts = np.zeros(proto_len)
        obj = 0.0
        for s in seqs:
            path, dist = dtw_path(s, proto, band_ratio)
            i_idx = np.fromiter((p[0] for p in path), dtype=np.int64)
            j_idx = np.fromiter((p[1] for p in path), dtype=np.int64)
            np.add.at(sums, j_idx, s.reshape(len(s), -1)[i_idx])
            np.add.at(counts, j_idx, 1.0)
            obj += dist ** 2
        obj /= len(seqs)
        flat = proto.reshape(proto_len, -1).copy()
        assigned = counts > 0
        flat[assigned] = sums[assigned] / counts[assigned, None]
        proto = flat.reshape((proto_len,) + feat_shape)
        if verbose:
            print(f'    DBA iter {it + 1}: objective {obj:.6f}')
        if prev_obj is not None and abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
    return proto.astype(np.float32)


class SIPNPreprocessor:
    """Stage 1 of SIPN: class-specific DBA prototypes + label-free DTW alignment.

    Fit once per modality on the training split (per fold for DHG leave-one-subject-out),
    save to disk, and keep frozen afterwards. Applying the transform never uses labels:
    the alignment target is the nearest prototype (eq. 1), and that index is discarded.
    """

    def __init__(self, band_ratio=0.1, max_iter=10, tol=1e-4):
        self.band_ratio = band_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.prototypes = {}    # class id -> (Tp_c, V, C) array

    # ---------------------------------------------------------------- fitting / io
    def fit(self, sequences, labels, verbose=False):
        labels = np.asarray(labels)
        self.prototypes = {}
        for c in np.unique(labels):
            members = [sequences[k] for k in np.nonzero(labels == c)[0]]
            proto_len = int(np.median([len(s) for s in members]))   # class-median length
            if verbose:
                print(f'  class {c}: {len(members)} instances, Tp = {proto_len}')
            self.prototypes[int(c)] = dba_prototype(
                members, proto_len, self.band_ratio, self.max_iter, self.tol, verbose)
        return self

    def save(self, path):
        np.savez(path,
                 band_ratio=self.band_ratio, max_iter=self.max_iter, tol=self.tol,
                 **{f'proto_{c}': p for c, p in self.prototypes.items()})

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls(float(data['band_ratio']), int(data['max_iter']), float(data['tol']))
        obj.prototypes = {int(k.split('_')[1]): data[k] for k in data.files if k.startswith('proto_')}
        return obj

    @property
    def max_len(self):
        return max(len(p) for p in self.prototypes.values())

    # ---------------------------------------------------------------- transform
    def nearest_prototype(self, seq):
        """c_hat = argmin_c DTW(S, P_c), eq. (1). No ground-truth label is used."""
        best_c, best_d = None, np.inf
        for c, proto in self.prototypes.items():
            d = dtw_distance(seq, proto, self.band_ratio)
            if d < best_d:
                best_c, best_d = c, d
        return best_c, best_d

    def __call__(self, seq):
        """Return the phase-aligned sequence S' of shape (Tp_c_hat, V, C)."""
        c_hat, _ = self.nearest_prototype(seq)
        proto = self.prototypes[c_hat]
        path, _ = dtw_path(seq, proto, self.band_ratio)
        return warp_to_prototype(seq, path, len(proto)).astype(np.float32)


def build_sipn_preprocessors(train_sequences, train_labels, modalities=MODALITIES,
                             parents=SHREC_PARENTS, cache_dir=None, verbose=True, **kwargs):
    """Fit (or load from cache) one prototype bank per modality."""
    banks = {}
    for m in modalities:
        path = os.path.join(cache_dir, f'sipn_prototypes_{m}.npz') if cache_dir else None
        if path is not None and os.path.isfile(path):
            banks[m] = SIPNPreprocessor.load(path)
            continue
        if verbose:
            print(f'Fitting DBA prototypes for modality "{m}"')
        seqs = [to_modality(s, m, parents) for s in train_sequences]
        banks[m] = SIPNPreprocessor(**kwargs).fit(seqs, train_labels, verbose=False)
        if path is not None:
            os.makedirs(cache_dir, exist_ok=True)
            banks[m].save(path)
    return banks


class SIPNSkeletonDataset(torch.utils.data.Dataset):
    """Dataset wrapper producing phase-aligned, zero-padded sequences.

    Each item: (x, length, label) with x of shape (C, T_pad, V, 1) and `length` the
    number of valid frames. The model's MGTS layer (or uniform resampling) then maps
    the valid frames to the fixed network length N.

    use_dtw=False gives the "MGTS only" / baseline configurations (raw timebase).
    """

    def __init__(self, sequences, labels, modality='joint', preprocessor=None, use_dtw=True,
                 parents=SHREC_PARENTS, pad_len=None, precompute=True):
        if use_dtw and preprocessor is None:
            raise ValueError('use_dtw=True requires a fitted SIPNPreprocessor')
        self.sequences = sequences
        self.labels = np.asarray(labels, dtype=np.int64)
        self.modality = modality
        self.preprocessor = preprocessor
        self.use_dtw = use_dtw
        self.parents = parents
        if pad_len is None:
            pad_len = preprocessor.max_len if use_dtw else max(len(s) for s in sequences)
        self.pad_len = pad_len
        self.cache = [self._process(k) for k in range(len(sequences))] if precompute else None

    def _process(self, k):
        seq = to_modality(self.sequences[k], self.modality, self.parents)
        if self.use_dtw:
            seq = self.preprocessor(seq)
        seq = seq[:self.pad_len]
        length = len(seq)
        out = np.zeros((self.pad_len,) + seq.shape[1:], dtype=np.float32)
        out[:length] = seq
        return out, length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, k):
        seq, length = self.cache[k] if self.cache is not None else self._process(k)
        x = torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(-1).contiguous()   # C, T, V, 1
        return x, torch.tensor(length, dtype=torch.long), torch.tensor(self.labels[k])


# =====================================================================================
# SIPN stage 2: Motion-Gated Temporal Sampling (torch, inside the model)
# =====================================================================================
def valid_frame_mask(lengths, T):
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


def uniform_resample(x, lengths, out_len):
    """Baseline resampling: uniform linear interpolation of the first `lengths` frames
    of x (N, C, T, V, M) to out_len frames."""
    N, C, T, V, M = x.shape
    L = lengths.to(x.dtype).clamp(min=1)
    pos = torch.linspace(0, 1, out_len, device=x.device, dtype=x.dtype)[None] * (L[:, None] - 1)
    lo = pos.floor().long()
    hi = torch.minimum(lo + 1, (lengths[:, None] - 1).clamp(min=0))
    frac = (pos - lo.to(x.dtype))[:, None, :, None, None]

    def take(idx):
        return torch.gather(x, 2, idx[:, None, :, None, None].expand(N, C, out_len, V, M))

    return take(lo) * (1 - frac) + take(hi) * frac


class MotionGatedTemporalSampling(nn.Module):
    """MGTS (Sec. 3.3): stratified, motion-biased selection of N frames from the
    phase-aligned sequence S', with a learnable temperature tau = softplus(rho).

    Note: when N == B (paper setting, 150/150) every stratum contributes one frame and
    the hard argmax inside a stratum does not depend on tau (softmax is monotone), so
    tau is learned purely through the straight-through surrogate gradient.
    """

    def __init__(self, out_len=150, num_strata=150, init_tau=1.0, learnable=True, straight_through=True):
        super().__init__()
        assert out_len >= num_strata, 'eq. (6) requires N >= B'
        self.out_len = out_len
        self.num_strata = num_strata
        self.straight_through = straight_through
        rho = torch.tensor(math.log(math.expm1(init_tau)), dtype=torch.float32)   # softplus(rho) = init_tau
        if learnable:
            self.rho = nn.Parameter(rho)
        else:
            self.register_buffer('rho', rho)

    @property
    def tau(self):
        return F.softplus(self.rho)

    @staticmethod
    def motion_score(x, lengths):
        """eq. (4): m_t = sum_i ||v_t,i - v_t-1,i||_2, m_1 = 0 (summed over persons as well)."""
        disp = torch.linalg.vector_norm(x[:, :, 1:] - x[:, :, :-1], dim=1)   # N, T-1, V, M
        m = disp.sum(dim=(-1, -2))
        m = F.pad(m, (1, 0))                                                  # N, T
        return m.masked_fill(~valid_frame_mask(lengths, x.shape[2]), 0.)

    def importance_weights(self, x, lengths):
        """eq. (5): temperature-scaled softmax over the valid frames."""
        m = self.motion_score(x, lengths)
        logits = (m / self.tau).masked_fill(~valid_frame_mask(lengths, x.shape[2]), float('-inf'))
        return torch.softmax(logits, dim=-1)

    def strata_membership(self, lengths, T):
        """Boolean (N, B, T): frame t belongs to stratum b.

        Strata are B contiguous, equal-width intervals of the phase axis [0, L). A frame
        belongs to every stratum its unit interval overlaps, so no stratum is empty
        even when B > L (the paper's setting, L ~ 35 and B = 150); frames are then
        repeated across strata as stated in Sec. 3.3.3.
        """
        B = self.num_strata
        L = lengths.to(torch.float64)[:, None]
        b = torch.arange(B, device=lengths.device, dtype=torch.float64)[None]
        lo = torch.floor(b * L / B)
        hi = torch.maximum(torch.ceil((b + 1) * L / B), lo + 1)
        lo = torch.minimum(lo, L - 1)
        t = torch.arange(T, device=lengths.device, dtype=torch.float64)[None, None]
        return (t >= lo[..., None]) & (t < hi[..., None])

    def forward(self, x, lengths=None):
        N, C, T, V, M = x.shape
        if lengths is None:
            lengths = torch.full((N,), T, dtype=torch.long, device=x.device)
        lengths = lengths.to(x.device).clamp(min=1, max=T)

        w = self.importance_weights(x, lengths)                  # N, T  (sum_t w_t = 1)
        member = self.strata_membership(lengths, T)              # N, B, T
        member_w = member.to(w.dtype) * w[:, None, :]            # w_t restricted to B_b

        if self.out_len == self.num_strata:
            # n_b = 1 for every stratum: argmax of w_t inside each stratum (eq. 7)
            idx = member_w.masked_fill(~member, -1.).argmax(-1)  # N, B
            slot_stratum = None
        else:
            idx, slot_stratum = self._allocate(member, member_w)

        hard = torch.gather(x, 2, idx[:, None, :, None, None].expand(N, C, idx.shape[1], V, M))
        if not (self.straight_through and self.training and torch.is_grad_enabled()):
            return hard

        # STE: backward treats X_b as sum_{t in B_b} w_t S'_t, forward keeps the hard selection
        soft = torch.einsum('nbt,nctvm->ncbvm', member_w, x)     # N, C, B, V, M
        if slot_stratum is not None:
            soft = torch.gather(soft, 2, slot_stratum[:, None, :, None, None].expand_as(hard))
        return hard + soft - soft.detach()

    @torch.no_grad()
    def _allocate(self, member, member_w):
        """General case N > B: allocation eq. (6) and within-stratum top-n_b selection eq. (7)."""
        N_out, B = self.out_len, self.num_strata
        W = member_w.sum(-1)
        W = W / W.sum(-1, keepdim=True)
        idx_all, slot_all = [], []
        for n in range(member.shape[0]):
            n_b = 1 + torch.round((N_out - B) * W[n]).long()
            n_b[W[n].argmax()] += N_out - int(n_b.sum())            # rounding residual
            idx, slots = [], []
            for b in range(B):
                if n_b[b] <= 0:
                    continue
                frames = torch.nonzero(member[n, b]).flatten()
                order = frames[torch.argsort(member_w[n, b, frames], descending=True)]
                chosen = order[torch.arange(int(n_b[b]), device=order.device) % len(order)]
                idx.append(torch.sort(chosen).values)                # ascending phase order
                slots.append(torch.full((int(n_b[b]),), b, device=order.device, dtype=torch.long))
            idx_all.append(torch.cat(idx))
            slot_all.append(torch.cat(slots))
        return torch.stack(idx_all), torch.stack(slot_all)


# =====================================================================================
# GCN backbone (unchanged from the baseline, SIPN does not modify the architecture)
# =====================================================================================
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=8):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)


class dgmstcn(nn.Module):
    """Multi-scale temporal convolution (MS-TCN) with a global-joint branch."""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_joints=25, dropout=0.,
                 ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'), stride=1):
        super().__init__()
        ms_cfg = list(ms_cfg)
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()
        self.num_joints = num_joints
        # the size of add_coeff can be larger than the actual number of joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels
        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None)))
        self.branches = nn.ModuleList(branches)

        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        return self.transform(local_feat + global_feat)

    def forward(self, x):
        return self.drop(self.bn(self.inner_forward(x)))


class criss_ctrgcn(nn.Module):
    """Spatial graph convolution with channel-wise topology refinement (GC-GC style) and
    a per-frame temporal topology A1 (GT-GC style, shape N x K x T x V x V)."""

    def __init__(self, in_channels, out_channels, A, ratio=0.25, ctr='T', ada='T', subset_wise=False,
                 ada_act='softmax', ctr_act='tanh', norm='BN', act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        self.subset_wise = subset_wise
        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets        # default: 1 / number of subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # introduce non-linearity
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        # coordinate attention on the projected features
        self.se = CoordAtt(mid_channels * num_subsets, mid_channels * num_subsets)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.pai = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        # temporal-scale projection for the per-frame topology
        self.conv3 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        # ITV diagnostics: store the per-frame topology when enabled
        self.record_topology = False
        self.temporal_topology = None

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        x0 = x
        res = self.down(x)
        A = self.A
        A0 = A[None, :, None, None]                  # 1 (N), K, 1 (C), 1 (T), V, V
        pre_x = self.se(self.pre(x))
        pre_x = pre_x.reshape(n, self.num_subsets, self.mid_channels, t, v)   # N, K, C, T, V

        # temporal gate before pooling over time
        tmp_x = x
        gate_t = self.softmax(tmp_x.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True))
        tmp_x = (tmp_x * gate_t).mean(dim=-2, keepdim=True)          # n c t v -> n c 1 v
        x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)   # n k d 1 v
        x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)

        if self.ctr is not None:
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)             # pairwise difference, n k d 1 v v
            ada_graph = getattr(self, self.ctr_act)(diff)
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A0

        # graph aggregation X W A
        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6        # n k c t v v, where C and T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A.squeeze(2).squeeze(2)).contiguous()
            elif A.shape[2] == 1:
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A.squeeze(2)).contiguous()
            elif A.shape[3] == 1:
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A.squeeze(3)).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            A = A.squeeze()                 # K, V, V
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()

        # per-frame (temporal) topology: one V x V adjacency per frame and subset
        x3 = rearrange(self.conv3(x0), 'n (k d) t v -> n k d t v', k=self.num_subsets)
        x3 = x3.mean(dim=-3, keepdim=True)                 # n k 1 t v
        A1 = self.tanh(x3.unsqueeze(-1) - x3.unsqueeze(-2)).squeeze(2)   # n k t v v
        if self.record_topology:
            self.temporal_topology = A1.detach()
        x_t = torch.einsum('nkctv,nktvw->nkctw', pre_x, A1).contiguous()
        x = x + x_t * self.pai[0]
        x = self.post(x.reshape(n, -1, t, v))
        return self.act(self.bn(x) + res)


class DGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, num_joints=25):
        super().__init__()
        self.gcn = criss_ctrgcn(in_channels, out_channels, A, ratio=None, ctr='T', ada=None,
                                subset_wise=False, ada_act='softmax', ctr_act='tanh', norm='BN', act='ReLU')
        self.tcn = dgmstcn(out_channels, out_channels, mid_channels=None, num_joints=num_joints, dropout=0.,
                           ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'], stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)  # temporal downsampling

    def forward(self, x, A=None):
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


# =====================================================================================
# Full model: SIPN stage 2 (MGTS) + backbone
# =====================================================================================
class Model(nn.Module):
    """GCN backbone preceded by the SIPN temporal sampler.

    Inputs:
        x       : (N, C, T_pad, V, M) phase-aligned (or raw) sequences, zero-padded in time
        lengths : (N,) number of valid frames per sample; None means all T_pad frames are valid
    sampler:
        'mgts'    -> Motion-Gated Temporal Sampling (SIPN, the proposed method)
        'uniform' -> uniform linear interpolation to out_len (baseline / "DTW only" ablation)
        'none'    -> input is already resampled to a fixed length
    Train one model per modality (J, B, JM, BM); each owns its own tau.
    """

    def __init__(self,
                 graph_cfg=dict(),
                 num_classes=14,
                 in_channels=3,
                 base_channels=64,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=(5, 8),
                 down_stages=(5, 8),
                 data_bn_type='VC',
                 num_person=1,
                 pretrained=None,
                 dropout=0.0,
                 A=None,
                 sampler='mgts',
                 out_len=150,
                 num_strata=150,
                 init_tau=1.0,
                 learn_tau=True):
        super().__init__()

        # build the graph
        if A is None:
            if graph_cfg and Graph is not None:
                self.graph = Graph(**graph_cfg)
                A = self.graph.A
            else:
                A = hand_adjacency()
        A = torch.as_tensor(np.asarray(A), dtype=torch.float32)

        # SIPN stage 2
        assert sampler in ('mgts', 'uniform', 'none')
        self.sampler = sampler
        self.out_len = out_len
        if sampler == 'mgts':
            self.mgts = MotionGatedTemporalSampling(out_len, num_strata, init_tau, learnable=learn_tau)

        # input normalisation
        self.data_bn_type = data_bn_type
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        num_joints = A.size(-1)

        # channel inflation and temporal downsampling
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            # first layer: in_channels (3) -> base_channels (64)
            modules = [DGBlock(in_channels, base_channels, A.clone(), 1, residual=False, num_joints=num_joints)]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            # stages 2 .. num_stages; stride 2 at the down stages
            stride = 1 + (i in down_stages)
            in_ch = base_channels
            if i in inflate_stages:
                inflate_times += 1      # channels x ch_ratio at every inflate stage
            out_ch = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_ch
            modules.append(DGBlock(in_ch, out_ch, A.clone(), stride, num_joints=num_joints))

        if self.in_channels == self.base_channels:  # no stem layer in this case
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else None
        self.fc_cls = nn.Linear(base_channels, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_class = num_classes

    @property
    def tau(self):
        return self.mgts.tau if self.sampler == 'mgts' else None

    def temporal_resample(self, x, lengths=None):
        if self.sampler == 'mgts':
            return self.mgts(x, lengths)
        if self.sampler == 'uniform':
            if lengths is None:
                lengths = torch.full((x.shape[0],), x.shape[2], dtype=torch.long, device=x.device)
            return uniform_resample(x, lengths.to(x.device), self.out_len)
        return x

    def forward(self, x, y=None, lengths=None):
        x = self.temporal_resample(x, lengths)      # SIPN stage 2: (N, C, out_len, V, M)

        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> n m v c t')
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.reshape(N, M * V * C, T))
        else:
            x = self.data_bn(x.reshape(N * M, V * C, T))
        x = x.reshape(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().reshape(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = self.pool(x).reshape(N, M, -1).mean(1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x), y

    # ------------------------------------------------------------ ITV support
    def record_topology(self, enabled=True):
        """Store the last block's per-frame adjacency during forward (for ITV)."""
        self.gcn[-1].gcn.record_topology = enabled

    def get_temporal_topology(self):
        """Last block's per-frame adjacency, averaged over subsets: (N*M, T', V, V)."""
        A1 = self.gcn[-1].gcn.temporal_topology
        return None if A1 is None else A1.mean(1)


# =====================================================================================
# Training helpers (Table 3)
# =====================================================================================
def build_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4):
    """SGD with Nesterov momentum; rho (tau) is optimised with the GCN weights,
    excluded from weight decay so that tau is driven by the cross-entropy only."""
    tau_params = [p for n, p in model.named_parameters() if n.endswith('mgts.rho')]
    other = [p for n, p in model.named_parameters() if not n.endswith('mgts.rho')]
    groups = [{'params': other, 'weight_decay': weight_decay}]
    if tau_params:
        groups.append({'params': tau_params, 'weight_decay': 0.0})
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def build_scheduler(optimizer, warmup_epochs=20, milestones=(70, 100), gamma=0.1):
    """Per-epoch schedule: linear warm-up, then step decay x0.1 at the milestones."""
    def factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return gamma ** sum(epoch >= m for m in milestones)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def fuse_scores(scores):
    """Equal-weight score fusion over modality streams (J, B, JM, BM)."""
    return torch.stack([torch.as_tensor(s) for s in scores]).sum(0)


# =====================================================================================
# Diagnostics: ITV (eq. 8-9), KIV (eq. 10), duration-stratified accuracy (Sec. 4.4)
# =====================================================================================
def _mean_pairwise_distance(feat):
    """feat: (n, T, D) -> mean over t of the mean pairwise L2 distance over n != n'."""
    n = feat.shape[0]
    if n < 2:
        return float('nan')
    d = torch.cdist(feat.transpose(0, 1).double(), feat.transpose(0, 1).double())   # T, n, n
    return float((d.sum(dim=(1, 2)) / (n * (n - 1))).mean())


def intra_class_topology_variance(adjacency, labels):
    """ITV. adjacency: (num_instances, T, V, V) per-frame adjacency of the last GCN layer
    on the test set; labels: (num_instances,). Returns (mean ITV, {class: ITV_c})."""
    adjacency = torch.as_tensor(adjacency)
    labels = torch.as_tensor(labels)
    per_class = {}
    for c in torch.unique(labels).tolist():
        A_c = adjacency[labels == c]
        per_class[c] = _mean_pairwise_distance(A_c.flatten(2))     # Frobenius norm
    vals = [v for v in per_class.values() if not math.isnan(v)]
    return float(np.mean(vals)), per_class


def kinematic_descriptor(x):
    """Phi (n, T, D), D = 3V + 3V + V(V-1)/2 from joint trajectories x (n, T, V, 3):
    normalised positions, inter-frame displacements and pairwise joint distances."""
    x = torch.as_tensor(x, dtype=torch.float64)
    n, T, V, _ = x.shape
    centred = x - x.mean(dim=(1, 2), keepdim=True)
    scale = centred.flatten(1).norm(dim=1).clamp(min=1e-8) / math.sqrt(T * V)
    pos = centred / scale[:, None, None, None]
    disp = torch.zeros_like(pos)
    disp[:, 1:] = pos[:, 1:] - pos[:, :-1]
    iu = torch.triu_indices(V, V, offset=1)
    dist = torch.cdist(pos.reshape(n * T, V, 3), pos.reshape(n * T, V, 3))[:, iu[0], iu[1]].reshape(n, T, -1)
    return torch.cat([pos.reshape(n, T, -1), disp.reshape(n, T, -1), dist], dim=-1)


def kinematic_intra_class_variance(x, labels):
    """KIV, model-independent counterpart of ITV (eq. 10); x: (n, T, V, 3) on the phase axis."""
    phi = kinematic_descriptor(x)
    labels = torch.as_tensor(labels)
    per_class = {c: _mean_pairwise_distance(phi[labels == c]) for c in torch.unique(labels).tolist()}
    vals = [v for v in per_class.values() if not math.isnan(v)]
    return float(np.mean(vals)), per_class


def duration_stratified_accuracy(preds, labels, durations, cut_points=None):
    """Top-1 accuracy per duration quartile of the original frame counts (Sec. 4.4)."""
    preds, labels, durations = map(np.asarray, (preds, labels, durations))
    if cut_points is None:
        cut_points = np.percentile(durations, [25, 50, 75])
    bins = np.digitize(durations, cut_points, right=True)
    result = {f'Q{q + 1}': float((preds[bins == q] == labels[bins == q]).mean() * 100)
              for q in range(len(cut_points) + 1) if (bins == q).any()}
    result['All'] = float((preds == labels).mean() * 100)
    return result, cut_points


# =====================================================================================
# Smoke test with synthetic hand sequences
# =====================================================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes, V = 4, 22

    # synthetic variable-length gestures (T_S in [20, 50]): class-specific trajectories
    def make_seq(c, T):
        phase = np.linspace(0, 1, T) ** rng.uniform(0.7, 1.4)
        base = rng.normal(size=(1, V, 3)) * 0.05
        traj = np.stack([np.sin(2 * np.pi * (c + 1) * phase), np.cos(np.pi * phase * (c + 1)), phase], -1)
        return (base + traj[:, None, :] * (1 + 0.1 * np.arange(V))[None, :, None]).astype(np.float32)

    train_labels = np.repeat(np.arange(num_classes), 6)
    train_seqs = [make_seq(c, int(rng.integers(20, 51))) for c in train_labels]
    test_labels = np.repeat(np.arange(num_classes), 2)
    test_seqs = [make_seq(c, int(rng.integers(20, 51))) for c in test_labels]

    # stage 1: prototypes (training split only), then label-free alignment
    pre = SIPNPreprocessor(band_ratio=0.1, max_iter=10, tol=1e-4).fit(train_seqs, train_labels)
    print('prototype lengths:', {c: len(p) for c, p in pre.prototypes.items()})
    train_set = SIPNSkeletonDataset(train_seqs, train_labels, 'joint', pre)
    test_set = SIPNSkeletonDataset(test_seqs, test_labels, 'joint', pre, pad_len=train_set.pad_len)
    loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

    # stage 2 + backbone
    net = Model(num_classes=num_classes, num_person=1, sampler='mgts', out_len=150, num_strata=150).to(device)
    optimizer = build_optimizer(net)
    x, lengths, y = next(iter(loader))
    x, lengths, y = x.to(device), lengths.to(device), y.to(device)
    logits, _ = net(x, lengths=lengths)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    print('logits:', tuple(logits.shape), '| tau:', float(net.tau.detach()), '| d loss / d rho:', float(net.mgts.rho.grad))
    optimizer.step()

    # general allocation path (N > B)
    mgts = MotionGatedTemporalSampling(out_len=150, num_strata=50).to(device)
    print('MGTS N=150, B=50 ->', tuple(mgts(x, lengths).shape))

    # ITV on the test set
    net.eval()
    net.record_topology(True)
    with torch.no_grad():
        xt = torch.stack([test_set[k][0] for k in range(len(test_set))]).to(device)
        lt = torch.stack([test_set[k][1] for k in range(len(test_set))]).to(device)
        net(xt, lengths=lt)
    itv, _ = intra_class_topology_variance(net.get_temporal_topology().cpu(), test_labels)
    print(f'ITV: {itv:.4f}')
