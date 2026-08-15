"""
Collate untuk FastDataset — versi cepat berbasis indeks
=======================================================

Pengganti collate.py. Bekerja langsung dengan array numerik FastDataset,
bukan objek Sample. Menghasilkan input IDENTIK secara nilai; hanya jalur
datanya yang berbeda (dari RAM, bukan disk).

MAMBA tidak ditangani di sini (butuh resource dari file mentah, lihat train_mamba.py).
"""

from __future__ import annotations

import numpy as np
import torch


# ==========================================================================
# 1. BoA + MLP
# ==========================================================================
def collate_boa(indices, ds):
    V = len(ds.api_vocab)
    X = np.zeros((len(indices), V), dtype=np.float32)
    Y = np.zeros((len(indices), len(ds.techniques)), dtype=np.float32)
    for row, i in enumerate(indices):
        for a in ds.api[i]:
            X[row, a] += 1.0
        Y[row] = ds.label_vec(i)
    return torch.from_numpy(X), torch.from_numpy(Y)


# ==========================================================================
# 2. Sequence (LSTM / Transformer)
# ==========================================================================
def collate_sequence(indices, ds, max_len: int = 256):
    """Pangkas ke 256 API call pertama (paper lama juga pakai 256).

    Banyak sampel punya ribuan call; memproses semuanya membuat LSTM/Transformer
    sangat lambat tanpa manfaat -- 256 langkah sudah menangkap perilaku awal yang
    diskriminatif. Ini konsisten dengan konvensi di literatur.
    """
    seqs, lengths, Y = [], [], []
    for i in indices:
        seq = ds.api[i][:max_len].astype(np.int64) + 1     # 0 = padding
        seqs.append(seq)
        lengths.append(len(seq))
        Y.append(ds.label_vec(i))
    T = max(lengths)
    padded = np.zeros((len(seqs), T), dtype=np.int64)
    for r, seq in enumerate(seqs):
        padded[r, :len(seq)] = seq
    mask = padded != 0
    return (torch.from_numpy(padded), torch.tensor(lengths),
            torch.from_numpy(mask), torch.tensor(np.stack(Y)))


# ==========================================================================
# 3. Graf transisi (GATv2 Anda)
# ==========================================================================
def build_graph_tensors(i, ds):
    """Bangun graf transisi HANYA dari API yang muncul di sampel ini.

    PENTING: versi sebelumnya memakai seluruh 330 node global untuk setiap graf,
    sehingga tiap sampel membawa ratusan node TERISOLASI (API yang tak pernah
    dipanggil). Itu (a) lambat, dan (b) salah secara semantik -- node terisolasi
    ikut mean-pooling dan mengencerkan representasi perilaku.

    Sekarang: node = API unik yang muncul; edge di-remap ke indeks lokal.
    node_ids tetap menyimpan ID API GLOBAL agar lookup embedding benar.
    """
    seq = ds.api[i].astype(np.int64)
    if len(seq) < 2:
        seq = np.array([0, 0], dtype=np.int64)

    present = np.unique(seq)                       # API yang benar-benar dipakai
    local = {int(a): k for k, a in enumerate(present)}

    counts = {}
    for a, b in zip(seq[:-1], seq[1:]):
        key = (local[int(a)], local[int(b)])
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        counts = {(0, 0): 1}

    edges = list(counts.keys())
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    out_sum = {}
    for (a, _), w in counts.items():
        out_sum[a] = out_sum.get(a, 0) + w
    edge_weight = torch.tensor([counts[e] / out_sum[e[0]] for e in edges],
                               dtype=torch.float32)
    node_ids = torch.tensor(present, dtype=torch.long)   # ID API global
    return {"node_ids": node_ids, "edge_index": edge_index,
            "edge_weight": edge_weight, "n_nodes": len(present)}


# ==========================================================================
# 4. AEGCN (matriks padat Markov)
# ==========================================================================
def build_aegcn_tensors(i, ds):
    V = len(ds.api_vocab)
    seq = ds.api[i].astype(np.int64)
    A = np.zeros((V, V), dtype=np.float32)
    counts = np.zeros((V, V), dtype=np.float32)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[a, b] += 1.0
        A[a, b] = 1.0
    row = counts.sum(1, keepdims=True)
    P = np.divide(counts, row, out=np.zeros_like(counts), where=row > 0)
    X = np.eye(V, dtype=np.float32)
    return (torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(P))
