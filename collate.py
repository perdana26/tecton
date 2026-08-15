"""
Collate — dari Sample ke input tiap model, dari SUMBER data yang SAMA
====================================================================

Prinsip keadilan: MAMBA, AEGCN, dan model Anda WAJIB membaca sampel yang sama,
label space yang sama (27 teknik), dan split yang sama. Perbedaan hasil harus
berasal dari ARSITEKTUR, bukan dari perbedaan preprocessing. Modul ini menjamin
itu dengan satu titik konversi.

Butuh: torch, numpy. (Graf dibangun tanpa torch_geometric agar tidak menambah
dependency; jika model Anda memakai PyG, ganti build_graph_tensors sesuai.)
"""

from __future__ import annotations

import numpy as np
import torch

from dataset import AvastCTUDataset, Sample


# ==========================================================================
# 1. Untuk BoA + MLP (baseline frekuensi)
# ==========================================================================
def collate_boa(samples: list[Sample], ds: AvastCTUDataset):
    V = len(ds.api_vocab)
    X = np.zeros((len(samples), V), dtype=np.float32)
    Y = np.zeros((len(samples), len(ds.techniques)), dtype=np.float32)
    for i, s in enumerate(samples):
        for a in s.api_sequence(ds.api_vocab):
            X[i, a] += 1.0
        Y[i] = s.label_vector(ds.techniques)
    return torch.from_numpy(X), torch.from_numpy(Y)


# ==========================================================================
# 2. Untuk LSTM / Transformer (sekuens)
# ==========================================================================
def collate_sequence(samples: list[Sample], ds: AvastCTUDataset,
                     max_len: int = 512):
    seqs, lengths, Y = [], [], []
    for s in samples:
        seq = s.api_sequence(ds.api_vocab, max_len=max_len)
        seq = [a + 1 for a in seq]                 # 0 = padding
        seqs.append(seq)
        lengths.append(len(seq))
        Y.append(s.label_vector(ds.techniques))
    T = max(lengths)
    padded = np.zeros((len(seqs), T), dtype=np.int64)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    mask = padded != 0
    return (torch.from_numpy(padded), torch.tensor(lengths),
            torch.from_numpy(mask), torch.tensor(np.stack(Y)))


# ==========================================================================
# 3. Untuk graf transisi (model Anda + AEGCN)
# ==========================================================================
def build_graph_tensors(s: Sample, ds: AvastCTUDataset):
    """Bangun graf transisi API sekali; dipakai model Anda & AEGCN.

    Mengembalikan node_ids, edge_index, edge_weight (probabilitas transisi),
    dan matriks adjacency+transisi padat untuk AEGCN.
    """
    V = len(ds.api_vocab)
    seq = s.api_sequence(ds.api_vocab)
    counts = {}
    for a, b in zip(seq[:-1], seq[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    if not counts:
        counts = {(0, 0): 1}                       # graf kosong -> self-loop aman

    edges = list(counts.keys())
    edge_index = torch.tensor(edges, dtype=torch.long).t()      # [2, E]
    # bobot = probabilitas transisi (dinormalisasi per node sumber)
    out_sum = {}
    for (a, _), w in counts.items():
        out_sum[a] = out_sum.get(a, 0) + w
    edge_weight = torch.tensor([counts[e] / out_sum[e[0]] for e in edges],
                               dtype=torch.float32)
    node_ids = torch.arange(V, dtype=torch.long)                # semua node global
    return {"node_ids": node_ids, "edge_index": edge_index,
            "edge_weight": edge_weight, "n_nodes": V}


def build_aegcn_tensors(s: Sample, ds: AvastCTUDataset):
    """Matriks padat |V|x|V| untuk AEGCN (adjacency biner + transisi Markov)."""
    V = len(ds.api_vocab)
    seq = s.api_sequence(ds.api_vocab)
    A = np.zeros((V, V), dtype=np.float32)
    counts = np.zeros((V, V), dtype=np.float32)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[a, b] += 1.0
        A[a, b] = 1.0
    row = counts.sum(1, keepdims=True)
    P = np.divide(counts, row, out=np.zeros_like(counts), where=row > 0)
    X = np.eye(V, dtype=np.float32)                # fitur node one-hot global
    return (torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(P))


# ==========================================================================
# 4. Untuk MAMBA (kategori + nama API + resource; butuh PV-DM)
# ==========================================================================
def collate_mamba(samples: list[Sample], ds: AvastCTUDataset,
                  res_embedder, attck_knowledge, binding_mlp,
                  max_resources: int = 3):
    """Siapkan struktur per-sampel untuk MAMBAFull.

    res_embedder    : ResourceEmbedder terlatih (PV-DM)
    attck_knowledge : {technique: [resource_string, ...]} hasil crawl ATT&CK
    binding_mlp     : BindingMLP terlatih -> z_r untuk tiap resource kandidat

    Menghasilkan list dict sesuai MAMBAFull.forward. Ini bagian termahal;
    cache embedding resource di disk saat integrasi nyata.
    """
    # Embedding resource kandidat ATT&CK (sekali, di luar loop idealnya)
    cand_res, cand_emb = [], []
    for tech, resources in attck_knowledge.items():
        for r in resources:
            cand_res.append((tech, r))
            cand_emb.append(res_embedder.embed(r))
    attck_res = torch.tensor(np.stack(cand_emb), dtype=torch.float32)  # [K, d]
    with torch.no_grad():
        z = binding_mlp.binding(attck_res)                            # [K, z]

    batch = []
    cat_vocab, api_vocab = ds.cat_vocab, ds.api_vocab
    for s in samples:
        cat_ids, api_ids, call_res = [], [], []
        for p in s.procs:
            c_ids, a_ids, c_res = [], [], []
            for c in p["calls"]:
                c_ids.append(cat_vocab.get(c["category"], 0))
                a_ids.append(api_vocab.get(c["api"], 0))
                # embed hingga max_resources resource per call
                embs = [res_embedder.embed(r) for r in c["resources"][:max_resources]]
                while len(embs) < max_resources:
                    embs.append(np.zeros(res_embedder.dim, dtype=np.float32))
                c_res.append(np.stack(embs))
            cat_ids.append(c_ids); api_ids.append(a_ids); call_res.append(c_res)
        # pad antar-proses saat integrasi; di sini per-proses list
        batch.append({
            "cat_ids": [torch.tensor(x) for x in cat_ids],
            "api_ids": [torch.tensor(x) for x in api_ids],
            "call_res": [torch.tensor(np.stack(x)) for x in call_res],
            "attck_res": attck_res, "z": z,
            "y": torch.tensor(s.label_vector(ds.techniques)),
        })
    return batch
