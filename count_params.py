"""
Hitung jumlah parameter tiap model — tanpa pelatihan
=====================================================

Mengisi kolom Params pada tabel biaya. Model diinstansiasi lalu parameternya
dihitung; tidak ada data yang dimuat dan tidak ada langkah pelatihan, jadi
selesai dalam hitungan detik.

Termasuk varian graf vocabulary-wide versus per-sample. Keduanya punya jumlah
parameter IDENTIK -- perbedaan biayanya murni pada jumlah node yang diproses
per batch, bukan pada ukuran model. Itu poin yang layak dinyatakan di paper,
karena pembaca mungkin mengira penghematan berasal dari model yang lebih kecil.

Jalankan:
    python count_params.py dataset_v2.npz
"""

from __future__ import annotations

import sys

import torch

from fast_dataset import FastDataset
from models import (BoAMLP, LSTMBaseline, TransformerBaseline,
                    MalwareTechniqueModel)


def count(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def fmt(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)


def main():
    if len(sys.argv) < 2:
        print("Pemakaian: python count_params.py <dataset.npz>")
        sys.exit(1)
    ds = FastDataset(sys.argv[1])
    V, K = len(ds.api_vocab), len(ds.techniques)
    od = ds.osint_dim

    models = [
        ("BoA + MLP",        lambda: BoAMLP(V, K)),
        ("LSTM",             lambda: LSTMBaseline(V, K)),
        ("Transformer",      lambda: TransformerBaseline(V, K)),
        ("TECTON",           lambda: MalwareTechniqueModel(V, 0, K, fusion="none")),
        ("TECTON + OSINT",   lambda: MalwareTechniqueModel(V, od, K, fusion="gated")),
    ]

    print(f"\n  vocab={V}  techniques={K}  osint_dim={od}\n")
    print(f"  {'Model':22s} {'Total':>10s} {'Trainable':>12s}  {'Untuk tabel':>12s}")
    print("  " + "-" * 60)
    for name, fn in models:
        try:
            t, tr = count(fn())
            print(f"  {name:22s} {t:10,d} {tr:12,d}  {fmt(t):>12s}")
        except Exception as e:
            print(f"  {name:22s} GAGAL: {e}")

    # AEGCN ditangani terpisah -- signature konstruktornya berbeda
    try:
        from aegcn import AEGCN
        t, tr = count(AEGCN(n_vocab=V, n_out=K, multilabel=True, pool="mean"))
        print(f"  {'AEGCN':22s} {t:10,d} {tr:12,d}  {fmt(t):>12s}")
    except Exception as e:
        print(f"  {'AEGCN':22s} GAGAL: {e}")

    # ---- rung ablation ladder ----
    print(f"\n  {'Rung':22s} {'Total':>10s} {'Trainable':>12s}  {'Untuk tabel':>12s}")
    print("  " + "-" * 60)
    try:
        from run_ablation_ladder import AblationModel, LADDER
        for name, cfg in LADDER:
            if cfg["arch"] == "aegcn":
                continue
            m = AblationModel(n_api=V, n_classes=K,
                              learned_emb=cfg["learned_emb"],
                              use_edge_attr=cfg["edge_attr"],
                              pool=cfg["pool"], arch=cfg["arch"])
            t, tr = count(m)
            print(f"  {name:22s} {t:10,d} {tr:12,d}  {fmt(t):>12s}")
    except Exception as e:
        print(f"  rung ablasi GAGAL: {e}")

    print("\n  Catatan: varian graf vocabulary-wide dan per-sample memakai model")
    print("  yang sama persis, sehingga jumlah parameternya identik. Selisih")
    print("  biaya pelatihan berasal dari jumlah node per batch (330 versus")
    print("  rata-rata 82), bukan dari ukuran model.")


if __name__ == "__main__":
    main()
