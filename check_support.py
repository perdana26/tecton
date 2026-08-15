"""
Cek support per-teknik di test set tiap split
=============================================

Macro-F1 atas subset kecil bisa sangat berisik bila beberapa teknik hanya punya
segelintir sampel positif di test set. Sebelum menyatakan "Transformer
mengungguli GATv2 pada family split (subset andal)", kita harus tahu apakah
angka itu berdiri di atas support yang memadai.

Skrip ini melaporkan, untuk tiap split dan tiap teknik:
  - jumlah sampel POSITIF di test set
  - apakah memenuhi ambang minimum untuk F1 yang stabil (default 30)

Lalu menghitung ulang Macro-F1 subset ANDAL hanya atas teknik yang
support-nya memadai -- versi yang dapat dipertahankan di paper.

Jalankan:
    python check_support.py E:\multi_osint\dataset.npz E:\multi_osint\label_validation.csv results
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from fast_dataset import FastDataset
from splits import (random_split, family_disjoint_split, temporal_split,
                    behavioral_cluster_split)

MIN_SUPPORT = 30          # ambang minimum agar F1 per-teknik bermakna


def load_reliable(csv_path: Path, thresh=0.99):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    per = defaultdict(lambda: {"ok": 0, "all": 0})
    for r in rows:
        v = r.get("verdict", "").strip().lower()
        if not v:
            continue
        t = r["technique"]; n = int(r["n_samples"])
        per[t]["all"] += n
        if v == "correct":
            per[t]["ok"] += n
    return {t for t, d in per.items() if d["ok"] / max(d["all"], 1) >= thresh}


def main():
    if len(sys.argv) < 4:
        print("Pemakaian: python check_support.py <dataset.npz> <label_validation.csv> <results_dir>")
        sys.exit(1)
    ds = FastDataset(sys.argv[1])
    reliable = load_reliable(Path(sys.argv[2]))
    res_dir = Path(sys.argv[3])

    techs = ds.techniques
    splits = {
        "random":   random_split(len(ds), seed=42),
        "family":   family_disjoint_split(ds.families(), seed=42),
        "cluster":  behavioral_cluster_split(ds.boa_matrix(), n_clusters=50, seed=42),
        "temporal": temporal_split(np.array([str(t)[:19] for t in ds.timestamps()])),
    }

    print(f"Subset ANDAL ({len(reliable)}): {sorted(reliable)}\n")

    adequate = {}
    for sname, split in splits.items():
        Yte = ds.label[split.test]
        print("=" * 66)
        print(f"SPLIT: {sname}   (test n={len(split.test)})")
        print("=" * 66)
        print(f"{'teknik':8s} {'support':>8s} {'%test':>7s}  {'andal':>6s} {'cukup':>6s}")
        print("-" * 44)
        ok_list = []
        for k, t in enumerate(techs):
            sup = int(Yte[:, k].sum())
            is_rel = t in reliable
            is_ok = sup >= MIN_SUPPORT
            if is_rel:
                mark_r = "ya"
                if is_ok:
                    ok_list.append(t)
            else:
                mark_r = "-"
            if is_rel or sup < MIN_SUPPORT:
                print(f"{t:8s} {sup:8d} {100*sup/len(Yte):6.2f}%  {mark_r:>6s} "
                      f"{'ya' if is_ok else 'TIDAK':>6s}")
        adequate[sname] = ok_list
        excluded = sorted(reliable - set(ok_list))
        print(f"\n  andal & support cukup ({len(ok_list)}): {sorted(ok_list)}")
        if excluded:
            print(f"  andal TAPI support < {MIN_SUPPORT} ({len(excluded)}): {excluded}")
            print(f"  -> Macro-F1 atas subset andal penuh TIDAK STABIL untuk split ini")
        print()

    # --- hitung ulang Macro-F1 hanya atas teknik andal DAN support cukup ---
    results = {}
    for f in res_dir.glob("*results*.json"):
        try:
            results.update(json.loads(f.read_text()))
        except Exception:
            pass
    results = {k: v for k, v in results.items()
               if isinstance(v, dict) and "per_class_f1" in v and k.count("/") == 2}

    print("=" * 66)
    print("MACRO-F1 ATAS TEKNIK ANDAL DENGAN SUPPORT MEMADAI")
    print("=" * 66)
    models = sorted({k.split("/")[1] for k in results})
    for sname in splits:
        subset = adequate[sname]
        if not subset:
            print(f"\n--- {sname}: tidak ada teknik andal dengan support cukup ---")
            continue
        print(f"\n--- {sname} (n_teknik={len(subset)}) ---")
        rows = []
        for m in models:
            best, bk = None, None
            for loss in ("bce", "asl"):
                k = f"{sname}/{m}/{loss}"
                if k in results:
                    v = results[k]["macro_f1"]["mean"]
                    if best is None or v > best:
                        best, bk = v, k
            if bk is None:
                continue
            pc = results[bk]["per_class_f1"]
            vals = [pc[t]["mean"] for t in subset if t in pc]
            rows.append((m, float(np.mean(vals)) if vals else float("nan")))
        for m, v in sorted(rows, key=lambda x: -x[1]):
            print(f"  {m:22s} {v:.4f}")

    print("\n" + "=" * 66)
    print("BACA: bandingkan peringkat di sini dengan tabel subset andal penuh.")
    print("Bila inversi pada family split HILANG setelah teknik ber-support")
    print("rendah dikeluarkan, maka inversi itu artefak sampel kecil -- JANGAN")
    print("dijadikan klaim. Bila TETAP, inversi itu nyata dan layak dibahas.")


if __name__ == "__main__":
    main()
