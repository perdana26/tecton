"""
Isi metadata komposisi fold ke family_folds.json — TANPA pelatihan
==================================================================

JS shift, jumlah famili hold-out, ukuran val/test, subset teknik terverifikasi,
dan flag val_carved_from_train semuanya ditentukan oleh SPLIT, bukan oleh model.
Jadi semuanya bisa dihitung ulang dalam hitungan detik untuk kelima fold, tanpa
melatih apa pun.

Dipakai setelah memulihkan family_folds.json versi lama, yang belum punya
field-field ini.

Script ini TIDAK PERNAH menyentuh nilai metrik (macro_f1_*, per_class_f1).
Ia hanya menambahkan field metadata, dan memverifikasi bahwa split yang
dihasilkan sekarang identik dengan yang tersimpan.

Jalankan:
    python backfill_fold_meta.py dataset_v2.npz label_validation.csv results/family_folds.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from fast_dataset import FastDataset
from splits import family_disjoint_split, split_diagnostics

FOLD_SEEDS = [42, 7, 13, 99, 2024]
MIN_SUPPORT = 30


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
        print("Pemakaian: python backfill_fold_meta.py <dataset.npz> "
              "<label_validation.csv> <family_folds.json>")
        sys.exit(1)

    ds = FastDataset(sys.argv[1])
    reliable = load_reliable(Path(sys.argv[2]))
    res_path = Path(sys.argv[3])
    results = json.loads(res_path.read_text())
    techs = list(ds.techniques)
    fams = ds.families()

    print(f"\n{len(results)} entri dimuat dari {res_path}")
    print(f"{len(reliable)} teknik terverifikasi: {sorted(reliable)}\n")

    Yfloat = ds.label.astype(float)
    n_updated, mismatches = 0, []

    print(f"{'fold':>8s} {'n_test':>7s} {'n_val':>6s} {'famili':>7s} "
          f"{'JS':>7s} {'subset':>7s} {'val<-train':>11s}")
    print("-" * 60)

    for fold in FOLD_SEEDS:
        split = family_disjoint_split(fams, seed=fold)
        held = [str(f) for f in split.meta.get("test_families", [])]
        diag = split_diagnostics(split, Yfloat, techs)
        Yte = ds.label[split.test]
        subset = [str(t) for k, t in enumerate(techs)
                  if t in reliable and Yte[:, k].sum() >= MIN_SUPPORT]

        meta = {
            "js_shift": float(diag["label_prevalence_shift_js"]),
            "n_held_out_families": len(held),
            "held_out_families": held,
            "n_test": int(len(split.test)),
            "n_val": int(len(split.val)),
            "subset": subset,
            "val_carved_from_train": split.meta.get("val_carved_from_train"),
            "techniques_unseen_in_train": diag["techniques_unseen_in_train"],
        }

        print(f"{'fold'+str(fold):>8s} {meta['n_test']:7d} {meta['n_val']:6d} "
              f"{meta['n_held_out_families']:7d} {meta['js_shift']:7.3f} "
              f"{len(subset):7d} {str(meta['val_carved_from_train']):>11s}")

        # --- verifikasi: split yang dihitung sekarang harus sama dgn tersimpan
        for key in [k for k in results if k.startswith(f"fold{fold}/")]:
            e = results[key]
            for field, tol in (("n_test", 0), ("subset", None)):
                if field not in e:
                    continue
                old, new = e[field], meta[field]
                if field == "subset":
                    if sorted(map(str, old)) != sorted(new):
                        mismatches.append((key, "subset", old, new))
                elif int(old) != int(new):
                    mismatches.append((key, "n_test", old, new))
            # tulis metadata; JANGAN sentuh metrik
            for k, v in meta.items():
                e[k] = v
            n_updated += 1

    if mismatches:
        print("\n!! SPLIT TIDAK COCOK dengan yang tersimpan:")
        for key, field, old, new in mismatches[:10]:
            print(f"   {key}  {field}: tersimpan={old}  dihitung-ulang={new}")
        print("   Hasil lama dibuat dengan split berbeda. JANGAN gabungkan;")
        print("   metadata tidak ditulis.")
        sys.exit(1)

    backup = res_path.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(json.dumps(json.loads(res_path.read_text()), indent=2))
        print(f"\ncadangan -> {backup}")
    res_path.write_text(json.dumps(results, indent=2))
    print(f"{n_updated} entri diperbarui (metadata saja) -> {res_path}")
    print("\nSplit terverifikasi identik dengan yang tersimpan. "
          "Nilai metrik tidak disentuh.")


if __name__ == "__main__":
    main()
