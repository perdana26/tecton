"""
Analisis hasil validasi label -> angka & teks untuk paper
=========================================================

Membaca lembar anotasi terisi, menghitung metrik yang dapat dilaporkan, dan
mengestimasi DAMPAK pemetaan bermasalah pada label space.

Yang dihasilkan:
  - Mapping precision: % pemetaan yang dinilai 'correct'
  - Sample-weighted precision: % SAMPEL yang labelnya berasal dari pemetaan
    'correct' (lebih penting -- pemetaan buruk pada teknik besar lebih merusak
    daripada pada teknik kecil)
  - Per-teknik: teknik mana yang labelnya paling tidak dapat dipercaya
  - Draf paragraf Threats to Validity siap sunting

Jalankan:
    python analyze_validation.py E:\multi_osint\label_validation.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Pemakaian: python analyze_validation.py <label_validation.csv>")
        sys.exit(1)
    path = Path(sys.argv[1])

    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    filled = [r for r in rows if r.get("verdict", "").strip()]
    if not filled:
        print("Belum ada anotasi terisi di kolom 'verdict'.")
        sys.exit(1)

    print(f"Dianotasi: {len(filled)}/{len(rows)} pemetaan\n")

    # --- metrik agregat ---
    by_verdict = defaultdict(list)
    for r in filled:
        by_verdict[r["verdict"].strip().lower()].append(r)

    n = len(filled)
    n_samples_total = sum(int(r["n_samples"]) for r in filled)
    print("DISTRIBUSI VERDICT")
    print(f"{'verdict':12s} {'#pemetaan':>10s} {'%':>7s} {'#sampel':>10s} {'%sampel':>9s}")
    print("-" * 52)
    for v in ("correct", "overbroad", "incorrect", "unclear"):
        items = by_verdict.get(v, [])
        ns = sum(int(r["n_samples"]) for r in items)
        print(f"{v:12s} {len(items):10d} {100*len(items)/n:6.1f}% "
              f"{ns:10d} {100*ns/max(n_samples_total,1):8.1f}%")

    n_correct = len(by_verdict.get("correct", []))
    s_correct = sum(int(r["n_samples"]) for r in by_verdict.get("correct", []))
    map_prec = n_correct / n
    samp_prec = s_correct / max(n_samples_total, 1)
    # 'overbroad' dihitung setengah-benar dalam metrik longgar
    n_ob = len(by_verdict.get("overbroad", []))
    s_ob = sum(int(r["n_samples"]) for r in by_verdict.get("overbroad", []))
    map_prec_lenient = (n_correct + 0.5 * n_ob) / n
    samp_prec_lenient = (s_correct + 0.5 * s_ob) / max(n_samples_total, 1)

    print(f"\nMapping precision (ketat)        : {map_prec:.3f}")
    print(f"Mapping precision (longgar)      : {map_prec_lenient:.3f}")
    print(f"Sample-weighted precision (ketat): {samp_prec:.3f}")
    print(f"Sample-weighted precision (long.): {samp_prec_lenient:.3f}")

    # --- per teknik ---
    per_tech = defaultdict(lambda: {"correct": 0, "total": 0, "s_ok": 0, "s_all": 0})
    for r in filled:
        t = r["technique"]; ns = int(r["n_samples"])
        per_tech[t]["total"] += 1
        per_tech[t]["s_all"] += ns
        if r["verdict"].strip().lower() == "correct":
            per_tech[t]["correct"] += 1
            per_tech[t]["s_ok"] += ns

    print("\nTEKNIK DENGAN LABEL PALING MERAGUKAN (sample-weighted precision):")
    ranked = sorted(per_tech.items(),
                    key=lambda x: x[1]["s_ok"] / max(x[1]["s_all"], 1))
    for t, d in ranked[:10]:
        p = d["s_ok"] / max(d["s_all"], 1)
        print(f"  {t}  precision={p:.2f}  ({d['correct']}/{d['total']} pemetaan, "
              f"{d['s_all']} sampel)")

    # --- daftar pemetaan bermasalah ---
    print("\nPEMETAAN BERMASALAH (incorrect / overbroad, urut dampak):")
    bad = by_verdict.get("incorrect", []) + by_verdict.get("overbroad", [])
    for r in sorted(bad, key=lambda r: -int(r["n_samples"]))[:15]:
        print(f"  [{r['verdict']:9s}] {r['technique']} <- {r['signature']:35s} "
              f"({r['n_samples']} sampel)  {r.get('notes','')[:40]}")

    # --- draf paragraf ---
    print("\n" + "=" * 70)
    print("DRAF PARAGRAF THREATS TO VALIDITY (sunting sesuai gaya Anda)")
    print("=" * 70)
    print(f"""
Label quality. Ground-truth ATT&CK labels are derived from CAPEv2 signature
mappings rather than manual expert curation. To assess their reliability, we
validated all {n} unique signature-to-technique mappings against the official
MITRE ATT&CK technique definitions. A single annotator assessed each mapping;
because a second independent annotator was unavailable, we report precision
against an external reference rather than inter-rater agreement. Of the
mappings examined, {100*map_prec:.1f}% were judged consistent with the official
definition, {100*len(by_verdict.get('overbroad',[]))/n:.1f}% overbroad (the
signature captures behaviour wider than the technique), and
{100*len(by_verdict.get('incorrect',[]))/n:.1f}% inconsistent. Weighting by the
number of affected samples, {100*samp_prec:.1f}% of technique assignments derive
from mappings judged correct. We additionally removed T1116, whose signature
(invalid authenticode) fired on 93% of samples and carries minimal adversarial
signal. Residual label noise means reported absolute scores should be read as
performance against signature-derived labels rather than against expert-curated
ground truth; relative comparisons between models, which share identical labels,
are unaffected.
""")
    print("Catatan: ganti angka bila anotasi belum lengkap.")


if __name__ == "__main__":
    main()
