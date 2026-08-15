"""
Cohen's kappa parsial -- anotator 1 vs anotator 2
================================================

Menghitung kesepakatan antar dua penilai independen atas 20 pemetaan
berdampak tertinggi, dan mencetak teks siap sunting untuk paper.

Yang dilaporkan:
  - Kesepakatan mentah (persentase)
  - Cohen's kappa (tak berbobot)
  - Kappa berbobot linier -- correct/overbroad/incorrect bersifat ordinal,
    sehingga ketidaksepakatan correct<->overbroad lebih ringan daripada
    correct<->incorrect. Kappa berbobot mencerminkan ini.
  - CI 95% bootstrap (dengan n=20, CI akan lebar -- laporkan apa adanya)
  - Matriks konfusi & daftar item yang tidak disepakati

Jalankan:
    python compute_kappa.py E:\\multi_osint\\label_validation.csv ^
                            E:\\multi_osint\\kappa_sheet_annotator2.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CATS = ["correct", "overbroad", "incorrect", "unclear"]
# Jarak ordinal untuk kappa berbobot. 'unclear' diperlakukan maksimal
# berbeda dari ketiganya karena bukan titik pada skala yang sama.
ORDINAL = {"correct": 0, "overbroad": 1, "incorrect": 2}


def load(path: Path, key_cols=("technique", "signature")):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = {}
    for r in rows:
        v = (r.get("verdict") or "").strip().lower()
        if not v:
            continue
        out[tuple(r[c].strip() for c in key_cols)] = v
    return out


def weight_matrix():
    """Bobot ketidaksepakatan linier atas skala ordinal."""
    n = len(CATS)
    W = np.zeros((n, n))
    for i, a in enumerate(CATS):
        for j, b in enumerate(CATS):
            if a == b:
                W[i, j] = 0.0
            elif a in ORDINAL and b in ORDINAL:
                W[i, j] = abs(ORDINAL[a] - ORDINAL[b]) / 2.0
            else:
                W[i, j] = 1.0        # apa pun vs 'unclear'
    return W


def kappa(a, b, weighted=False):
    idx = {c: i for i, c in enumerate(CATS)}
    n = len(a)
    O = np.zeros((len(CATS), len(CATS)))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= n
    pa = O.sum(axis=1)
    pb = O.sum(axis=0)
    E = np.outer(pa, pb)
    if weighted:
        W = weight_matrix()
        po = 1 - (W * O).sum()
        pe = 1 - (W * E).sum()
    else:
        po = np.trace(O)
        pe = (pa * pb).sum()
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan"), po


def bootstrap_ci(a, b, weighted=False, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        aa = [a[i] for i in s]
        bb = [b[i] for i in s]
        if len(set(aa)) < 2 or len(set(bb)) < 2:
            continue
        k, _ = kappa(aa, bb, weighted)
        if not np.isnan(k):
            vals.append(k)
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def interpret(k):
    if k < 0.0:   return "poor"
    if k <= 0.20: return "slight"
    if k <= 0.40: return "fair"
    if k <= 0.60: return "moderate"
    if k <= 0.80: return "substantial"
    return "almost perfect"


def main():
    if len(sys.argv) < 3:
        print("Pemakaian: python compute_kappa.py <anotator1.csv> <anotator2.csv>")
        sys.exit(1)
    a1 = load(Path(sys.argv[1]))
    a2 = load(Path(sys.argv[2]))

    shared = sorted(set(a1) & set(a2))
    if not shared:
        print("Tidak ada item yang dinilai kedua anotator.")
        sys.exit(1)
    A = [a1[k] for k in shared]
    B = [a2[k] for k in shared]

    bad = {v for v in A + B if v not in CATS}
    if bad:
        print(f"Nilai verdict tak dikenal: {bad}. Gunakan: {CATS}")
        sys.exit(1)

    n = len(shared)
    k_unw, po = kappa(A, B, weighted=False)
    k_w, po_w = kappa(A, B, weighted=True)
    ci_unw = bootstrap_ci(A, B, weighted=False)
    ci_w = bootstrap_ci(A, B, weighted=True)

    print("=" * 64)
    print(f"COHEN'S KAPPA PARSIAL  (n = {n} pemetaan)")
    print("=" * 64)
    print(f"  Kesepakatan mentah      : {100*po:.1f}%")
    print(f"  Kappa tak berbobot      : {k_unw:.3f}  "
          f"[{ci_unw[0]:.3f}, {ci_unw[1]:.3f}]  ({interpret(k_unw)})")
    print(f"  Kappa berbobot linier   : {k_w:.3f}  "
          f"[{ci_w[0]:.3f}, {ci_w[1]:.3f}]  ({interpret(k_w)})")

    print("\nMATRIKS KONFUSI (baris = anotator 1, kolom = anotator 2)")
    idx = {c: i for i, c in enumerate(CATS)}
    M = np.zeros((len(CATS), len(CATS)), dtype=int)
    for x, y in zip(A, B):
        M[idx[x], idx[y]] += 1
    print(f"{'':12s}" + "".join(f"{c[:9]:>11s}" for c in CATS))
    for i, c in enumerate(CATS):
        print(f"{c:12s}" + "".join(f"{M[i,j]:11d}" for j in range(len(CATS))))

    dis = [(k, a1[k], a2[k]) for k in shared if a1[k] != a2[k]]
    if dis:
        print(f"\nITEM TIDAK DISEPAKATI ({len(dis)}):")
        for (tech, sig), v1, v2 in dis:
            print(f"  {tech:7s} {sig:38s} A1={v1:10s} A2={v2}")
        print("\n  -> selesaikan lewat diskusi; laporkan kappa PRA-diskusi di paper,")
        print("     dan label final pasca-diskusi bila dipakai untuk analisis.")

    print("\n" + "=" * 64)
    print("TEKS SIAP SUNTING UNTUK PAPER")
    print("=" * 64)
    print(f"""
To assess the reliability of these judgments, a second annotator
independently assessed the {n} mappings with the largest sample impact,
without access to the first annotator's verdicts. Raw agreement was
{100*po:.1f}\\%, with Cohen's $\\kappa = {k_unw:.3f}$ (95\\% CI
[{ci_unw[0]:.3f}, {ci_unw[1]:.3f}]) and linearly weighted
$\\kappa = {k_w:.3f}$, indicating {interpret(k_unw)} agreement.
Disagreements concentrated on the boundary between \\emph{{correct}} and
\\emph{{overbroad}}, reflecting genuine ambiguity in how narrowly a
technique definition should be read rather than disagreement about
observed behaviour. Remaining cases were resolved through discussion;
we report pre-discussion agreement above.
""")
    print("Catatan: dengan n kecil, CI akan lebar. Laporkan apa adanya --")
    print("melaporkan kappa tanpa CI pada n=20 akan mengundang keberatan.")


if __name__ == "__main__":
    main()
