"""
AEGCN — baseline graf (Hsiao & Chu, "Attention-Enhanced Graph Convolution
Network for Malware Family Feature Extraction and Embedding", IEEE TNSM 2025).
Referensi [21] di paper Anda.

Reimplementasi berdasarkan Section III & Eq. (3) paper.

Mengapa AEGCN adalah baseline yang TEPAT untuk Anda:
AEGCN merepresentasikan trace sebagai graf transisi Markov atas API call --
nyaris identik dengan "Transition-GATv2" Anda. Perbedaannya justru tepat pada
klaim Anda:
    AEGCN : fitur node one-hot + Feature Attention + GCN berbobot transisi
    Anda  : embedding API terlatih + GATv2 (atensi dinamis antar node)
Jadi AEGCN mengisolasi kontribusi Anda. Kalau Anda menang atas AEGCN pada
tugas & split yang sama, itu bukti langsung bahwa GATv2 + embedding terlatih
mengalahkan attention-GCN + one-hot. Kalau tidak, Anda belajar sesuatu yang
penting sebelum reviewer yang memberitahunya.

==========================================================================
DEVIASI YANG WAJIB DILAPORKAN DI PAPER
==========================================================================
1. TUGAS: AEGCN asli = klasifikasi FAMILI (multi-class, softmax). Tugas Anda =
   teknik ATT&CK (multi-label, sigmoid). Kami mengganti kepala klasifikasi
   softmax -> sigmoid multi-label dan melatih ulang pada label ATT&CK Anda.
   Nyatakan ini eksplisit. Arsitektur backbone tidak diubah.

2. INTERPRETASI YANG AMBIGU DI PAPER (dokumentasikan pilihan Anda):
   - Feature Attention (FA): paper menulis H^0 = X (x) FA dengan notasi dimensi
     yang ambigu untuk graf berukuran variabel. Kami mengimplementasikan FA
     sebagai atensi per-DIMENSI-FITUR terlatih (vektor D), aman untuk |V| variabel.
   - Adjacency Attention (AA): paper menyetel AA ke matriks transisi Markov lalu
     menyebutnya "trainable". Matriks transisi bersifat per-graf (ukuran variabel),
     sehingga matriks terlatih global tidak bermakna. Kami mengimplementasikan AA
     sebagai pembobotan adjacency oleh probabilitas transisi Markov (edge weight),
     dengan gate skalar terlatih. Ini interpretasi paling setia yang dapat
     digeneralisasi; laporkan sebagai catatan reproduksi.

3. Paper memakai "flatten" -> menuntut |V| tetap. Kami memakai kosakata API
   GLOBAL tetap (327 API Anda), sehingga tiap sampel adalah subgraf pada node
   set yang sama dan flatten valid. Alternatif global-pooling disediakan untuk
   skalabilitas; sebutkan mana yang Anda pakai.
==========================================================================

Butuh: torch. (Adjacency dibangun sebagai dense |V|x|V| dengan |V| = vocab API.)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 1. Graph Generation — Markov model (Section III.C)
# ==========================================================================
def build_markov_graph(api_seq: list[int], n_vocab: int,
                       k_gram: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Bangun matriks transisi Markov atas kosakata API GLOBAL tetap.

    Node = API unik (indeks 0..n_vocab-1). Edge weight = probabilitas transisi
    (hitungan dinormalisasi per baris). Paper menemukan k=1 optimal (Section IV.G);
    k_gram>1 akan memperbesar ruang node dan biasanya tidak membantu di sini.

    Mengembalikan:
      A     [n_vocab, n_vocab]  adjacency biner (1 bila transisi teramati)
      P     [n_vocab, n_vocab]  matriks transisi (edge weight), baris ternormalisasi
    """
    A = np.zeros((n_vocab, n_vocab), dtype=np.float32)
    counts = np.zeros((n_vocab, n_vocab), dtype=np.float32)
    for a, b in zip(api_seq[:-1], api_seq[1:]):
        counts[a, b] += 1.0
        A[a, b] = 1.0
    row = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row, out=np.zeros_like(counts), where=row > 0)
    return A, P


def normalize_adj(A_weighted: torch.Tensor) -> torch.Tensor:
    """Â = D̃^{-1/2} Ã D̃^{-1/2}, dengan self-loop (Eq. 1/3)."""
    I = torch.eye(A_weighted.size(0), device=A_weighted.device)
    A_hat = A_weighted + I
    deg = A_hat.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    D = torch.diag(d_inv_sqrt)
    return D @ A_hat @ D


# ==========================================================================
# 2. AEGCN (Section III.D, Eq. 3)
# ==========================================================================
class AEGCNLayer(nn.Module):
    """Satu lapis konvolusi: H^{l+1} = sigma( Â H^l W^l ).

    Â di sini sudah memuat Adjacency Attention (bobot transisi Markov) dan
    self-loop. sigma = sigmoid, sesuai paper (bukan ReLU).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, A_hat):
        return torch.sigmoid(A_hat @ self.W(H))


class AEGCN(nn.Module):
    """AEGCN backbone + kepala yang dapat ditukar.

    Feature Attention (FA): vektor per-dimensi-fitur terlatih (lihat catatan
    deviasi). Adjacency Attention (AA): pembobotan transisi Markov + gate skalar
    terlatih `aa_gate`.
    """

    def __init__(self, n_vocab: int, n_out: int,
                 k1: int = 128, k2: int = 256, k3: int = 64,
                 dropout: float = 0.35, multilabel: bool = True,
                 pool: str = "flatten"):
        super().__init__()
        self.n_vocab = n_vocab
        self.multilabel = multilabel
        self.pool = pool

        # Fitur node = one-hot identitas API (dim = n_vocab), lihat Section III.C
        d = n_vocab
        # Feature Attention: skala terlatih per dimensi fitur (interpretasi kami)
        self.FA = nn.Parameter(torch.ones(d))
        # Adjacency Attention: gate skalar terlatih atas bobot transisi Markov
        self.aa_gate = nn.Parameter(torch.tensor(1.0))

        self.gc1 = AEGCNLayer(d, k1)
        self.gc2 = AEGCNLayer(k1, k2)
        self.gc3 = AEGCNLayer(k2, k3)
        self.drop = nn.Dropout(dropout)

        if pool == "flatten":
            head_in = n_vocab * k3           # butuh |V| tetap (kosakata global)
        else:                                # "mean" | "max" — lebih skalabel
            head_in = k3
        self.classifier = nn.Sequential(nn.Linear(head_in, 128), nn.ReLU(),
                                        self.drop, nn.Linear(128, n_out))

    def forward(self, X_onehot, A_bin, P_trans):
        """
        X_onehot [V, V]   fitur node one-hot (identitas API)
        A_bin    [V, V]   adjacency biner
        P_trans  [V, V]   matriks transisi Markov (edge weight)
        """
        # Adjacency Attention: bobot transisi (gated) di atas adjacency
        A_weighted = A_bin * (self.aa_gate * P_trans)          # AA (x) A
        A_hat = normalize_adj(A_weighted)

        # Feature Attention: H^0 = X (x) FA
        H = X_onehot * self.FA.unsqueeze(0)
        H = self.gc1(H, A_hat)
        H = self.drop(H)
        H = self.gc2(H, A_hat)
        H = self.drop(H)
        H = self.gc3(H, A_hat)                                  # [V, k3]

        if self.pool == "flatten":
            g = H.reshape(1, -1)                                # [1, V*k3]
        elif self.pool == "max":
            g = H.max(dim=0, keepdim=True).values
        else:
            g = H.mean(dim=0, keepdim=True)
        logits = self.classifier(g)                            # [1, n_out]
        return logits                                          # sigmoid di loss

    def feature_attention(self) -> torch.Tensor:
        """Untuk analisis explainability yang setara dengan Fig. 24 paper."""
        return self.FA.detach()

    # ======================================================================
    # Versi BATCHED — memproses banyak graf sekaligus (jauh lebih cepat)
    # ======================================================================
    def forward_batched(self, A_bin_b, P_trans_b):
        """
        A_bin_b  [B, V, V]  adjacency biner per graf
        P_trans_b[B, V, V]  transisi Markov per graf
        Fitur node one-hot identitas dibangun implisit (X = I), sehingga
        H^0 = I * FA = diag(FA) -> tiap baris i adalah FA. Kita bentuk langsung.

        Semua operasi via batched matmul (bmm) -> memanfaatkan GPU penuh,
        menggantikan loop Python per sampel.
        """
        B = A_bin_b.size(0)
        V = A_bin_b.size(1)
        dev = A_bin_b.device

        # Adjacency Attention + self-loop, per graf
        A_w = A_bin_b * (self.aa_gate * P_trans_b)            # [B,V,V]
        I = torch.eye(V, device=dev).unsqueeze(0)             # [1,V,V]
        A_hat = A_w + I
        deg = A_hat.sum(dim=2).clamp(min=1e-12)               # [B,V]
        d_inv = deg.pow(-0.5)                                 # [B,V]
        # D^-1/2 A D^-1/2  via broadcasting
        A_norm = A_hat * d_inv.unsqueeze(2) * d_inv.unsqueeze(1)  # [B,V,V]

        # H^0 = X (x) FA dengan X = I (one-hot identitas).
        # X * FA element-wise => matriks diagonal diag(FA), BUKAN FA di tiap baris.
        # Baris ke-i = one-hot(i) * FA = vektor nol dengan FA[i] di posisi i.
        H0 = torch.diag(self.FA).unsqueeze(0).expand(B, -1, -1)   # [B,V,V]
        H = torch.sigmoid(torch.bmm(A_norm, self.gc1.W(H0)))  # [B,V,k1]
        H = self.drop(H)
        H = torch.sigmoid(torch.bmm(A_norm, self.gc2.W(H)))   # [B,V,k2]
        H = self.drop(H)
        H = torch.sigmoid(torch.bmm(A_norm, self.gc3.W(H)))   # [B,V,k3]

        if self.pool == "flatten":
            g = H.reshape(B, -1)
        elif self.pool == "max":
            g = H.max(dim=1).values
        else:
            g = H.mean(dim=1)
        return self.classifier(g)                             # [B, n_out]


# ==========================================================================
# 3. Catatan integrasi
# ==========================================================================
INTEGRATION_NOTE = """
Menjalankan AEGCN sebagai baseline pada tugas Anda:

  vocab = 327                       # API Anda setelah filtering
  model = AEGCN(n_vocab=vocab, n_out=17, multilabel=True, pool='flatten')
  loss  = nn.BCEWithLogitsLoss()    # ATAU pakai ASL Anda -- lihat catatan bawah

  Untuk tiap sampel:
    A_bin, P = build_markov_graph(api_seq, n_vocab=vocab, k_gram=1)
    X = torch.eye(vocab)            # one-hot node global
    logits = model(X, torch.tensor(A_bin), torch.tensor(P))

KEADILAN PERBANDINGAN (kritis, kalau tidak reviewer akan menolak):
  - Latih AEGCN dengan LOSS YANG SAMA seperti model Anda. Kalau Anda memakai ASL
    untuk model Anda tetapi BCE untuk AEGCN, kemenangan Anda bisa jadi hanya
    efek loss, bukan arsitektur. Laporkan AEGCN dengan BCE (setia paper) DAN
    dengan ASL (setara kondisi Anda) -- keduanya.
  - Evaluasi pada SPLIT yang sama (random, family, cluster, temporal).
  - Threshold per-kelas yang sama (tune di val).
  - 5 seed, mean +/- std.

Kalau salah satu di atas berbeda antara AEGCN dan model Anda, perbandingannya
terkonfound dan angka kemenangan Anda tidak bisa dipertahankan.
"""
