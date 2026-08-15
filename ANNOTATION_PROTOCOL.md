# Protokol Anotasi — Validasi Pemetaan Signature ke Teknik ATT&CK

**Untuk anotator kedua.** Perkiraan waktu: 60–90 menit untuk 20 item.

---

## Tujuan

Kami memakai label teknik ATT&CK yang dihasilkan otomatis oleh signature sandbox CAPEv2. Tugas Anda adalah menilai, untuk tiap pemetaan, apakah signature tersebut secara wajar mengindikasikan teknik yang dipetakan **menurut definisi resmi MITRE** — bukan menurut intuisi atau kebiasaan praktik.

Penilaian Anda akan dibandingkan dengan penilaian anotator pertama untuk menghitung Cohen's κ.

## Aturan penting

**Jangan berdiskusi dengan anotator pertama sebelum Anda selesai.** Independensi adalah syarat sahnya κ. Diskusi penyelesaian dilakukan setelah kedua lembar terisi.

**Nilai terhadap definisi resmi.** Buka tautan di kolom `attack_url` untuk tiap item dan baca dua paragraf deskripsi utama. Kalau definisi menyebut mekanisme spesifik (API, utility, command), itu petunjuk terkuat.

**Nilai pemetaannya, bukan sampelnya.** Pertanyaannya bukan "apakah sampel ini benar-benar melakukan teknik itu", melainkan "apakah signature ini, ketika menyala, secara wajar menunjukkan teknik ini".

---

## Codebook

### `correct`
Signature menangkap perilaku yang sesuai definisi teknik. Mekanisme dan maksud selaras.

> Contoh: `enumerates_running_processes` → T1057 Process Discovery. Definisi T1057 secara harfiah adalah memperoleh informasi tentang proses yang berjalan.

### `overbroad`
Signature memang menyala pada kasus teknik ini, tetapi juga pada kasus yang bukan teknik ini. Teknik terdeteksi, tapi disertai kasus lain.

> Contoh: `antivm_vbox_files` → T1083 File and Directory Discovery. Sampel memang melakukan enumerasi file, sehingga secara mekanis cocok; tetapi maksudnya deteksi sandbox, bukan pengintaian sistem berkas. Signature ini akan menyala pada evasion yang bukan discovery.

Gunakan kategori ini juga bila nama signature terlalu generik untuk menunjukkan teknik secara spesifik.

### `incorrect`
Signature tidak berhubungan dengan teknik yang dipetakan.

> Contoh: `bypass_firewall` → T1031 Modify Existing Service. Melewati firewall tidak melibatkan modifikasi service Windows; di ATT&CK saat ini perilaku itu masuk T1562.004.

### `unclear`
Tidak cukup informasi untuk menilai — nama signature tidak menjelaskan perilaku, dan definisi teknik tidak memberi petunjuk yang cukup.

Gunakan sehemat mungkin. Bila Anda ragu antara `overbroad` dan `incorrect`, pilih salah satunya dan jelaskan keraguan di kolom `notes`.

---

## Pola yang mungkin Anda temui

Selama anotasi pertama, satu pola berulang: signature menangkap **mekanisme** dengan benar, tetapi teknik yang dipetakan mengkodekan **maksud** yang berbeda. CAPE memetakan berdasarkan apa yang dilakukan sampel, sementara ATT&CK mengkategorikan berdasarkan tujuan adversarial.

Anda tidak wajib setuju dengan pembacaan ini. Kalau menurut Anda kecocokan mekanis sudah memadai, nilai `correct` — perbedaan penilaian semacam inilah yang justru diukur κ.

Pola kedua: satu signature dipetakan ke beberapa teknik sekaligus. Nilai tiap baris secara terpisah; sebuah signature bisa `correct` untuk satu teknik dan `incorrect` untuk teknik lain.

---

## Cara mengisi

1. Buka `kappa_sheet_annotator2.csv` di Excel
2. Untuk tiap baris, buka `attack_url`, baca definisi
3. Isi kolom `verdict` dengan salah satu: `correct`, `overbroad`, `incorrect`, `unclear`
4. Isi `notes` bila ada pertimbangan yang perlu dicatat (opsional, tapi berguna saat diskusi penyelesaian)
5. Simpan sebagai CSV

Setelah selesai, jalankan:

```
python compute_kappa.py label_validation.csv kappa_sheet_annotator2.csv
```

---

## Catatan tentang pelaporan

Dengan n = 20, interval kepercayaan κ akan lebar. Ini konsekuensi ukuran sampel, bukan kelemahan protokol — dan harus dilaporkan apa adanya. Melaporkan κ tanpa CI pada n sekecil ini akan mengundang keberatan reviewer.

Sebagai pembanding, MAMBA (Huang et al., TDSC 2022) melaporkan κ = 0.739 untuk ekstraksi resource ATT&CK mereka, yang dikategorikan *substantial*.

Kami melaporkan κ **pra-diskusi**. Ketidaksepakatan diselesaikan setelahnya, dan label final digunakan untuk analisis, tetapi angka kesepakatan yang dilaporkan adalah sebelum penyelesaian — itu praktik standar.
