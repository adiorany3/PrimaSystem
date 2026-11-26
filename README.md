# Prediktor Unggas — Streamlit

Aplikasi dua model dari analisis Prima System (model mortalitas dikecualikan karena AUC 0,645):

| Tab | Model | Keluaran |
|---|---|---|
| Grade C (chick-in) | regresi logistik + kalibrasi Platt | probabilitas terkalibrasi + keputusan pada ambang biaya + faktor penentu |
| Peringatan dini (≤21 hari) | HistGradientBoosting | probabilitas + kategori risiko |
| Tentang model | — | metrik out-of-fold, skenario biaya, cara pakai |

## Angka yang dipakai

Model A (Platt, 5-fold grouped by farm): AUC 0,929 · PR-AUC 0,243 · Brier 0,021 · ECE 0,007.
Uji temporal 2025→2026: AUC 0,898 · Brier 0,013.

Ambang keputusan = 1/(1+R), R = biaya gagal panen : biaya kunjungan.

| R | Ambang | Flock ditandai | Grade C tertangkap | Kunjungan/kasus |
|---:|---:|---:|---:|---:|
| 10 | 9,1% | 1.376 | 197 (77%) | 7,0 |
| 20 | 4,8% | 1.782 | 237 (93%) | 7,5 |
| 50 | 2,0% | 2.203 | 252 (98%) | 8,7 |

Model B: AUC 0,833 · PR-AUC 0,251 (baseline 2,7%) · Brier 0,023; kategori sedang ≥2%, tinggi ≥10%.

## Menjalankan

```bash
cd "/Users/macbookpro/Downloads/Telegram Desktop/Prediktor-Unggas-PrimaSystem/app"
python3 -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501
```

atau `./run_app.sh` (PORT/HOST bisa di-override). Buka http://127.0.0.1:8501.
Dari komputer lain gunakan SSH tunnel:

```bash
ssh -N -L 8501:127.0.0.1:8501 <user>@<server>
```

## Perilaku form

- Pilihan **Lokasi** mengikuti **Area**, **Plant** mengikuti **Area**, **Customer Name** mengikuti **Customer Group**, **Sub Customer** mengikuti **Customer Name** (peta di `schema.json`: `localities`, `plant_map`, `cust_map`, `sub_map`).
- Tab chick-in menghitung ulang setiap input berubah (tanpa tombol submit); tab peringatan dini memakai tombol **Hitung risiko**.
- Prediksi massal lewat CSV tersedia di expander; unduh template dulu agar kolomnya cocok.

## Berkas

- `app.py` — antarmuka dan logika prediksi.
- `schema.json` — opsi kategori, default, peta kaskade, ambang, dan metrik kalibrasi.
- `models/grade_placement_calibrated.joblib` — model A produksi (fallback ke `grade_placement.joblib` bila tidak ada).
- `models/grade_early.joblib` — model B.
- `../reports/training_results.json` — sumber metrik yang dibaca aplikasi.

## Uji

```bash
cd /Users/macbookpro/Downloads/Telegram Desktop/Prediktor-Unggas-PrimaSystem
python3 scripts/test_app.py   # AppTest: semua tab, kaskade, kasus risiko tinggi
curl -s http://127.0.0.1:8501/_stcore/health                       # harus "ok"
```

## Catatan penting

- Semua prediksi bersifat **prioritas verifikasi lapangan**, bukan keputusan otomatis.
- Probabilitas A boleh dibaca sebagai peluang (terkalibrasi), tetapi ambangnya keputusan manajemen.
- PR-AUC turun dari 0,291 (pra-kalibrasi) ke 0,243 — konsekuensi kalibrasi, bukan penurunan mutu peringkat.
- Aplikasi ini tidak punya autentikasi. Jangan bind ke `0.0.0.0` tanpa reverse proxy berpassword.
