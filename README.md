# Prediktor Unggas — Streamlit

Aplikasi prediksi tiga model dari analisis Prima System:

| Tab | Model | Keluaran |
|---|---|---|
| Grade C (chick-in) | regresi logistik | probabilitas + kategori + faktor penentu |
| Peringatan dini (≤21 hari) | HistGradientBoosting | probabilitas + kategori |
| Mortalitas (nekropsi) | regresi logistik | probabilitas mortalitas >5% (alat audit, bukan keputusan) |

## Menjalankan

```bash
cd /opt/data/research/unggas-tren/app
/opt/data/research/sandbox/.venv/bin/streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Buka http://127.0.0.1:8501. Dari komputer lain, gunakan SSH tunnel:

```bash
ssh -N -L 8501:127.0.0.1:8501 <user>@<server>
```

lalu buka http://localhost:8501.

## Berkas

- `app.py` — antarmuka dan logika prediksi.
- `schema.json` — daftar opsi kategori, default, dan ambang kategori dari distribusi out-of-fold.
- `../models/*.joblib` — model terlatih (A, B, C).
- `../reports/training_results.json` — metrik yang ditampilkan di tab "Tentang model".

## Catatan penting

- Model A belum terkalibrasi: gunakan kategori/peringkat, bukan angka probabilitas mentah.
- Model C lemah (AUC 0,645) dan hanya untuk memilih kasus audit.
- Semua prediksi bersifat **prioritas verifikasi lapangan**, bukan keputusan otomatis.
- Aplikasi ini tidak punya autentikasi. Jangan bind ke `0.0.0.0` tanpa reverse proxy berpassword.

## Uji cepat

```bash
curl -s http://127.0.0.1:8501/_stcore/health   # harus "ok"
```
