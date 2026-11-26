# Prediktor Unggas — Prima System (model A & B)

Aplikasi Streamlit untuk dua model yang direkomendasikan laporan:

| Tab | Model | Keluaran |
|---|---|---|
| Grade C (chick-in) | regresi logistik | probabilitas + kategori + faktor penentu |
| Peringatan dini (≤21 hari) | HistGradientBoosting | probabilitas + kategori |

Model mortalitas dari skor lesi nekropsi **tidak dipakai**: AUC 0,645 dan tidak layak untuk keputusan operasional.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py            # http://127.0.0.1:8501
```

Akses dari komputer lain pakai SSH tunnel:

```bash
ssh -N -L 8501:127.0.0.1:8501 <user>@<server>
```

lalu buka http://localhost:8501.

## Berkas

- `app.py` — antarmuka dan logika prediksi.
- `schema.json` — opsi kategori, default, dan ambang risiko dari distribusi out-of-fold.
- `models/grade_placement.joblib`, `models/grade_early.joblib` — model A dan B.
- `training_results.json` — metrik yang ditampilkan di tab "Tentang model".

## Insight & pencegahan

Tab **Insight & pencegahan** merangkum kapasitas prioritas 5% dari metrik validasi, tindakan tindak lanjut menurut kategori risiko, pencegahan per fase pemeliharaan, serta kualitas data dan batasan model. Angka metrik dibaca langsung dari `training_results.json`; panduan budidaya tetap perlu disesuaikan dengan strain, kondisi kandang, serta arahan dokter hewan.

## Pedoman penggunaan insight

- Ambang model A adalah persentil data latih; jangan tafsirkan sebagai probabilitas klinis.
- Faktor model bersifat asosiasi, bukan sebab-akibat atau rekomendasi mengubah produk/pelanggan.
- Pantau hasil panen dan kalibrasi berkala; hitung ulang ambang setiap pelatihan ulang.

## Catatan penting

- Model A belum terkalibrasi: gunakan kategori/peringkat, bukan angka probabilitas mentah.
- Model B terkalibrasi (Brier 0,023); ambang kategori absolut 2% dan 10%.
- Semua prediksi adalah **prioritas verifikasi lapangan**, bukan keputusan otomatis.
- Aplikasi ini tidak punya autentikasi. Jangan bind ke `0.0.0.0` tanpa reverse proxy berpassword.

## Uji cepat

```bash
curl -s http://127.0.0.1:8501/_stcore/health   # harus "ok"
```
