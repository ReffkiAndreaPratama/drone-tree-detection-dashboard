# DroneVision Dashboard — Analisis Citra Drone & Visualisasi Deteksi Pohon (Sesi 5)

## 1. Identitas Proyek

| Item | Keterangan |
|---|---|
| Nama Proyek | DroneVision Dashboard |
| Sesi | 5 dari 6 |
| Judul Penelitian | Perancangan Dashboard Visualisasi Hasil Deteksi dan Perhitungan Pohon Menggunakan Citra Drone dan Deep Learning |
| Jenis | Dashboard analitik citra + estimasi jumlah pohon |

## 2. Latar Belakang

Pemantauan jumlah pohon di area luas memerlukan alat bantu visual yang mudah dipahami.
Proyek ini menggabungkan analisis citra dengan prinsip Human-Computer Interaction (HCI)
untuk menampilkan hasil deteksi/perhitungan pohon secara jelas dan dapat diekspor.

## 3. Tujuan

1. Menampilkan citra asli dan hasil analisis.
2. Menyediakan pipeline deteksi/estimasi pohon.
3. Menampilkan jumlah objek dari hasil deteksi model.
4. Menyediakan dashboard statistik & tabel hasil per gambar.
5. Mengekspor hasil ke CSV dan citra anotasi.
6. Mengevaluasi model bila anotasi ground truth tersedia.

## 4. Teknologi

Python, OpenCV, NumPy, Pandas, Streamlit, Plotly, pytest.
(opsional) ultralytics/PyTorch untuk mode YOLO dengan bobot kustom.

## 5. Sumber Dataset & Model

- **Mode klasik (default):** estimasi kanopi pohon berbasis indeks kelebihan hijau (ExG) +
  watershed. Tidak memerlukan bobot model.
- **Mode YOLO:** menerima **bobot model deteksi objek kustom** yang dilatih pada kelas pohon.
  Bobot tidak disertakan; pengguna menyediakan sendiri via `src/detector.deteksi_yolo()`.
- **Demo:** citra drone sintetik (`src/detector.buat_citra_drone_demo`) untuk uji pipeline.

> **Catatan penting:** mode klasik hanya **memperkirakan** kanopi pohon dominan hijau, bukan
> deteksi pohon terverifikasi. Jangan menggunakan model generik dan mengklaim deteksi pohon
> akurat tanpa verifikasi kelas model. Akurasi sebenarnya hanya dapat diukur bila anotasi
> ground truth tersedia.

## 6. Struktur Folder

```
Sesi-5-Drone-Image-Analysis-Dashboard/
├── app.py
├── requirements.txt
├── src/
│   ├── detector.py            # Deteksi klasik (ExG+watershed), YOLO, demo, anotasi
│   └── analytics.py           # Statistik batch, luas area, evaluasi ground truth
├── scripts/demo_detect.py
├── tests/
├── data/
├── models/                    # tempat bobot YOLO kustom (tidak disertakan)
└── outputs/                   # citra anotasi & CSV hasil
```

## 7. Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Cara Menjalankan

```bash
python scripts/demo_detect.py   # contoh deteksi pada citra demo
streamlit run app.py            # dashboard
```

## 9. Cara Melatih / Menyiapkan Model Deteksi

- Mode klasik tidak perlu pelatihan.
- Untuk mode YOLO: latih model deteksi objek (mis. YOLOv8) pada dataset pohon berlabel,
  simpan bobot `.pt`, lalu panggil `deteksi_yolo(citra, "models/bobot.pt")`.
  Contoh dataset publik dengan anotasi pohon dapat diperoleh dari platform open dataset.

## 10. Cara Evaluasi

Bila jumlah pohon ground truth tersedia, masukkan jumlah prediksi & aktual pada tab
"Evaluasi" untuk menghitung MAE, RMSE, dan bias.

## 11. Contoh Hasil (mode klasik, citra demo sintetik)

| Berkas | Perkiraan Pohon | Luas Kanopi (piksel) |
|---|---|---|
| demo_1.png | 14 | 36.318 |
| demo_2.png | 16 | 55.658 |
| demo_3.png | 21 | 70.198 |

> Nilai ini berasal dari citra sintetik; hanya menunjukkan pipeline berfungsi.

## 12. Prinsip HCI yang Diterapkan

- Hierarki visual jelas (judul → metrik → grafik → detail).
- Feedback saat proses (spinner/progress bar).
- Empty state informatif pada setiap tab.
- Konsistensi tata letak (tab & kolom).
- Grafik berlabel jelas sesuai kebutuhan.
- Tidak menampilkan statistik yang tidak berasal dari data.

## 13. Keterbatasan

- Estimasi klasik sensitif terhadap ambang, latar, dan kualitas citra.
- Tidak otomatis sama dengan deteksi pohon akurat tanpa model terlatih khusus.
- Evaluasi bermakna memerlukan anotasi ground truth.

## 14. Pengujian

```bash
python -m pytest tests -q
```

## 15. Referensi

- ExG / indeks vegetasi (Woebbecke et al., 1995).
- Watershed segmentation (Beucher & Meyer, 1993).
- HCI: prinsip visualisasi data (Few, 2012); YOLO (Redmon et al., 2016).
