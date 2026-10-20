"""DroneVision Dashboard - Antarmuka Streamlit (Sesi 5).

Prinsip HCI yang diterapkan:
- Hierarki visual jelas (judul, subjudul, metrik utama).
- Feedback saat proses berlangsung (spinner/progress).
- Empty state informatif.
- Konsistensi tata letak (tab, kolom).
- Grafik dengan label yang mudah dipahami.
- Tidak menampilkan statistik yang tidak berasal dari data.

Jalankan:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    hitung_area_hektar,
    ringkasan_batch,
    statistik_ringkas,
)
from src.detector import (  # noqa: E402
    buat_citra_drone_demo,
    deteksi_klasik,
    gambar_anotasi,
)

st.set_page_config(page_title="DroneVision Dashboard", page_icon="🌳", layout="wide")


def _baca_upload(upload) -> np.ndarray | None:
    data = np.asarray(bytearray(upload.getvalue()), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _deteksi_satu(bgr: np.ndarray, ambang: float, min_luas: int):
    det = deteksi_klasik(bgr, ambang_hijau=ambang, min_luas=min_luas)
    vis = gambar_anotasi(bgr, det)
    return det, vis


def render_single() -> None:
    st.subheader("Analisis Satu Citra Drone")
    upload = st.file_uploader("Unggah citra drone (JPG/PNG)", type=["jpg", "jpeg", "png"])
    use_demo = st.checkbox("Gunakan citra demo sintetik", value=upload is None)

    with st.expander("Pengaturan deteksi"):
        ambang = st.slider("Ambang indeks hijau (ExG)", 0.1, 0.9, 0.4, 0.05)
        min_luas = st.slider("Luas minimum kanopi (piksel)", 10, 500, 40, 10)

    bgr = None
    nama = "-"
    if upload is not None:
        bgr = _baca_upload(upload)
        nama = upload.name
    elif use_demo:
        bgr = buat_citra_drone_demo()
        nama = "demo_sintetik"
        st.info("Mode demo: citra dibuat sintetik, BUKAN foto drone nyata.")

    if bgr is None:
        st.info("Unggah citra drone atau aktifkan mode demo untuk memulai analisis.")
        return

    with st.spinner("Mendeteksi objek..."):
        det, vis = _deteksi_satu(bgr, ambang, min_luas)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Citra Asli**")
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("**Hasil Deteksi**")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("**Statistik Deteksi**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Perkiraan Jumlah Pohon", det.jumlah_pohon)
    c2.metric("Luas Kanopi (piksel)", f"{det.luas_kanopi_piksel:,}")
    c3.metric("Resolusi Citra", f"{det.lebar}×{det.tinggi}")

    gsd = st.number_input("Ground Sampling Distance (cm/piksel)", 0.0, value=0.0, step=0.5,
                          help="Isi bila diketahui untuk menghitung luas area (hektar).")
    area = hitung_area_hektar(det.luas_kanopi_piksel, gsd)
    if area is not None:
        st.metric("Perkiraan Luas Kanopi (hektar)", f"{area:.4f}")

    png = cv2.imencode(".png", vis)[1].tobytes()
    st.download_button("Unduh Citra Anotasi (PNG)", data=png, file_name=f"anotasi_{nama}.png",
                       mime="image/png")

    with st.expander("Catatan metode"):
        st.markdown(
            "- Mode ini menggunakan indeks kelebihan hijau (ExG) + watershed untuk "
            "**memperkirakan** kanopi pohon dominan hijau.\n"
            "- Ini **bukan** deteksi pohon terverifikasi; akurasi bergantung pada "
            "ambang, latar belakang, dan kualitas citra.\n"
            "- Untuk hasil akurat, gunakan bobot model deteksi objek kustom yang dilatih "
            "pada kelas pohon (lihat README)."
        )


def render_batch() -> None:
    st.subheader("Analisis Batch")
    uploads = st.file_uploader(
        "Unggah beberapa citra drone", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if not uploads:
        st.info("Unggah beberapa citra untuk analisis batch.")
        return

    hasil = []
    progress = st.progress(0.0)
    for i, u in enumerate(uploads, 1):
        bgr = _baca_upload(u)
        if bgr is None:
            continue
        det = deteksi_klasik(bgr)
        d = det.sebagai_dict()
        d["nama_file"] = u.name
        hasil.append(d)
        progress.progress(i / len(uploads))

    df = ringkasan_batch(hasil)
    st.dataframe(df, use_container_width=True)

    stat = statistik_ringkas(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Gambar", stat["total_gambar"])
    c2.metric("Total Pohon (perkiraan)", stat["total_pohon"])
    c3.metric("Rata-rata/Gambar", f"{stat['rata_pohon']:.1f}")
    c4.metric("Maksimum", stat["max_pohon"])

    if not df.empty:
        fig = px.bar(df, x="nama_file", y="jumlah_pohon",
                     labels={"nama_file": "Berkas", "jumlah_pohon": "Jumlah Pohon (perkiraan)"},
                     title="Jumlah Pohon Terdeteksi per Gambar")
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Unduh Hasil Batch (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_deteksi_batch.csv",
        mime="text/csv",
    )


def render_evaluasi() -> None:
    st.subheader("Evaluasi (Opsional)")
    st.caption("Bandingkan jumlah pohon prediksi dengan anotasi ground truth.")
    st.info(
        "Masukkan jumlah prediksi & aktual per gambar (dipisah koma) untuk menghitung MAE/RMSE."
    )
    col1, col2 = st.columns(2)
    pred_str = col1.text_input("Prediksi", "20, 30, 15")
    akt_str = col2.text_input("Aktual (ground truth)", "22, 28, 18")
    if st.button("Hitung Evaluasi"):
        try:
            pred = [int(x.strip()) for x in pred_str.split(",") if x.strip()]
            akt = [int(x.strip()) for x in akt_str.split(",") if x.strip()]
        except ValueError:
            st.error("Masukkan angka dipisah koma.")
            return
        if len(pred) != len(akt):
            st.error("Jumlah nilai prediksi dan aktual harus sama.")
            return
        from src.analytics import evaluasi_ground_truth

        metrik = evaluasi_ground_truth(pred, akt)
        if metrik:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{metrik['mae']:.2f}")
            c2.metric("RMSE", f"{metrik['rmse']:.2f}")
            c3.metric("Bias", f"{metrik['bias']:+.2f}")


def main() -> None:
    st.title("🌳 DroneVision Dashboard")
    st.caption(
        "Analisis Citra Drone & Visualisasi Deteksi/Perhitungan Pohon - Sesi 5"
    )
    tab1, tab2, tab3 = st.tabs(["Citra Tunggal", "Batch", "Evaluasi"])
    with tab1:
        render_single()
    with tab2:
        render_batch()
    with tab3:
        render_evaluasi()


if __name__ == "__main__":
    main()
