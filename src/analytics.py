"""Analitik hasil deteksi pohon (Sesi 5).

Menggabungkan hasil deteksi beberapa citra menjadi tabel statistik, dan
(opsional) menghitung metrik bila anotasi ground truth tersedia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ringkasan_batch(hasil: list[dict]) -> pd.DataFrame:
    """Mengubah daftar hasil deteksi menjadi DataFrame ringkas per gambar."""
    rows = []
    for h in hasil:
        rows.append(
            {
                "nama_file": h.get("nama_file", "-"),
                "jumlah_pohon": int(h.get("jumlah_pohon", 0)),
                "luas_kanopi_piksel": int(h.get("luas_kanopi_piksel", 0)),
                "lebar": int(h.get("lebar", 0)),
                "tinggi": int(h.get("tinggi", 0)),
                "mode": h.get("mode", "-"),
            }
        )
    return pd.DataFrame(rows)


def statistik_ringkas(df: pd.DataFrame) -> dict:
    """Statistik ringkas dari tabel hasil batch."""
    if df.empty:
        return {"total_gambar": 0, "total_pohon": 0, "rata_pohon": 0.0,
                "min_pohon": 0, "max_pohon": 0, "median_pohon": 0.0}
    jumlah = df["jumlah_pohon"]
    return {
        "total_gambar": int(len(df)),
        "total_pohon": int(jumlah.sum()),
        "rata_pohon": float(jumlah.mean()),
        "min_pohon": int(jumlah.min()),
        "max_pohon": int(jumlah.max()),
        "median_pohon": float(jumlah.median()),
    }


def hitung_area_hektar(luas_piksel: int, gsd_cm_per_piksel: float | None) -> float | None:
    """Menghitung luas area (hektar) dari luas piksel bila ground sampling distance diketahui."""
    if not gsd_cm_per_piksel or gsd_cm_per_piksel <= 0:
        return None
    m_per_piksel = gsd_cm_per_piksel / 100.0
    luas_m2 = luas_piksel * (m_per_piksel ** 2)
    return luas_m2 / 10_000.0


def evaluasi_ground_truth(prediksi: list[int], aktual: list[int]) -> dict:
    """Menghitung MAE/RMSE selisih jumlah pohon bila anotasi ground truth tersedia."""
    if not prediksi or len(prediksi) != len(aktual):
        return {}
    p = np.array(prediksi, dtype=float)
    a = np.array(aktual, dtype=float)
    return {
        "mae": float(np.mean(np.abs(p - a))),
        "rmse": float(np.sqrt(np.mean((p - a) ** 2))),
        "bias": float(np.mean(p - a)),
        "n": int(len(p)),
    }
