"""Pengujian alur teknis DroneVision Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    evaluasi_ground_truth,
    hitung_area_hektar,
    ringkasan_batch,
    statistik_ringkas,
)
from src.detector import (  # noqa: E402
    buat_citra_drone_demo,
    deteksi_klasik,
    excess_green_index,
    gambar_anotasi,
)


def test_demo_image_shape() -> None:
    img = buat_citra_drone_demo(ukuran=256, n_pohon=5)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8


def test_exg_range() -> None:
    img = buat_citra_drone_demo(ukuran=128, n_pohon=3)
    exg = excess_green_index(img)
    assert exg.min() >= 0
    assert exg.max() <= 1


def test_deteksi_klasik_finds_trees() -> None:
    img = buat_citra_drone_demo(ukuran=320, n_pohon=20, seed=1)
    det = deteksi_klasik(img)
    assert det.jumlah_pohon > 0
    assert len(det.pusat) == det.jumlah_pohon
    assert det.mode == "klasik"


def test_deteksi_klasik_empty_image() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    det = deteksi_klasik(img)
    assert det.jumlah_pohon == 0


def test_gambar_anotasi_shape() -> None:
    img = buat_citra_drone_demo(ukuran=200, n_pohon=8)
    det = deteksi_klasik(img)
    vis = gambar_anotasi(img, det)
    assert vis.shape == img.shape


def test_ringkasan_batch() -> None:
    hasil = [
        {"nama_file": "a.jpg", "jumlah_pohon": 10, "luas_kanopi_piksel": 500, "lebar": 100, "tinggi": 100, "mode": "klasik"},
        {"nama_file": "b.jpg", "jumlah_pohon": 20, "luas_kanopi_piksel": 800, "lebar": 100, "tinggi": 100, "mode": "klasik"},
    ]
    df = ringkasan_batch(hasil)
    assert len(df) == 2
    stat = statistik_ringkas(df)
    assert stat["total_pohon"] == 30
    assert stat["max_pohon"] == 20


def test_statistik_kosong() -> None:
    stat = statistik_ringkas(pd.DataFrame())
    assert stat["total_gambar"] == 0


def test_hitung_area_hektar() -> None:
    assert hitung_area_hektar(0, None) is None
    assert hitung_area_hektar(0, 0) is None
    area = hitung_area_hektar(10_000, 10)  # 10 cm/piksel -> 0.1 m/piksel
    assert abs(area - 0.01) < 1e-6


def test_evaluasi_ground_truth() -> None:
    metrik = evaluasi_ground_truth([20, 30, 15], [22, 28, 18])
    assert metrik["n"] == 3
    assert metrik["mae"] > 0
    assert "rmse" in metrik


def test_evaluasi_kosong() -> None:
    assert evaluasi_ground_truth([], []) == {}
    assert evaluasi_ground_truth([1, 2], [1]) == {}
