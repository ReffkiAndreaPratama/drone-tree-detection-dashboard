"""Pipeline deteksi/perhitungan pohon pada citra drone (Sesi 5).

Dua mode deteksi:
1. `klasik`  : deteksi berbasis segmentasi warna vegetasi (ExG) + watershed,
   memberi perkiraan jumlah/pusat pohon dari citra. Cocok untuk demo tanpa bobot model.
2. `yolo`    : menerima bobot model deteksi objek kustom (mis. YOLO) yang dilatih untuk
   kelas pohon/vegetasi. Bobot TIDAK disertakan; pengguna menyediakan sendiri.

PENTING: mode klasik memberi *perkiraan* kanopi dominan hijau, bukan deteksi pohon
terverifikasi. Jangan mengklaim akurasi tanpa anotasi ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DeteksiPohon:
    """Hasil deteksi satu citra."""

    jumlah_pohon: int
    pusat: list = field(default_factory=list)       # list (x, y)
    radius: list = field(default_factory=list)       # list int (perkiraan radius)
    luas_kanopi_piksel: int = 0
    lebar: int = 0
    tinggi: int = 0
    mode: str = "klasik"

    def sebagai_dict(self) -> dict:
        return {
            "jumlah_pohon": self.jumlah_pohon,
            "luas_kanopi_piksel": self.luas_kanopi_piksel,
            "lebar": self.lebar,
            "tinggi": self.tinggi,
            "mode": self.mode,
        }


def excess_green_index(bgr: np.ndarray) -> np.ndarray:
    """Indeks Kelebihan Hijau (2G - R - B), dinormalisasi ke 0-1."""
    b, g, r = bgr[:, :, 0].astype(np.float32), bgr[:, :, 1].astype(np.float32), bgr[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
    return exg


def deteksi_klasik(
    bgr: np.ndarray,
    ambang_hijau: float = 0.4,
    min_luas: int = 40,
    ukuran_maks: int = 900,
) -> DeteksiPohon:
    """Deteksi kanopi pohon berbasis ExG + watershed (perkiraan)."""
    h0, w0 = bgr.shape[:2]
    skala = 1.0
    if max(h0, w0) > ukuran_maks:
        skala = ukuran_maks / max(h0, w0)
        bgr = cv2.resize(bgr, (int(w0 * skala), int(h0 * skala)), interpolation=cv2.INTER_AREA)

    exg = excess_green_index(bgr)
    mask = (exg > ambang_hijau).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, puncak = cv2.threshold(dist_norm, 0.35, 1.0, cv2.THRESH_BINARY)
    puncak = (puncak * 255).astype(np.uint8)
    puncak = cv2.morphologyEx(puncak, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, label, stats, centroids = cv2.connectedComponentsWithStats(puncak, connectivity=8)

    pusat, radius = [], []
    for i in range(1, n):
        luas = int(stats[i, cv2.CC_STAT_AREA])
        if luas < min_luas:
            continue
        cx, cy = centroids[i]
        r = float(np.sqrt(luas / np.pi))
        pusat.append((float(cx / skala), float(cy / skala)))
        radius.append(float(r / skala))

    luas_kanopi = int(mask.sum() / (skala * skala))
    return DeteksiPohon(
        jumlah_pohon=len(pusat),
        pusat=pusat,
        radius=radius,
        luas_kanopi_piksel=luas_kanopi,
        lebar=w0,
        tinggi=h0,
        mode="klasik",
    )


def deteksi_yolo(bgr: np.ndarray, bobot: str | Path, conf: float = 0.25) -> DeteksiPohon:
    """Deteksi menggunakan bobot YOLO kustom (ultralytics). Bobot disediakan pengguna."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ultralytics belum terpasang. Jalankan: pip install ultralytics"
        ) from exc

    bobot = Path(bobot)
    if not bobot.exists():
        raise FileNotFoundError(f"Bobot model tidak ditemukan: {bobot}")

    model = YOLO(str(bobot))
    hasil = model.predict(bgr, conf=conf, verbose=False)[0]
    h, w = bgr.shape[:2]
    pusat, radius = [], []
    luas = 0
    for box in hasil.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = max(x2 - x1, y2 - y1) / 2
        pusat.append((cx, cy))
        radius.append(r)
        luas += int((x2 - x1) * (y2 - y1))
    return DeteksiPohon(
        jumlah_pohon=len(pusat),
        pusat=pusat,
        radius=radius,
        luas_kanopi_piksel=luas,
        lebar=w,
        tinggi=h,
        mode="yolo",
    )


def gambar_anotasi(bgr: np.ndarray, det: DeteksiPohon) -> np.ndarray:
    """Menggambar anotasi pusat & lingkaran pohon pada citra."""
    vis = bgr.copy()
    for (cx, cy), r in zip(det.pusat, det.radius):
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"Perkiraan pohon: {det.jumlah_pohon} (mode: {det.mode})",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return vis


def buat_citra_drone_demo(
    ukuran: int = 640, n_pohon: int = 25, seed: int = 42
) -> np.ndarray:
    """Membuat citra demo (BGR) menyerupai kanopi pohon dari udara.

    BUKAN foto drone nyata; hanya untuk menguji pipeline deteksi.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((ukuran, ukuran, 3), dtype=np.uint8)
    img[:, :] = (60, 90, 70)  # tanah/latar gelap kehijauan
    for _ in range(n_pohon):
        cx, cy = rng.integers(40, ukuran - 40, 2)
        r = int(rng.integers(18, 40))
        warna = (
            int(rng.integers(30, 70)),
            int(rng.integers(110, 180)),
            int(rng.integers(30, 70)),
        )
        cv2.circle(img, (int(cx), int(cy)), r, warna, -1)
        for _ in range(rng.integers(3, 8)):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0, r * 0.8)
            px, py = int(cx + rad * np.cos(ang)), int(cy + rad * np.sin(ang))
            cv2.circle(img, (px, py), int(rng.integers(3, 8)), warna, -1)
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
