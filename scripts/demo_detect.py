"""Skrip bantu: menghasilkan citra drone demo dan menjalankan deteksi contoh.

Jalankan:
    python scripts/demo_detect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import ringkasan_batch  # noqa: E402
from src.detector import buat_citra_drone_demo, deteksi_klasik, gambar_anotasi  # noqa: E402


def main() -> int:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    hasil = []
    for i in range(1, 4):
        img = buat_citra_drone_demo(ukuran=640, n_pohon=15 + i * 5, seed=i)
        det = deteksi_klasik(img)
        vis = gambar_anotasi(img, det)
        cv2.imwrite(str(out_dir / f"demo_deteksi_{i}.png"), vis)
        d = det.sebagai_dict()
        d["nama_file"] = f"demo_{i}.png"
        hasil.append(d)
        print(f"demo_{i}.png -> perkiraan {det.jumlah_pohon} pohon")

    df = ringkasan_batch(hasil)
    df.to_csv(out_dir / "demo_batch.csv", index=False)
    print(f"\nHasil disimpan di: {out_dir}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
