#!/usr/bin/env python3
"""Generate a printable calibration board (PNG + PDF) sized for A4.

Designed for the PIPER ↔ ZED 2i auto-calibration script
(`piper_calibrate_zed_extrinsics_auto.py`). Default geometry produces a
**100 × 120 mm** board (5 × 6 squares of 20 mm) — a comfortable size to grip
between the PIPER fingers, and small enough to leave a generous cut margin
on a single A4 sheet.

Output
------
The page contains:
  * The calibration board centred on the page.
  * Black corner cut-marks so you can scissor-trim cleanly.
  * A 50 mm reference ruler beneath the board: after printing, measure it
    with a real ruler to confirm your printer didn't auto-scale.
  * A text label with the exact board parameters — copy these into the
    matching `--cols/--rows/--square` flags when running the calibration
    script.

Print
-----
Open the PDF, set the printer to **100 % / "Actual size"** (NOT "fit to
page"), single-sided, A4. Verify the 50 mm ruler is exactly 50 mm with a
real ruler. Then cut along the corner marks and rigid-mount the result on
something flat (foam-board, acrylic, hard cardboard).

Usage
-----
    # Default 100x120 mm checkerboard, 20 mm squares (4x5 inner corners):
    uv run --no-sync --active scripts_realbot/cam_calibration/generate_calibration_board.py

    # ChArUco variant, same physical size:
    uv run --no-sync --active scripts_realbot/cam_calibration/generate_calibration_board.py \\
        --board charuco --marker 15

    # Custom: 6x8 squares of 15 mm = 90x120 mm board:
    uv run --no-sync --active scripts_realbot/cam_calibration/generate_calibration_board.py \\
        --squares-x 6 --squares-y 8 --square 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

A4_W_MM = 210.0
A4_H_MM = 297.0
DEFAULT_DPI = 300


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def _make_checkerboard_px(squares_x: int, squares_y: int, sq_px: int) -> np.ndarray:
    """Black/white checkerboard. Top-left square is BLACK (OpenCV convention)."""
    w = squares_x * sq_px
    h = squares_y * sq_px
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                img[r * sq_px:(r + 1) * sq_px, c * sq_px:(c + 1) * sq_px] = 0
    return img


def _make_charuco_px(
    squares_x: int,
    squares_y: int,
    square_mm: float,
    marker_mm: float,
    sq_px: int,
) -> np.ndarray:
    """ChArUco board image. Uses DICT_5X5_100 (matches the calibration script)."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is unavailable — install opencv-contrib-python or use "
            "--board checkerboard."
        )
    aruco = cv2.aruco
    adict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        float(square_mm) / 1000.0,
        float(marker_mm) / 1000.0,
        adict,
    )
    size_px = (squares_x * sq_px, squares_y * sq_px)
    if hasattr(board, "generateImage"):
        return board.generateImage(size_px)
    return board.draw(size_px)  # legacy API


def _draw_corner_cut_marks(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                           cut_len_px: int, gap_px: int) -> None:
    """Draw 4 L-shaped cut marks just OUTSIDE the board rectangle."""
    for (cx, cy, dx, dy) in [
        (x0, y0, -1, -1),
        (x1, y0,  1, -1),
        (x0, y1, -1,  1),
        (x1, y1,  1,  1),
    ]:
        # Horizontal arm
        h_x0 = cx + dx * gap_px
        h_x1 = cx + dx * (gap_px + cut_len_px)
        cv2.line(canvas, (min(h_x0, h_x1), cy), (max(h_x0, h_x1), cy), 0, 2)
        # Vertical arm
        v_y0 = cy + dy * gap_px
        v_y1 = cy + dy * (gap_px + cut_len_px)
        cv2.line(canvas, (cx, min(v_y0, v_y1)), (cx, max(v_y0, v_y1)), 0, 2)


def _draw_ruler(canvas: np.ndarray, x_left_px: int, y_top_px: int,
                length_mm: float, dpi: int) -> None:
    """Draw a horizontal 50 mm ruler with mm ticks for print-scale verification."""
    length_px = _mm_to_px(length_mm, dpi)
    bar_h = _mm_to_px(2.0, dpi)
    cv2.rectangle(canvas, (x_left_px, y_top_px),
                  (x_left_px + length_px, y_top_px + bar_h), 0, -1)
    for mm in range(int(length_mm) + 1):
        x = x_left_px + _mm_to_px(mm, dpi)
        if mm % 10 == 0:
            tick_h = _mm_to_px(4.0, dpi)
        elif mm % 5 == 0:
            tick_h = _mm_to_px(2.5, dpi)
        else:
            tick_h = _mm_to_px(1.2, dpi)
        cv2.line(canvas, (x, y_top_px + bar_h),
                 (x, y_top_px + bar_h + tick_h), 0, 1)
    cv2.putText(
        canvas,
        f"{int(length_mm)} mm scale (verify after printing)",
        (x_left_px, y_top_px - _mm_to_px(2.0, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, 0, 1, cv2.LINE_AA,
    )


def _compose_a4(board_img: np.ndarray, board_w_mm: float, board_h_mm: float,
                dpi: int, label_lines: list[str]) -> np.ndarray:
    page_w = _mm_to_px(A4_W_MM, dpi)
    page_h = _mm_to_px(A4_H_MM, dpi)
    canvas = np.full((page_h, page_w), 255, dtype=np.uint8)

    bw = _mm_to_px(board_w_mm, dpi)
    bh = _mm_to_px(board_h_mm, dpi)
    x0 = (page_w - bw) // 2
    y0 = _mm_to_px(35.0, dpi)  # 35 mm from top — leaves room for label below
    if board_img.shape != (bh, bw):
        board_img = cv2.resize(board_img, (bw, bh), interpolation=cv2.INTER_NEAREST)
    canvas[y0:y0 + bh, x0:x0 + bw] = board_img

    _draw_corner_cut_marks(
        canvas, x0, y0, x0 + bw, y0 + bh,
        cut_len_px=_mm_to_px(8.0, dpi),
        gap_px=_mm_to_px(2.0, dpi),
    )

    # Label block under the board
    text_y = y0 + bh + _mm_to_px(15.0, dpi)
    for line in label_lines:
        cv2.putText(canvas, line, (x0, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
        text_y += _mm_to_px(6.0, dpi)

    # Ruler near the bottom for print-scale verification
    ruler_y = page_h - _mm_to_px(25.0, dpi)
    _draw_ruler(canvas, x_left_px=x0, y_top_px=ruler_y,
                length_mm=50.0, dpi=dpi)

    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--board", choices=["checkerboard", "charuco"], default="checkerboard")
    p.add_argument("--squares-x", type=int, default=5,
                   help="number of squares horizontally (default 5)")
    p.add_argument("--squares-y", type=int, default=6,
                   help="number of squares vertically (default 6)")
    p.add_argument("--square", type=float, default=20.0,
                   help="square edge length in mm (default 20). "
                        "Default 5x6 of 20mm = 100x120mm board.")
    p.add_argument("--marker", type=float, default=15.0,
                   help="ChArUco marker edge length in mm (default 15)")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                   help=f"output DPI (default {DEFAULT_DPI})")
    p.add_argument("--output-dir",
                   default=str(Path(__file__).resolve().parent.parent.parent / "data" / "calib_boards"))
    args = p.parse_args()

    if args.board == "charuco" and args.marker >= args.square:
        raise SystemExit(
            f"--marker ({args.marker}mm) must be smaller than --square ({args.square}mm)."
        )

    board_w_mm = args.squares_x * args.square
    board_h_mm = args.squares_y * args.square

    if board_w_mm > A4_W_MM - 20 or board_h_mm > A4_H_MM - 60:
        raise SystemExit(
            f"Board {board_w_mm:.0f}x{board_h_mm:.0f}mm is too big for A4 "
            f"(need ≤ {A4_W_MM-20:.0f}x{A4_H_MM-60:.0f}mm with margins)."
        )

    sq_px = _mm_to_px(args.square, args.dpi)
    if args.board == "checkerboard":
        board_img = _make_checkerboard_px(args.squares_x, args.squares_y, sq_px)
        inner_x = args.squares_x - 1
        inner_y = args.squares_y - 1
        suffix = f"checker_{args.squares_x}x{args.squares_y}_sq{int(args.square)}mm"
        label_lines = [
            f"Checkerboard  {args.squares_x}x{args.squares_y} squares of {args.square:.0f} mm",
            f"Board size: {board_w_mm:.0f} x {board_h_mm:.0f} mm   "
            f"|   Inner corners: {inner_x} x {inner_y}",
            f"calib flags:  --cols {inner_x}  --rows {inner_y}  "
            f"--square {args.square/1000:.4f}",
        ]
    else:
        board_img = _make_charuco_px(
            args.squares_x, args.squares_y, args.square, args.marker, sq_px
        )
        suffix = (
            f"charuco_{args.squares_x}x{args.squares_y}"
            f"_sq{int(args.square)}_mk{int(args.marker)}mm"
        )
        label_lines = [
            f"ChArUco  {args.squares_x}x{args.squares_y} squares  "
            f"(DICT_5X5_100, marker={args.marker:.0f} mm)",
            f"Board size: {board_w_mm:.0f} x {board_h_mm:.0f} mm",
            f"calib flags:  --board charuco  --cols {args.squares_x}  "
            f"--rows {args.squares_y}  --square {args.square/1000:.4f}  "
            f"--marker {args.marker/1000:.4f}",
        ]

    page = _compose_a4(board_img, board_w_mm, board_h_mm, args.dpi, label_lines)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"calib_{suffix}.png"
    pdf_path = out_dir / f"calib_{suffix}.pdf"

    pil_img = Image.fromarray(page)
    pil_img.save(str(png_path), dpi=(args.dpi, args.dpi))
    pil_img.convert("RGB").save(str(pdf_path), resolution=float(args.dpi))

    print(f"Board: {board_w_mm:.0f} x {board_h_mm:.0f} mm  "
          f"({args.squares_x}x{args.squares_y} sq @ {args.square:.0f} mm, "
          f"{args.dpi} DPI)")
    print(f"PNG : {png_path}")
    print(f"PDF : {pdf_path}")
    print()
    print("Print at 100% / 'Actual size' on A4 (NOT 'fit to page').")
    print("Verify the 50 mm ruler at the bottom is exactly 50 mm with a real ruler.")
    print("Cut along the corner marks and mount on a flat rigid backing.")


if __name__ == "__main__":
    main()
