"""Convert .ppm fractal images to .png.

Usage:
    python conversion.py <input_dir> [output_dir]

If output_dir is omitted, PNGs are written to <input_dir>/converted_pngs/.
"""
import os
import sys
from PIL import Image


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(input_folder, "converted_pngs")
    os.makedirs(output_folder, exist_ok=True)

    converted = 0
    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(".ppm"):
            continue
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname.replace(".ppm", ".png"))
        try:
            Image.open(in_path).save(out_path)
            print(f"Converted: {fname}")
            converted += 1
        except Exception as e:
            print(f"Failed to convert {fname}: {e}", file=sys.stderr)

    print(f"\nDone. {converted} PPM files converted to PNG in {output_folder}")


if __name__ == "__main__":
    main()
