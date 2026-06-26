#!/usr/bin/env python3
"""CLI image prediction helper.

The default engine is intentionally dependency-light: it reads an image,
extracts visual features, and returns deterministic labels that are useful for
quick sanity checks. For object-class predictions, use the optional Hugging
Face engine with a pretrained image-classification model.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RGB = tuple[int, int, int]


@dataclass(frozen=True)
class ImageData:
    width: int
    height: int
    pixels: list[RGB]

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def luminance(pixel: RGB) -> float:
    red, green, blue = pixel
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def read_ppm(path: Path) -> ImageData:
    """Read a P3/P6 portable pixmap without external dependencies."""
    blob = path.read_bytes()
    index = 0

    def skip_ws_and_comments() -> None:
        nonlocal index
        while index < len(blob):
            char = blob[index]
            if char in b" \t\r\n":
                index += 1
                continue
            if char == ord("#"):
                while index < len(blob) and blob[index] not in b"\r\n":
                    index += 1
                continue
            break

    def next_token() -> bytes:
        nonlocal index
        skip_ws_and_comments()
        start = index
        while index < len(blob) and blob[index] not in b" \t\r\n#":
            index += 1
        if start == index:
            raise ValueError("Invalid PPM header: expected token")
        return blob[start:index]

    magic = next_token()
    if magic not in {b"P3", b"P6"}:
        raise ValueError("Only P3/P6 PPM images are supported without Pillow")

    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())
    if width <= 0 or height <= 0:
        raise ValueError("Invalid PPM dimensions")
    if max_value <= 0 or max_value > 65535:
        raise ValueError("Invalid PPM max value")

    expected = width * height
    pixels: list[RGB] = []

    if magic == b"P3":
        for _ in range(expected):
            red = int(next_token())
            green = int(next_token())
            blue = int(next_token())
            pixels.append(scale_pixel((red, green, blue), max_value))
    else:
        skip_ws_and_comments()
        if max_value > 255:
            raise ValueError("P6 PPM with max value > 255 is not supported")
        raw = blob[index : index + expected * 3]
        if len(raw) < expected * 3:
            raise ValueError("PPM file ended before all pixels were read")
        for offset in range(0, len(raw), 3):
            pixels.append((raw[offset], raw[offset + 1], raw[offset + 2]))

    return ImageData(width=width, height=height, pixels=pixels)


def scale_pixel(pixel: tuple[int, int, int], max_value: int) -> RGB:
    if max_value == 255:
        return pixel
    return tuple(clamp_channel(channel * 255 / max_value) for channel in pixel)  # type: ignore[return-value]


def load_image(path: Path) -> ImageData:
    """Load common images via Pillow, falling back to stdlib PPM support."""
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        if path.read_bytes()[:2] in {b"P3", b"P6"}:
            return read_ppm(path)
        raise RuntimeError(
            "Pillow is required for PNG/JPG/WebP images. Install with "
            "`pip install Pillow`, or pass a P3/P6 .ppm file."
        ) from None

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        pixels = list(rgb.getdata())
    return ImageData(width=width, height=height, pixels=pixels)


def iter_sampled_pixels(image: ImageData, limit: int = 80_000) -> Iterable[RGB]:
    total = len(image.pixels)
    if total <= limit:
        yield from image.pixels
        return
    stride = max(1, total // limit)
    for index in range(0, total, stride):
        yield image.pixels[index]


def quantized_color(pixel: RGB) -> str:
    # Four buckets per channel keeps the palette readable and stable.
    centers = [32, 96, 160, 224]
    bucket = tuple(centers[min(channel // 64, 3)] for channel in pixel)
    return "#{:02x}{:02x}{:02x}".format(*bucket)


def estimate_edge_strength(image: ImageData, max_checks: int = 60_000) -> float:
    if image.width < 2 or image.height < 2:
        return 0.0

    step = max(1, int(math.sqrt((image.width * image.height) / max_checks)))
    diffs: list[float] = []
    for y_coord in range(0, image.height - 1, step):
        row = y_coord * image.width
        next_row = (y_coord + 1) * image.width
        for x_coord in range(0, image.width - 1, step):
            here = luminance(image.pixels[row + x_coord])
            right = luminance(image.pixels[row + x_coord + 1])
            down = luminance(image.pixels[next_row + x_coord])
            diffs.append(abs(here - right))
            diffs.append(abs(here - down))
    if not diffs:
        return 0.0
    return statistics.fmean(diffs) / 255


def analyze_offline(image: ImageData, top_k: int = 5) -> dict[str, object]:
    sampled = list(iter_sampled_pixels(image))
    if not sampled:
        raise ValueError("Image has no pixels")

    reds = [pixel[0] for pixel in sampled]
    greens = [pixel[1] for pixel in sampled]
    blues = [pixel[2] for pixel in sampled]
    lumas = [luminance(pixel) for pixel in sampled]
    saturations = [
        0.0 if max(pixel) == 0 else (max(pixel) - min(pixel)) / max(pixel)
        for pixel in sampled
    ]

    mean_rgb = (
        clamp_channel(statistics.fmean(reds)),
        clamp_channel(statistics.fmean(greens)),
        clamp_channel(statistics.fmean(blues)),
    )
    brightness = statistics.fmean(lumas) / 255
    contrast = statistics.pstdev(lumas) / 255 if len(lumas) > 1 else 0.0
    saturation = statistics.fmean(saturations)
    edge_strength = estimate_edge_strength(image)
    palette_counts = Counter(quantized_color(pixel) for pixel in sampled)
    palette = [
        {"hex": color, "share": round(count / len(sampled), 4)}
        for color, count in palette_counts.most_common(top_k)
    ]

    predictions = rank_offline_labels(
        image=image,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        edge_strength=edge_strength,
        mean_rgb=mean_rgb,
    )[:top_k]

    return {
        "engine": "offline-visual",
        "image": {
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.aspect_ratio, 3),
        },
        "features": {
            "mean_rgb": mean_rgb,
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "saturation": round(saturation, 4),
            "edge_strength": round(edge_strength, 4),
            "dominant_palette": palette,
        },
        "predictions": predictions,
    }


def rank_offline_labels(
    *,
    image: ImageData,
    brightness: float,
    contrast: float,
    saturation: float,
    edge_strength: float,
    mean_rgb: RGB,
) -> list[dict[str, object]]:
    red, green, blue = mean_rgb
    scores: dict[str, float] = {
        "general photo or graphic": 0.45,
        "plain background or low-detail image": max(0.0, 0.85 - contrast * 3 - edge_strength * 4),
        "document, sketch, or screenshot-like image": (
            0.30 + edge_strength * 2.4 + max(0.0, 0.22 - saturation)
        ),
        "colorful detailed scene": 0.20 + saturation * 1.2 + edge_strength * 1.3,
        "wide scene or landscape-like image": 0.20 + max(0.0, image.aspect_ratio - 1.2) * 0.35,
        "dark image or night scene": max(0.0, 0.95 - brightness * 1.4),
        "bright image or high-key scene": max(0.0, brightness * 1.1 - 0.25),
        "warm-toned object or indoor scene": max(0.0, (red - blue) / 255 + 0.20),
        "cool-toned sky or water-like palette": max(0.0, (blue - red) / 255 + 0.20),
        "green/nature-like color palette": max(0.0, (green - max(red, blue)) / 255 + 0.15),
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_score = max(ranked[0][1], 1e-6)
    return [
        {"label": label, "confidence": round(min(score / top_score, 1.0), 4)}
        for label, score in ranked
        if score > 0.05
    ]


def build_demo_image() -> ImageData:
    width, height = 96, 64
    pixels: list[RGB] = []
    for y_coord in range(height):
        for x_coord in range(width):
            red = int(50 + 160 * (x_coord / max(width - 1, 1)))
            green = int(80 + 120 * (y_coord / max(height - 1, 1)))
            blue = 180 if x_coord < width * 0.55 else 70
            if 22 < x_coord < 72 and 20 < y_coord < 45:
                red, green, blue = 220, 185, 70
            pixels.append((red, green, blue))
    return ImageData(width=width, height=height, pixels=pixels)


def predict_huggingface(path: Path, model: str, top_k: int) -> dict[str, object]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for --engine hf. Install requirements first."
        ) from exc

    try:
        from PIL import Image  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --engine hf.") from exc

    classifier = pipeline("image-classification", model=model)
    raw_predictions = classifier(str(path), top_k=top_k)
    predictions = [
        {"label": item["label"], "confidence": round(float(item["score"]), 4)}
        for item in raw_predictions
    ]
    return {
        "engine": "huggingface",
        "model": model,
        "image_path": str(path),
        "predictions": predictions,
    }


def format_text(result: dict[str, object]) -> str:
    lines = [f"Engine: {result['engine']}"]
    image = result.get("image")
    if isinstance(image, dict):
        lines.append(
            f"Image: {image['width']}x{image['height']} "
            f"(aspect {image['aspect_ratio']})"
        )
    features = result.get("features")
    if isinstance(features, dict):
        lines.append(
            "Features: "
            f"brightness={features['brightness']}, "
            f"contrast={features['contrast']}, "
            f"saturation={features['saturation']}, "
            f"edge_strength={features['edge_strength']}"
        )
        palette = features.get("dominant_palette", [])
        if isinstance(palette, list):
            colors = ", ".join(
                f"{entry['hex']} ({entry['share']:.1%})"
                for entry in palette
                if isinstance(entry, dict)
            )
            lines.append(f"Dominant palette: {colors}")
    lines.append("Predictions:")
    predictions = result.get("predictions", [])
    if isinstance(predictions, list):
        for index, item in enumerate(predictions, start=1):
            if isinstance(item, dict):
                lines.append(f"  {index}. {item['label']} ({item['confidence']:.2f})")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict or analyze an image.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to an image file.")
    parser.add_argument(
        "--engine",
        choices=["offline", "hf"],
        default="offline",
        help="offline uses deterministic visual features; hf uses a pretrained model.",
    )
    parser.add_argument(
        "--model",
        default="google/vit-base-patch16-224",
        help="Hugging Face model id for --engine hf.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of labels to return.")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against a generated demo image; no file or dependencies required.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    try:
        if args.demo:
            result = analyze_offline(build_demo_image(), top_k=args.top_k)
        elif args.engine == "offline":
            if args.image is None:
                raise SystemExit("Pass an image path or use --demo.")
            result = analyze_offline(load_image(args.image), top_k=args.top_k)
        else:
            if args.image is None:
                raise SystemExit("Pass an image path for --engine hf.")
            result = predict_huggingface(args.image, model=args.model, top_k=args.top_k)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
