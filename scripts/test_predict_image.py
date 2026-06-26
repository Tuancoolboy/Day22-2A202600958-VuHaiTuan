from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from predict_image import analyze_offline, load_image


def test_load_p3_ppm_and_analyze_offline(tmp_path: Path):
    image_path = tmp_path / "sample.ppm"
    image_path.write_text(
        "\n".join(
            [
                "P3",
                "# tiny warm image",
                "2 2",
                "255",
                "255 200 50",
                "240 180 60",
                "20 40 220",
                "30 60 200",
            ]
        ),
        encoding="ascii",
    )

    image = load_image(image_path)
    result = analyze_offline(image, top_k=3)

    assert result["engine"] == "offline-visual"
    assert result["image"] == {"width": 2, "height": 2, "aspect_ratio": 1.0}
    assert len(result["predictions"]) == 3
    assert result["features"]["dominant_palette"]


def test_cli_demo_json_runs():
    script = Path(__file__).resolve().parent / "predict_image.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--demo", "--format", "json", "--top-k", "2"],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["engine"] == "offline-visual"
    assert len(result["predictions"]) == 2
