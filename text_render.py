from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CACHE = {}


def _candidate_fonts() -> list:
    windir = Path("C:/Windows/Fonts")
    return [
        windir / "segoeui.ttf",
        windir / "arial.ttf",
        windir / "tahoma.ttf",
        windir / "calibri.ttf",
    ]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    for path in _candidate_fonts():
        if path.exists():
            _FONT_CACHE[size] = ImageFont.truetype(str(path), size)
            return _FONT_CACHE[size]

    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    size: int = 24,
    shadow: bool = False,
) -> np.ndarray:
    """Draw unicode text on OpenCV image using PIL font rendering."""
    if image is None:
        return image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = _get_font(size)

    x, y = position
    fill_rgb = (color[2], color[1], color[0])

    if shadow:
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))

    draw.text((x, y), text, font=font, fill=fill_rgb)

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    image[:, :, :] = bgr
    return image
