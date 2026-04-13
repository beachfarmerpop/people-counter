from pathlib import Path
from typing import List, Tuple

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


def _has_cyrillic(text: str) -> bool:
    """Fast check if text contains non-ASCII chars needing PIL."""
    for ch in text:
        if ord(ch) > 127:
            return True
    return False


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    size: int = 24,
    shadow: bool = False,
) -> np.ndarray:
    """Draw text — uses fast cv2.putText for ASCII, PIL only for Unicode."""
    if image is None:
        return image

    if not _has_cyrillic(text):
        # Fast path: pure ASCII — use OpenCV directly, no PIL conversion
        x, y = position
        font_scale = size / 28.0
        thickness = max(1, int(size / 14))
        # cv2 y is baseline, our y is top — offset by ~size
        cv2_y = y + size
        if shadow:
            cv2.putText(image, text, (x + 1, cv2_y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(image, text, (x, cv2_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        return image

    # Slow path: Cyrillic/Unicode — use PIL
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


def draw_texts_batch(
    image: np.ndarray,
    items: List[Tuple[str, Tuple[int, int], Tuple[int, int, int], int, bool]],
) -> np.ndarray:
    """Draw multiple text items in ONE PIL conversion.

    Each item: (text, (x, y), (b, g, r), size, shadow).
    """
    if image is None or not items:
        return image

    # Split into ASCII-only and Cyrillic items
    ascii_items = []
    cyrillic_items = []
    for item in items:
        text = item[0]
        if _has_cyrillic(text):
            cyrillic_items.append(item)
        else:
            ascii_items.append(item)

    # Draw ASCII items directly with cv2 (no conversion)
    for text, (x, y), color, size, shadow in ascii_items:
        font_scale = size / 28.0
        thickness = max(1, int(size / 14))
        cv2_y = y + size
        if shadow:
            cv2.putText(image, text, (x + 1, cv2_y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(image, text, (x, cv2_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    # Draw all Cyrillic items in one PIL pass
    if cyrillic_items:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)

        for text, (x, y), color, size, shadow in cyrillic_items:
            font = _get_font(size)
            fill_rgb = (color[2], color[1], color[0])
            if shadow:
                draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
            draw.text((x, y), text, font=font, fill=fill_rgb)

        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        image[:, :, :] = bgr

    return image
