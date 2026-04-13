from pathlib import Path

import cv2


def show_splash(app_dir: Path, timeout_ms: int = 2500) -> None:
    candidates = [
        app_dir / "splash_images" / "splash_program2.jpg",
        app_dir / "splash_images" / "splash_previous.jpg",
        app_dir / "_internal" / "splash_images" / "splash_program2.jpg",
        app_dir / "_internal" / "splash_images" / "splash_previous.jpg",
    ]
    image_path = None
    for c in candidates:
        if c.exists():
            image_path = c
            break

    if image_path is None:
        return

    image = cv2.imread(str(image_path))
    if image is None:
        return

    window = "Program12 Splash"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    cv2.imshow(window, image)
    cv2.waitKey(timeout_ms)
    cv2.destroyWindow(window)
