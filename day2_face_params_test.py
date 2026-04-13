import argparse
import json

import cv2

from passenger_identifier import PassengerIdentifier


def main():
    ap = argparse.ArgumentParser(description="Day 2: test face parameter extraction")
    ap.add_argument("--image", default="photo_2026-04-04_19-35-56.jpg", help="path to image")
    ap.add_argument("--out", default="day2_face_params.json", help="output json")
    ap.add_argument("--show", action="store_true", help="show preview with detected bbox")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Image not found: {args.image}")
        return 1

    ident = PassengerIdentifier()
    params = ident.extract_face_params(img)
    if params is None:
        print("Face params not extracted")
        ident.close()
        return 2

    descriptor = ident.compute_descriptor(params)
    params["descriptor_dim"] = int(len(descriptor))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    x, y, w, h = params["bbox_xywh"]
    print(f"Saved: {args.out}")
    print(f"bbox={x},{y},{w},{h}")
    print(f"descriptor_dim={len(descriptor)}")

    if args.show:
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cx, cy = params["centroid_xy"]
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.imshow("Day2 Face Params", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ident.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
