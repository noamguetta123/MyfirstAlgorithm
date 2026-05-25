"""
Image inpainting - removes objects using Stability AI API.
Save your image as 'image.jpg' in the same folder, then run.
"""

import os
import requests
from PIL import Image, ImageDraw
import io

API_KEY = os.environ.get("STABILITY_API_KEY", "YOUR_STABILITY_API_KEY_HERE")

# Coordinates of the ceiling shelf (upper-right area)
# Adjust these values based on your image dimensions
MASK_REGIONS = [
    # (x1, y1, x2, y2) - bounding boxes to remove
    (480, 0, 820, 260),   # ceiling shelf/cabinet on the upper right
]


def create_mask(image_path: str, regions: list) -> tuple:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for (x1, y1, x2, y2) in regions:
        # Scale if needed
        draw.rectangle([x1, y1, x2, y2], fill=255)

    return img, mask


def inpaint_stability(image_path: str, output_path: str = "result.png"):
    img, mask = create_mask(image_path, MASK_REGIONS)

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")
    mask_bytes.seek(0)

    print("Sending to Stability AI inpainting...")
    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
        headers={"authorization": f"Bearer {API_KEY}", "accept": "image/*"},
        files={
            "image": ("image.png", img_bytes, "image/png"),
            "mask": ("mask.png", mask_bytes, "image/png"),
        },
        data={
            "prompt": "clean white wall and ceiling, empty room, natural lighting",
            "negative_prompt": "shelf, furniture, cabinet, object",
            "output_format": "png",
        },
    )

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {output_path}")
    else:
        raise RuntimeError(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    inpaint_stability("image.jpg", "result.png")
