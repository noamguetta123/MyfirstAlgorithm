import cv2
import numpy as np
from PIL import Image
from rembg import remove

IMG1_PATH = "/root/.claude/uploads/5b91c081-31c7-5e52-a255-8c406516de76/3ce32133-IMG_1291.png"
IMG2_PATH = "/root/.claude/uploads/5b91c081-31c7-5e52-a255-8c406516de76/e5d76539-BE978159CF054A49AE37AFA87963DBB2.png"
OUTPUT_PATH = "/home/user/MyfirstAlgorithm/result_v2.png"

print("Loading images...")
img1_pil = Image.open(IMG1_PATH).convert("RGBA")
img2_pil = Image.open(IMG2_PATH).convert("RGBA")

print(f"Image 1 size: {img1_pil.size}")
print(f"Image 2 size: {img2_pil.size}")

# ── Step 1: get person mask from image 1 ──────────────────────────────────────
print("Removing background from image 1...")
img1_nobg = remove(img1_pil)
a1 = np.array(img1_nobg)[:, :, 3]
# hard threshold
a1_bin = (a1 > 30).astype(np.uint8)
rows1 = np.where(np.any(a1_bin, axis=1))[0]
cols1 = np.where(np.any(a1_bin, axis=0))[0]
rmin1, rmax1 = rows1[0], rows1[-1]
cmin1, cmax1 = cols1[0], cols1[-1]
person1_h = rmax1 - rmin1
person1_w = cmax1 - cmin1
print(f"  Person1 bbox  y:{rmin1}-{rmax1}  x:{cmin1}-{cmax1}  ({person1_w}x{person1_h})")

# ── Step 2: extract person from image 2 ───────────────────────────────────────
print("Removing background from image 2...")
img2_nobg = remove(img2_pil)
arr2 = np.array(img2_nobg)           # RGBA, uint8
a2 = arr2[:, :, 3]
a2_bin = (a2 > 30).astype(np.uint8)
rows2 = np.where(np.any(a2_bin, axis=1))[0]
cols2 = np.where(np.any(a2_bin, axis=0))[0]
rmin2, rmax2 = rows2[0], rows2[-1]
cmin2, cmax2 = cols2[0], cols2[-1]
person2_crop = arr2[rmin2:rmax2, cmin2:cmax2]   # RGBA crop
print(f"  Person2 bbox  y:{rmin2}-{rmax2}  x:{cmin2}-{cmax2}")

# ── Step 3: resize person2 to fit person1 slot ────────────────────────────────
target_w, target_h = person1_w, person1_h
person2_r = cv2.resize(person2_crop, (target_w, target_h),
                       interpolation=cv2.INTER_LANCZOS4)
# Hard threshold alpha: 0 or 255
alpha_hard = (person2_r[:, :, 3] > 60).astype(np.uint8) * 255
# Gentle feather only at the edge boundary (5 px)
alpha_feathered = cv2.GaussianBlur(alpha_hard.astype(np.float32), (11, 11), 3)
alpha_feathered = np.clip(alpha_feathered, 0, 255)

rgb2 = person2_r[:, :, :3].astype(np.float32)

# ── Step 4: inpaint background of image 1 behind original person ──────────────
img1_rgb = np.array(img1_pil.convert("RGB"))
mask_inp = cv2.dilate(a1_bin * 255, np.ones((20, 20), np.uint8), iterations=2)
bg_clean = cv2.inpaint(img1_rgb, mask_inp, 20, cv2.INPAINT_TELEA)

# ── Step 5: colour-grade person2 to match image 1 lighting ───────────────────
# Use the inpainted region as reference
roi = bg_clean[rmin1:rmin1+target_h, cmin1:cmin1+target_w].astype(np.float32)
solid = alpha_feathered > 128

for c in range(3):
    src_mean = float(np.mean(roi[:, :, c]))
    src_std  = max(float(np.std(roi[:, :, c])), 1.0)
    px = rgb2[:, :, c][solid]
    if px.size == 0:
        continue
    tgt_mean = float(np.mean(px))
    tgt_std  = max(float(np.std(px)), 1.0)
    # gentle blend: only 40 % colour shift so skin still looks natural
    shift = 0.40
    rgb2[:, :, c] = rgb2[:, :, c] + shift * (
        (rgb2[:, :, c] - tgt_mean) * (src_std / tgt_std) + src_mean - rgb2[:, :, c]
    )
    rgb2[:, :, c] = np.clip(rgb2[:, :, c], 0, 255)

# ── Step 6: composite ─────────────────────────────────────────────────────────
result = bg_clean.astype(np.float32)
a_norm = alpha_feathered[:, :, np.newaxis] / 255.0  # (H,W,1)

y0 = rmin1
x0 = cmin1
y1 = min(y0 + target_h, result.shape[0])
x1 = min(x0 + target_w, result.shape[1])
ph = y1 - y0
pw = x1 - x0

bg_roi  = result[y0:y1, x0:x1]
fg      = rgb2[:ph, :pw]
a_roi   = a_norm[:ph, :pw]

result[y0:y1, x0:x1] = fg * a_roi + bg_roi * (1.0 - a_roi)
result = np.clip(result, 0, 255).astype(np.uint8)

Image.fromarray(result).save(OUTPUT_PATH)
print(f"Saved → {OUTPUT_PATH}")
