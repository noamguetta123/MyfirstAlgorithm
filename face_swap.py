import cv2
import numpy as np
from PIL import Image, ImageFilter
from rembg import remove
import io

IMG1_PATH = "/root/.claude/uploads/5b91c081-31c7-5e52-a255-8c406516de76/3ce32133-IMG_1291.png"
IMG2_PATH = "/root/.claude/uploads/5b91c081-31c7-5e52-a255-8c406516de76/e5d76539-BE978159CF054A49AE37AFA87963DBB2.png"
OUTPUT_PATH = "/home/user/MyfirstAlgorithm/result.png"

print("Loading images...")
img1 = Image.open(IMG1_PATH).convert("RGBA")
img2 = Image.open(IMG2_PATH).convert("RGBA")

print(f"Image 1 size: {img1.size}")
print(f"Image 2 size: {img2.size}")

# Remove background from person in image 1 to get their mask/bbox
print("Removing background from image 1 (to get person bbox)...")
img1_nobg = remove(img1)
img1_nobg_arr = np.array(img1_nobg)

# Get the alpha mask of the person in image 1
alpha1 = img1_nobg_arr[:, :, 3]
rows1 = np.any(alpha1 > 10, axis=1)
cols1 = np.any(alpha1 > 10, axis=0)
rmin1, rmax1 = np.where(rows1)[0][[0, -1]]
cmin1, cmax1 = np.where(cols1)[0][[0, -1]]
person1_h = rmax1 - rmin1
person1_w = cmax1 - cmin1
print(f"Person in image1 bbox: x={cmin1}, y={rmin1}, w={person1_w}, h={person1_h}")

# Remove background from person in image 2
print("Removing background from image 2 (beach girl)...")
img2_nobg = remove(img2)
img2_nobg_arr = np.array(img2_nobg)

# Get bounding box of person in image 2
alpha2 = img2_nobg_arr[:, :, 3]
rows2 = np.any(alpha2 > 10, axis=1)
cols2 = np.any(alpha2 > 10, axis=0)
rmin2, rmax2 = np.where(rows2)[0][[0, -1]]
cmin2, cmax2 = np.where(cols2)[0][[0, -1]]
person2_crop = img2_nobg_arr[rmin2:rmax2, cmin2:cmax2]
print(f"Person in image2 bbox: x={cmin2}, y={rmin2}, w={cmax2-cmin2}, h={rmax2-rmin2}")

# Scale person from image 2 to match person in image 1 proportions
target_w = person1_w
target_h = person1_h

# Resize the cropped person2 to target size
person2_resized = cv2.resize(person2_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

# Create the output - start with background of image 1 (without original person)
img1_arr = np.array(img1.convert("RGB"))
img1_nobg_rgba = img1_nobg_arr.copy()

# Build clean background: fill the person area with inpainted background
# We'll use a simple approach: blur the edges of image1 background and composite
bg1 = img1_arr.copy()

# Create mask for inpainting (where the original person was)
mask_inpaint = (alpha1 > 30).astype(np.uint8) * 255
mask_inpaint_dilated = cv2.dilate(mask_inpaint, np.ones((15, 15), np.uint8), iterations=2)

# Inpaint the background
bg_inpainted = cv2.inpaint(bg1, mask_inpaint_dilated, 15, cv2.INPAINT_TELEA)

# Now place the new person on the inpainted background
result = bg_inpainted.copy().astype(np.float32)

# Extract alpha mask of the new person (resized from img2)
new_person_alpha = person2_resized[:, :, 3].astype(np.float32) / 255.0
new_person_rgb = person2_resized[:, :, :3].astype(np.float32)

# Feather the mask edges for natural blending
alpha_blur = cv2.GaussianBlur(new_person_alpha, (21, 21), 0)
alpha_3ch = np.stack([alpha_blur] * 3, axis=2)

# Position: place new person in the same location as original person was
y_start = rmin1
x_start = cmin1
y_end = min(y_start + target_h, result.shape[0])
x_end = min(x_start + target_w, result.shape[1])
ph = y_end - y_start
pw = x_end - x_start

# Color match: adjust lighting of new person to match image 1's color temperature
roi_bg = result[y_start:y_end, x_start:x_end]
person_rgb_crop = new_person_rgb[:ph, :pw]
alpha_crop = alpha_3ch[:ph, :pw]

# Color grading: match mean/std of the source image lighting
for c in range(3):
    src_mean = np.mean(roi_bg[:, :, c])
    src_std = np.std(roi_bg[:, :, c])
    tgt_mean = np.mean(person_rgb_crop[:, :, c][alpha_crop[:, :, c] > 0.3])
    tgt_std = np.std(person_rgb_crop[:, :, c][alpha_crop[:, :, c] > 0.3])
    if tgt_std > 0:
        person_rgb_crop[:, :, c] = (person_rgb_crop[:, :, c] - tgt_mean) * (src_std / tgt_std) + src_mean
    person_rgb_crop[:, :, c] = np.clip(person_rgb_crop[:, :, c], 0, 255)

# Composite: blend new person onto background
result[y_start:y_end, x_start:x_end] = (
    person_rgb_crop * alpha_crop +
    roi_bg * (1.0 - alpha_crop)
)

result = np.clip(result, 0, 255).astype(np.uint8)

# Save output
output_img = Image.fromarray(result)
output_img.save(OUTPUT_PATH)
print(f"Result saved to: {OUTPUT_PATH}")
print(f"Output size: {output_img.size}")
