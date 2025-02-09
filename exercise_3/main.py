import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the dataset path
BAGLS_PATH = "Mini_BAGLS_dataset"

# List all files in the dataset directory
files = os.listdir(BAGLS_PATH)

# Filter image files (excluding masks)
image_files = [f for f in files if ".png" in f and not "_seg.png" in f]

# Randomly select 4 image files
selected_images = np.random.choice(image_files, 4, replace=False)

# Initialize lists for corresponding metadata and masks
meta_files = []
mask_files = []
subject_status = []

for image_file in selected_images:
    # get image name(without extension)
    # os.path.splitext(image_file)
    # Output: ("image", ".png")
    base_name = os.path.splitext(image_file)[0]

    # Find corresponding metadata and mask files
    meta_file = f"{base_name}.meta"
    mask_file = f"{base_name}_seg.png"

    # Append to the corresponding file
    if meta_file in files:
        meta_files.append(meta_file)
    else:
        meta_files.append(None)

    if mask_file in files:
        mask_files.append(mask_file)
    else:
        mask_files.append(None)

    # Extract the "Subject disorder status" from the meta file
    if meta_file in files:
        with open(os.path.join(BAGLS_PATH, meta_file), "r") as f:
            metadata = json.load(f)
            subject_status.append(metadata.get("Subject disorder status", "Unknown"))
    else:
        subject_status.append("Unknown")


# Plot the images with masks overlaid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i, image_file in enumerate(selected_images):
    # Load image
    image_path = os.path.join(BAGLS_PATH, image_file)
    image = Image.open(image_path).convert("RGB")

    # Load mask (if available)
    mask = None
    if mask_files[i] is not None:
        mask_path = os.path.join(BAGLS_PATH, mask_files[i])
        mask = Image.open(mask_path).convert("L")

    # Overlay the mask on the image (if mask exists)
    if mask is not None:
        # Convert mask to RGB to overlay it on the image
        mask_rgb = np.array(mask.convert("RGB"))
        image_rgb = np.array(image)
        # Make sure the mask has transparency (alpha channel) for blending
        mask_rgb = np.dstack(
            [mask_rgb, np.full_like(mask_rgb[:, :, 0], 128)]
        )  # Add alpha channel

        # Apply the mask to the image by blending
        image_rgb = np.dstack([image_rgb, np.full_like(image_rgb[:, :, 0], 255)])
        image_with_mask = np.array(image)
        # Image blending here
        alpha = 0.6
        blended = cv2.addWeighted(image_rgb, 1, mask_rgb, 1, 0)
        image_with_mask = Image.fromarray(blended)

    else:
        image_with_mask = image

    # Plot the image with overlayed mask
    ax = axes[i // 2, i % 2]
    ax.imshow(image_with_mask)
    ax.set_title(f"{subject_status[i]}")
    ax.axis("off")

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
