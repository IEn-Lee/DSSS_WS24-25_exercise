import os
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt

# 設置隨機種子
np.random.seed(23365648)

# 定義資料集路徑
BAGLS_PATH = "Mini_BAGLS_dataset"

# 列出資料夾中的所有檔案
files = os.listdir(BAGLS_PATH)

# 選擇圖片檔案（不包括掩碼）
image_files = [f for f in files if ".png" in f and not "_seg.png" in f]

# 隨機選擇 1 張圖片（您希望對同一張圖片進行增強操作）
selected_image = np.random.choice(image_files, 1)[0]

# 載入選定的圖片
image = cv2.imread(os.path.join(BAGLS_PATH, selected_image))

# 嘗試找到對應的掩碼檔案
base_name = os.path.splitext(selected_image)[0]
mask_file = f"{base_name}_seg.png"
mask = None
if mask_file in files:
    mask = cv2.imread(os.path.join(BAGLS_PATH, mask_file), cv2.IMREAD_GRAYSCALE)

# 定義增強操作
transformations = [
    A.HorizontalFlip(p=1.0),  # 強制水平翻轉
    A.VerticalFlip(p=1.0),  # 強制垂直翻轉
    A.RandomRotate90(p=1.0),  # 強制轉置
    A.CenterCrop(height=256, width=256, p=1.0),  # 強制進行中心裁剪
    A.Transpose(p=1),
]

transformations_name = [
    "Horizontal Flip",
    "Vertical Flip",
    "Random Rotate 90",
    "Center Crop",
]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 顯示原始圖像及其掩碼
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# 遍歷每個增強操作並生成結果
for i, transform in enumerate(transformations):
    # 應用增強操作
    augmented = transform(image=image, mask=mask if mask is not None else None)
    augmented_image = augmented["image"]
    augmented_mask = augmented["mask"] if mask is not None else None

    # 創建顯示圖像的子圖

    if mask is not None:
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Original Mask")
        axes[1].axis("off")
    else:
        axes[1].axis("off")

    # 顯示增強後的圖像及其掩碼
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 8))

    axes2[0].imshow(augmented_image)
    axes2[0].set_title(f"{transformations_name[i]} Image")
    axes2[0].axis("off")

    if augmented_mask is not None:
        axes2[1].imshow(augmented_mask, cmap="gray")
        axes2[1].set_title(f"{transformations_name[i]} Mask")
        axes2[1].axis("off")
    else:
        axes2[1].axis("off")

    # 顯示圖像
    # plt.tight_layout()
    plt.show()
