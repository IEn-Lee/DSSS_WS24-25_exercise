import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 讀取影像
image_path = "leaves.jpg"  # 替換為你的影像路徑
image = Image.open(image_path).convert("RGB")
image_array = np.array(image).astype(np.float32)  # 使用float32進行計算

# 方法 1: Lightness Method
lightness = (np.max(image_array, axis=2) + np.min(image_array, axis=2)) / 2.0

# 方法 2: Average Method
average = np.mean(image_array, axis=2)

# 方法 3: Luminosity Method
luminosity = (
    0.2989 * image_array[:, :, 0]
    + 0.5870 * image_array[:, :, 1]
    + 0.1140 * image_array[:, :, 2]
)

# 視覺化結果
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 原始影像
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")

# Lightness
axes[1].imshow(lightness, cmap="gray")
axes[1].set_title("Lightness")
axes[1].axis("off")

# Average
axes[2].imshow(average, cmap="gray")
axes[2].set_title("Average")
axes[2].axis("off")

# Luminosity
axes[3].imshow(luminosity, cmap="gray")
axes[3].set_title("Luminosity")
axes[3].axis("off")

# plt.tight_layout()
plt.show()
