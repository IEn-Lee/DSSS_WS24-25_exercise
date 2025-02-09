import numpy as np
import flammkuchen
import matplotlib.pyplot as plt

# 讀取 hdf5 檔案中的數據
data = flammkuchen.load("rectangles_dsss.sec")
ground_truth_rects = data["ground_truth"]  # 真實矩形列表
predicted_rects = data["predicted"]  # 預測矩形列表


# IoU 計算函數
def calculate_iou(rect1, rect2):
    """
    Calculates the Intersection over Union (IoU) score between two rectangles using NumPy.

    Args:
        rect1 (tuple): Coordinates of the first rectangle in the format (x, y, width, height).
        rect2 (tuple): Coordinates of the second rectangle in the same format.

    Returns:
        float: The IoU score between the two rectangles.
    """

    # Convert rectangle coordinates and dimensions to NumPy arrays
    rect1 = np.array(rect1)
    rect2 = np.array(rect2)

    # Calculate coordinates of bounding boxes
    x1_gt, y1_gt, w_gt, h_gt = rect1
    x2_gt, y2_gt = x1_gt + w_gt, y1_gt + h_gt  # Bottom-right corner of ground truth

    x1_p, y1_p, w_p, h_p = rect2
    x2_p, y2_p = x1_p + w_p, y1_p + h_p  # Bottom-right corner of predicted box

    # Calculate coordinates of the intersection rectangle
    x_left = np.maximum(x1_gt, x1_p)
    y_top = np.maximum(y1_gt, y1_p)
    x_right = np.minimum(x2_gt, x2_p)
    y_bottom = np.minimum(y2_gt, y2_p)

    # Handle cases where there's no intersection
    intersection_width = np.maximum(0, x_right - x_left)
    intersection_height = np.maximum(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    # Calculate areas of both rectangles
    area_gt = w_gt * h_gt
    area_pred = w_p * h_p

    # Calculate IoU
    # Union area = Ground Truth area + Predicted area - Intersection area
    union_area = area_gt + area_pred - intersection_area
    iou = intersection_area / union_area

    return iou


# 計算所有矩形的 IoU 分數
iou_scores = []
for i in range(len(ground_truth_rects)):
    gt_rect = ground_truth_rects[i]
    pred_rect = predicted_rects[i]
    iou = calculate_iou(gt_rect, pred_rect)
    iou_scores.append(iou)

# 繪製 IoU 分數的直方圖
plt.figure(figsize=(8, 6))
plt.hist(iou_scores, bins=35, color="skyblue", edgecolor="black")
plt.title("Distribution of IoU scores")
plt.xlabel("IoU")
plt.ylabel("Number of rectangles")
# plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
