import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

# load data
data_original = pd.read_csv("winequality-red.csv")
data = data_original.copy()

# acquire features and targets
X = data.drop(columns=["quality"])
y = data["quality"]

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)


# 繪圖函數
def plot_2d_projection(X_proj, title, y):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, s=10)
    plt.colorbar(scatter, label="Wine Quality")
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()


# 繪製三種方法的降維結果
plot_2d_projection(X_pca, "PCA", y)
plot_2d_projection(X_tsne, "t-SNE", y)
plot_2d_projection(X_umap, "UMAP", y)


# 構建檢驗數據
grouped_quality = [
    data[data["quality"] == q]["residual sugar"]
    for q in sorted(data["quality"].unique())
]

# 使用 ANOVA 檢驗
f_stat, p_value = f_oneway(*grouped_quality)

# 結果報告
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.3e}")
if p_value < 0.05:
    print("Reject H₀: Residual sugar significantly affects wine quality.")
else:
    print(
        "Fail to reject H₀: Residual sugar does not significantly affect wine quality."
    )
