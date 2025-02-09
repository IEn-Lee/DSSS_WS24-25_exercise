import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_df = pd.read_csv("census_income_dataset.csv")
# data_df.head()

data_raw = data_df.copy()
data = data_raw
# Task 1-1

plt.figure(figsize=(5, 4))
sns.histplot(data["AGE"], bins=10, color="skyblue")
plt.title("Age distribution of respondents", fontsize=14)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig("age_distribution.svg", bbox_inches="tight")
# plt.show()

# Task 1-2
data = data_raw[~data_raw["RELATIONSHIP"].isnull()]
relationship_counts = data["RELATIONSHIP"].value_counts(normalize=True)
# relationship_counts = relationship_counts.sort_values()

plt.figure(figsize=(5, 4))
plt.pie(
    relationship_counts,
    labels=relationship_counts.index,  # 顯示每個類別的名稱
    autopct="%1.1f%%",  # 顯示百分比，格式為小數點後一位
    # colors=plt.cm.Paired.colors,  # 使用預設配色方案
)
plt.title("Relationship Status", fontsize=14)  # 添加標題
plt.tight_layout()  # 自動調整佈局避免重疊

# 保存為矢量圖
plt.savefig("relationship_status_pie.svg", bbox_inches="tight")
# plt.show()

# Task 1-3
data = data_raw[~data_raw["EDUCATION"].isnull()]
data = data[~data["SALARY"].isnull()]

# 將收入分為兩組
data["SALARY"] = data["SALARY"].apply(lambda x: 0 if x == " <=50K" else 1)

# 計算每個教育程度中的收入分佈
education_income = data.groupby(["EDUCATION", "SALARY"]).size().unstack()
# 計算百分比
# education_income_percentage = (
#     education_income.div(education_income.sum(axis=1), axis=0) * 100
# )

# 繪製分組條形圖
education_income.plot(
    kind="bar", figsize=(5, 4), stacked=False, color=["skyblue", "salmon"]
)
plt.title("Educational level and salary", fontsize=14)
plt.xlabel("Education Level", fontsize=12)
plt.ylabel("Number of respondents", fontsize=12)
plt.legend(["<=50K", ">50K"], title="Salary Group")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# 保存為矢量圖
plt.savefig("education_income.svg", bbox_inches="tight")
plt.show()
