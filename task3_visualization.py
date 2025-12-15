import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("iris_cleaned.csv")

df.drop(columns=["species"]).hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("outputs/histograms.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(columns=["species"]))
plt.title("Boxplot of Iris Features")
plt.savefig("outputs/boxplot.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species"
)
plt.title("Petal Length vs Petal Width")
plt.savefig("outputs/scatter_plot.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(
    df.drop(columns=["species"]).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.close()

print("All visualizations saved successfully in outputs folder!")
