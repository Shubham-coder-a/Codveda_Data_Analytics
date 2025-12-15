import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("iris_cleaned.csv")
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())
print("\nSpecies count:")
print(df['species'].value_counts())
df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(columns=['species']))
plt.title("Boxplot of Iris Features")
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species"
)
plt.title("Petal Length vs Petal Width")
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['species']).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
