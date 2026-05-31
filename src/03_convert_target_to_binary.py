import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wine = pd.read_csv('../data/wine.csv', delimiter=";")
print(wine.columns)
print(wine["quality"].unique())

wine_nas = wine.isnull().sum()
print(wine_nas)

plt.figure(figsize=(10,6))
sns.histplot(wine["quality"], bins=7, color="red", kde=False)
plt.title('Quality wine')
plt.xlabel("Quality")
plt.ylabel("Quantity of wine")
plt.savefig("../figures/wine-quality-histpgram.png")

plt.figure(figsize=(12,8))
sns.heatmap(wine.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix wine")
#plt.savefig("../figures/wine-quality-heatmap.png")

wine["quality"] = (wine["quality"] >= 6).astype(int)