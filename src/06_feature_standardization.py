import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
plt.savefig("../figures/wine-quality-histogram.png")

plt.figure(figsize=(12,8))
sns.heatmap(wine.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix wine")
plt.savefig("../figures/wine-quality-heatmap.png")

wine["quality"] = (wine["quality"] >= 6).astype(int)

X = wine.drop(columns= ["quality"])
y = wine["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_value_counts = y_train.value_counts(normalize=True)
test_value_counts = y_test.value_counts(normalize=True)

print(train_value_counts)
print(test_value_counts)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)