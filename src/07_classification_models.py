# Inicjalizacja modeli
from h5py.h5pl import append
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
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

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# Słowniki do przechowywania wyników
metrics = {"model": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

predictions = {name: None for name in models}

# Funkcja oceny modeli
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Poniżej umieść swoje rozwiązanie
for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    metrics["model"].append(model_name)
    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)

results_df = pd.DataFrame(metrics)
print(results_df)