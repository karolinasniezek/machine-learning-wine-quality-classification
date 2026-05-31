from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

X = wine.drop(columns= ["quality"])
y = wine["quality"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(pd.Series(y_train).value_counts(normalize=True))
print(pd.Series(y_test).value_counts(normalize=True))

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42)
}

# Słowniki do przechowywania wyników
metrics = {"model": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

predictions = {name: None for name in models}

# Funkcja oceny modeli
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
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

fig, axes = plt.subplots(1,3, figsize=(18, 5))

for i, (model_name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"Confusion Matrix - {model_name}")
    axes[i].set_xlabel("Predictions")
    axes[i].set_ylabel("Reality")


plt.tight_layout()
plt.savefig("../figures/classifications_models.png")


