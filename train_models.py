# importimi i librarive kryesore për manipulim dhe analiza
import pandas as pd
import numpy as np

# libraritë për ndarje të të dhënave dhe preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# metrikat për vlerësimin e modeleve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# modelet e machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# për vizualizim
import matplotlib.pyplot as plt


# =========================
# LOAD DATA
# =========================

# lexon dataset-in nga file csv dhe e kthen në DataFrame
df = pd.read_csv("data/heart_disease.csv")

# printon dimensionet e dataset-it (rreshta, kolona)
print("Dataset Shape:", df.shape)


# =========================
# TARGET DETECTION
# =========================

# fillimisht nuk e dimë cili është target-i
target_col = None

# kërkon në kolonat e dataset-it për emra të mundshëm të target-it
for col in df.columns:
    if col.lower() in ['target', 'output', 'heart disease status', 'heartdisease']:
        target_col = col
        break

# nëse nuk gjendet, merret kolona e fundit si target
if target_col is None:
    target_col = df.columns[-1]

# printon emrin e target kolonës
print("Target column:", target_col)


# ndan dataset-in në features (X) dhe target (y)
X_raw = df.drop(columns=[target_col])  # inputet
y_raw = df[target_col]                 # outputi


# =========================
# CLEAN TARGET
# =========================

# nëse target është tekstual (Yes/No), e konverton në numra (0,1)
if y_raw.dtype == 'object':
    y_raw = y_raw.astype('category').cat.codes

# mbush vlerat që mungojnë me 0
y_raw = pd.Series(y_raw).fillna(0)


# =========================
# BEFORE PREPROCESSING
# =========================

# kopje e dataset-it për testim pa preprocessing
X_before = X_raw.copy()

# konverton çdo vlerë në numër, gabimet bëhen NaN
X_before = X_before.apply(pd.to_numeric, errors='coerce')

# mbush NaN me 0
X_before = X_before.fillna(0)

# ndarja në train dhe test
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_before, y_raw, test_size=0.2, random_state=42
)

# definimi i modeleve
models_before = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# dictionary për ruajtjen e rezultateve
results_before = {}

print("\nBEFORE PREPROCESSING")

# trajnim dhe testim për secilin model
for name, model in models_before.items():

    # trajnon modelin
    model.fit(X_train_b, y_train_b)

    # bën parashikime
    y_pred = model.predict(X_test_b)

    # llogarit accuracy
    acc = accuracy_score(y_test_b, y_pred)

    # ruan rezultatin
    results_before[name] = acc

    # printon rezultatin me 4 shifra pas presjes
    print(f"{name}: {acc:.4f}")


# =========================
# AFTER PREPROCESSING
# =========================

# kopje e dataset-it për preprocessing të plotë
X_after = X_raw.copy()

# konvertim në numra
X_after = X_after.apply(pd.to_numeric, errors='coerce')

# zëvendëson vlerat negative me NaN
X_after = X_after.mask(X_after < 0)

# mbush NaN me median (më e qëndrueshme ndaj outliers)
X_after = X_after.fillna(X_after.median(numeric_only=True))

# nëse ende ka NaN, i mbush me 0
X_after = X_after.fillna(0)

# trajtimi i outliers me metodën IQR
for col in X_after.columns:
    Q1 = X_after[col].quantile(0.25)   # kuartili i parë
    Q3 = X_after[col].quantile(0.75)   # kuartili i tretë
    IQR = Q3 - Q1                      # diferenca

    # kufijtë për outliers
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # kufizon vlerat brenda intervalit
    X_after[col] = X_after[col].clip(lower, upper)

# normalizimi i të dhënave
scaler = StandardScaler()
X_after = scaler.fit_transform(X_after)

# ndarja në train/test
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_after, y_raw, test_size=0.2, random_state=42
)

# definimi i modeleve pas preprocessing
models_after = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,        # numri i pemëve
        class_weight='balanced', # balancim i klasave
        random_state=42
    )
}

results_after = {}

print("\nAFTER PREPROCESSING")

# trajnim dhe testim
for name, model in models_after.items():

    model.fit(X_train_a, y_train_a)
    y_pred = model.predict(X_test_a)

    acc = accuracy_score(y_test_a, y_pred)
    results_after[name] = acc

    # printon accuracy
    print(f"\n{name}: {acc:.4f}")

    # printon confusion matrix
    print(confusion_matrix(y_test_a, y_pred))

    # printon precision, recall, f1-score
    print(classification_report(y_test_a, y_pred, zero_division=0))


# =========================
# GRAFIKU KRAHASUES
# =========================

# merr emrat e modeleve
labels = list(results_before.keys())

# merr vlerat para dhe pas preprocessing
before_vals = list(results_before.values())
after_vals = list(results_after.values())

# krijon pozicionet në boshtin X
x = np.arange(len(labels))

plt.figure()

# grafiku për para preprocessing
plt.bar(x - 0.2, before_vals, width=0.4)

# grafiku për pas preprocessing
plt.bar(x + 0.2, after_vals, width=0.4)

# emrat në boshtin X
plt.xticks(x, labels, rotation=20)

# titulli
plt.title("Effect of Preprocessing on Model Performance")

# legjenda
plt.legend(["Before", "After"])

# kufiri i boshtit Y
plt.ylim(0, 1)

# shfaq grafikun
plt.show()


# =========================
# FINAL SUMMARY
# =========================

print("\nFINAL SUMMARY")

# printon rezultatet për secilin model
for model in labels:
    print(f"{model}: Before = {results_before[model]:.4f} | After = {results_after[model]:.4f}")