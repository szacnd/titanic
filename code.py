import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("titanic.csv")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].fillna("Unknown")
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

y = df["Survived"]
X = df.drop(columns=["Survived"])

num_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Deck"]

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

model = LogisticRegression(max_iter=2000)

clf = Pipeline(steps=[("preprocess", preprocess),
                     ("model", model)])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=101)

scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
print("ROC AUC CV:", scores.mean(), "+/-", scores.std())
