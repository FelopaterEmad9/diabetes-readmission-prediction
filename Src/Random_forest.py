import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , roc_auc_score

df = pd.read_csv("../Data/cleaned_data.csv")

X = df.drop(columns=["readmitted_30_days"])
Y= df["readmitted_30_days"]

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=9,stratify=Y)

num_transform = Pipeline(steps=
            [("imputer",SimpleImputer(strategy="median"))
             ])

cat_transform = Pipeline(steps=
            [("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore"))
            ])

preprocessor = ColumnTransformer(transformers=[("num",num_transform,num_cols),
                                               ("cat",cat_transform,cat_cols)
                                               ])

model = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(random_state=9,class_weight="balanced"))
])

params = {
    "classifier__n_estimators":[100,200],
    "classifier__max_depth":[10,20,None],
    "classifier__min_samples_split":[2,5],
    "classifier__min_samples_leaf":[1,2],
    "classifier__max_features":["sqrt","log2"]
}

grid =GridSearchCV(
    estimator = model,
    param_grid = params,
    cv = 3,
    verbose=2,
    scoring="roc_auc",
)

print("Training Random Forest...")
grid.fit(X_train,Y_train)
print("Training finished")

print("\n Best parameters")
print(grid.best_params_)

print("\n Best CV ROC AUC")
print(grid.best_score_)

best_model = grid.best_estimator_

Y_pred = best_model.predict(X_test)
Y_prob = best_model.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(Y_test,Y_pred))

roc_auc = roc_auc_score(Y_test,Y_prob)
print("\nRoc Auc:")
print(roc_auc)
