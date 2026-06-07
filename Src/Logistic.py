import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , roc_auc_score

df = pd.read_csv("../Data/cleaned_data.csv")

X = df.drop(columns=["readmitted_30_days"])
Y= df["readmitted_30_days"]

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=9,stratify=Y)

num_transform = Pipeline(steps=
            [("imputer",SimpleImputer(strategy="median")),
             ("scaler",StandardScaler())
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
    ("classifier",LogisticRegression(max_iter=3000,solver="lbfgs"))
])

print("Training...")
model.fit(X_train,Y_train)
print("Training finished")

Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(Y_test,Y_pred))

roc_auc = roc_auc_score(Y_test,Y_prob)
print("\nRoc Auc:")
print(roc_auc)
