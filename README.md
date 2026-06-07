# Diabetes Readmission Prediction

## Project idea

The project is about predicting if a diabetic patient will come back to the hospital within 30 days or not.

I changed the original `readmitted` column into a new column called `readmitted_30_days`.

The values became:

- `1` if the patient was readmitted in less than 30 days
- `0` if the patient was not readmitted within 30 days

So the project became a binary classification problem.

---

## Dataset

I used the Diabetes 130-US hospitals dataset.

The dataset has information about diabetic patients, like age, gender, admission type, number of medications, diagnosis codes, and readmission result.

---

## Data cleaning

The cleaning was done in:

```text
src/Preprocess.py
```

In this file I did these steps:

1. Read the original dataset.
2. Created the target column `readmitted_30_days`.
3. Replaced `?` with `NaN`.
4. Removed columns that were not useful for the model.
5. Saved the cleaned data in:

```text
Data/cleaned_data.csv
```

The columns removed were:

```text
encounter_id
patient_nbr
readmitted
weight
payer_code
medical_specialty
max_glu_serum
A1Cresult
```

---

## Preprocessing

After cleaning, I used the cleaned data in the model scripts.

I split the data into:

```text
X = features
y = target
```

Numerical columns were handled using:

```text
SimpleImputer(strategy="median")
```

Categorical columns were handled using:

```text
SimpleImputer(strategy="most_frequent")
```

Then I used:

```text
OneHotEncoder(handle_unknown="ignore")
```

to change text columns into numbers.

For Logistic Regression, I also used `StandardScaler` because this model works better when numbers are scaled.

---

## Models

I trained two models.

### 1. Logistic Regression

This was the first model.

File:

```text
src/Logistic.py
```

### 2. Random Forest

This was the second model.

I also used `GridSearchCV` to try more than one setting and choose the best one.

File:

```text
src/Random_forest.py
```

---

## Evaluation

I used:

- accuracy
- precision
- recall
- f1-score
- ROC AUC

Accuracy alone was not enough because the data is imbalanced.

Most patients are class `0`, so a model can get high accuracy by predicting `0` most of the time.

---

## Results

### Logistic Regression

```text
Accuracy: 0.89
ROC AUC: 0.6364
Class 1 Recall: 0.02
Class 1 F1-score: 0.04
```

### Random Forest

```text
Accuracy: 0.70
ROC AUC: 0.6498
Class 1 Recall: 0.45
Class 1 F1-score: 0.25
```

---

## Comparison

Logistic Regression had better accuracy.

Random Forest had better ROC AUC and better recall for class `1`.

Random Forest detected more patients who were readmitted within 30 days.

Because of that, We selected Random Forest as the final model.

---

## Final conclusion

The final model is Random Forest.

The results show that accuracy can be misleading in this dataset because most patients are not readmitted within 30 days.

Logistic Regression had higher accuracy, but Random Forest was better for this project because it had higher ROC AUC and much better recall for class `1`.

A better future improvement would be to handle the imbalance problem, for example by trying SMOTE or changing the classification threshold.

---

## Project files

```text
ML_Project/
│
├── Data/
│   ├── diabetic_data.csv
│   └── cleaned_data.csv
│
│
├── Reports/
│   └── final_comparison.txt
│
├── Src/
│   ├── Preprocess.py
│   ├── Logistic.py
│   ├── Random_forest.py
│   └── comparison.py
│
└── README.md
```

---

## How to run

Run the files in this order:

```bash
cd Src
python Preprocess.py
python Logistic.py
python Random_forest.py
```
