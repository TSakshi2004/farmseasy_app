import pandas as pd
import os, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "crop-disease-flask\data\crop_diseases.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
# Standardize column names (strip)
df.columns = [c.strip() for c in df.columns]

# Keep only required columns if present
required = ["Crop Stage","Region","Crop Disease","Cause"]
for col in required:
    if col not in df.columns:
        df[col] = "unknown"

# Preprocess simple: lower, strip
for c in required:
    df[c] = df[c].astype(str).str.strip().str.lower()

# Model A: Inputs = Crop Stage + Region -> predict Crop Disease
le_stage_A = LabelEncoder(); le_region_A = LabelEncoder(); le_disease_A = LabelEncoder()
X_A = pd.DataFrame({
    "stage": le_stage_A.fit_transform(df["Crop Stage"]),
    "region": le_region_A.fit_transform(df["Region"])
})
y_A = le_disease_A.fit_transform(df["Crop Disease"])
X_train, X_test, y_train, y_test = train_test_split(X_A, y_A, test_size=0.2, random_state=42, stratify=y_A)
clf_A = RandomForestClassifier(n_estimators=200, random_state=42)
clf_A.fit(X_train, y_train)
y_pred = clf_A.predict(X_test)
print("Model A accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_disease_A.classes_))

with open(os.path.join(MODELS_DIR, "model_A.pkl"), "wb") as f:
    pickle.dump({"model": clf_A, "le_stage": le_stage_A, "le_region": le_region_A, "le_disease": le_disease_A}, f)

# Model B: Inputs = Crop Stage + Region -> predict Crop Disease AND Cause (multioutput)
le_stage_B = LabelEncoder(); le_region_B = LabelEncoder()
le_disease_B = LabelEncoder(); le_cause_B = LabelEncoder()
X_B = pd.DataFrame({
    "stage": le_stage_B.fit_transform(df["Crop Stage"]),
    "region": le_region_B.fit_transform(df["Region"])
})
y_B1 = le_disease_B.fit_transform(df["Crop Disease"])
y_B2 = le_cause_B.fit_transform(df["Cause"])
# We'll train two separate classifiers and save them together
clf_B_disease = RandomForestClassifier(n_estimators=200, random_state=42)
clf_B_cause = RandomForestClassifier(n_estimators=200, random_state=42)
clf_B_disease.fit(X_B, y_B1)
clf_B_cause.fit(X_B, y_B2)

with open(os.path.join(MODELS_DIR, "model_B.pkl"), "wb") as f:
    pickle.dump({"clf_disease": clf_B_disease, "clf_cause": clf_B_cause,
                 "le_stage": le_stage_B, "le_region": le_region_B,
                 "le_disease": le_disease_B, "le_cause": le_cause_B}, f)

# Model C: Inputs = Crop Stage + Region + Cause -> predict Crop Disease
le_stage_C = LabelEncoder(); le_region_C = LabelEncoder(); le_cause_C = LabelEncoder(); le_disease_C = LabelEncoder()
X_C = pd.DataFrame({
    "stage": le_stage_C.fit_transform(df["Crop Stage"]),
    "region": le_region_C.fit_transform(df["Region"]),
    "cause": le_cause_C.fit_transform(df["Cause"])
})
y_C = le_disease_C.fit_transform(df["Crop Disease"])
X_train, X_test, y_train, y_test = train_test_split(X_C, y_C, test_size=0.2, random_state=42, stratify=y_C)
clf_C = RandomForestClassifier(n_estimators=200, random_state=42)
clf_C.fit(X_train, y_train)
y_pred = clf_C.predict(X_test)
print("Model C accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_disease_C.classes_))

with open(os.path.join(MODELS_DIR, "model_C.pkl"), "wb") as f:
    pickle.dump({"model": clf_C, "le_stage": le_stage_C, "le_region": le_region_C, "le_cause": le_cause_C, "le_disease": le_disease_C}, f)

print("Saved model_A.pkl, model_B.pkl, model_C.pkl in", MODELS_DIR)