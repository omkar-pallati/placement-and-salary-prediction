import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

students = pd.read_csv("indian_engineering_student_placement.csv")
targets = pd.read_csv("placement_targets.csv")

df = pd.merge(students, targets, on="Student_ID")

branch_map = {"CSE": 0, "IT": 1, "ECE": 2, "EEE": 3, "MECH": 4}
gender_map = {"Male": 1, "Female": 0}
yes_no_map = {"Yes": 1, "No": 0}
level_map = {"Low": 0, "Medium": 1, "High": 2}

def clean_data(data):
    data = data.copy()

    if "branch" in data.columns:
        data["branch"] = data["branch"].map(branch_map).fillna(0)

    if "gender" in data.columns:
        data["gender"] = data["gender"].map(gender_map).fillna(1)

    if "internet_access" in data.columns:
        data["internet_access"] = data["internet_access"].map(yes_no_map).fillna(1)

    if "part_time_job" in data.columns:
        data["part_time_job"] = data["part_time_job"].map(yes_no_map).fillna(0)

    if "extracurricular_involvement" in data.columns:
        data["extracurricular_involvement"] = data["extracurricular_involvement"].map(level_map).fillna(1)

    if "family_income_level" in data.columns:
        data["family_income_level"] = data["family_income_level"].map(level_map).fillna(1)

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.fillna(0)

    return data

X = df.drop(columns=["Student_ID", "placement_status", "salary_lpa"])
X = clean_data(X)

y_class = df["placement_status"].map({
    "Placed": 1,
    "Not Placed": 0
})

placed_df = df[df["placement_status"] == "Placed"]

X_reg = placed_df.drop(columns=["Student_ID", "placement_status", "salary_lpa"])
X_reg = clean_data(X_reg)

y_reg = placed_df["salary_lpa"]

clf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

reg_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

clf_model.fit(X, y_class)
reg_model.fit(X_reg, y_reg)

joblib.dump(clf_model, "model.pkl")
joblib.dump(reg_model, "reg_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("Models trained and saved successfully!")