import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = pd.read_csv("C:/Users/user/OneDrive - United States International University (USIU)/Documents/School Notes/DSA project/MentalHealthPrediction App/Mental Health Dataset.csv")

# Select features and target
X = data[["family_history", "Days_Indoors", "Growing_Stress"]]
y = data["treatment"].map({"Yes": 1, "No": 0})

# Convert object columns to categorical or numeric
X = X.apply(lambda col: col.astype("category").cat.codes if col.dtypes == "object" else pd.to_numeric(col, errors="coerce"))

# Handle any missing values (optional, but recommended)
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train({"objective": "binary:logistic"}, dtrain, num_boost_round=50)

# Save model
model.save_model("xgboost_model.json")




