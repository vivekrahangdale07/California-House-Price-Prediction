import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("housing.csv")
print("Original columns:")
print(df.columns.tolist())


# 2. Remove Unnecessary Column

# This is only the CSV index, not a useful ML feature
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# 3. Create Stratified Split

# income_category is used only to make the train/test split
# representative of different income groups.

if "income_category" in df.columns:

    train_set, test_set = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["income_category"]
    )

else:

    train_set, test_set = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )


# 4. Separate Features & Target

target = "median_house_value"

X_train = train_set.drop(columns=[target])
y_train = train_set[target]

X_test = test_set.drop(columns=[target])
y_test = test_set[target]


# income_category was only needed for stratification.
# Remove it before training the model.

if "income_category" in X_train.columns:
    X_train = X_train.drop(columns=["income_category"])

if "income_category" in X_test.columns:
    X_test = X_test.drop(columns=["income_category"])


print("\nTraining columns:")
print(X_train.columns.tolist())


# 5. Define Feature Types

numeric_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

categorical_features = [
    "ocean_proximity"
]


# 6. Numerical Pipeline


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 7. Categorical Pipeline

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# 8. Preprocessing Pipeline


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 9. Random Forest Model


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# 10. Complete ML Pipeline

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# 11. Train Model

print("\nTraining model...")

full_pipeline.fit(X_train, y_train)


# 12. Evaluate Model

predictions = full_pipeline.predict(X_test)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Evaluation")
print("----------------")
print("RMSE:", rmse)
print("R2 Score:", r2)


# 13. Save Model

joblib.dump(
    full_pipeline,
    "model.pkl"
)

print("\nmodel.pkl saved successfully!")


# 14. Save Preprocessor

# Extract preprocessing part separately
preprocessing_pipeline = full_pipeline.named_steps["preprocessor"]

joblib.dump(
    preprocessing_pipeline,
    "pipelen.pkl"
)

print("pipelen.pkl saved successfully!")

print("\nTraining completed.")