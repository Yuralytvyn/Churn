import os

import pandas as pd

from data_processing import delete_columns, get_numerical_features, scale_features,get_X_y
from init import DATA_PATH, RATIO, SCALER, MODEL
from train_test_split import divide_df_into_train_validation_test_set
from model_training import train_model
raw_data_path = os.path.join(
    DATA_PATH,
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)

deleted_columns_path = os.path.join(
    DATA_PATH,
    "deleted_columns.csv",
)

numerical_features_path = os.path.join(
    DATA_PATH,
    "numerical_features.csv",
)


df = pd.read_csv(raw_data_path)

if not os.path.exists(deleted_columns_path):
    delete_columns(df)
    print("Created csv file with deleted columns")

df_clean = pd.read_csv(deleted_columns_path)

if not os.path.exists(numerical_features_path):
    get_numerical_features(df_clean)
    print("Created csv file with numerical columns")

df_numerical = pd.read_csv(numerical_features_path)

train_set, val_set, test_set = divide_df_into_train_validation_test_set(
    df_numerical,
    RATIO,
)

print("Created train_set, val_set and test_set")
X_train, y_train = get_X_y(train_set)
print("Created X_train, y_train")
X_val, y_val = get_X_y(val_set)
print("Created X_val, y_val")
X_test, y_test = get_X_y(test_set)
print("Created X_test, y_test")
print("------------------------------")


scaled_X_train = scale_features(X_train, SCALER, fit=True)
print("Scaled train_X_set")
scaled_X_val = scale_features(X_val, SCALER)
print("Scaled val_X_set")
scaled_X_test = scale_features(X_test, SCALER)
print("Scaled test_X_set")

train_model(MODEL,scaled_X_train, y_train, scaled_X_val, y_val, scaled_X_test, y_test)


