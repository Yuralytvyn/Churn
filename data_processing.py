import pandas as pd
from init import DATA_PATH

def delete_columns(df, data_path=DATA_PATH)->pd.DataFrame:
    df = df.drop(columns=["customerID"])
    df.to_csv(data_path + "deleted_columns.csv", index=False)

def get_numerical_features(df, data_path=DATA_PATH)->pd.DataFrame:
    for col in df:
        if df[col].dtype == "str":
            df[col] = df[col].astype("category").cat.codes
    df.to_csv(data_path + "numerical_features.csv", index=False)


