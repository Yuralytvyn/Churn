import pandas as pd
from pandas import DataFrame

from init import DATA_PATH
from sklearn import preprocessing
import numpy as np


def delete_columns(df, data_path=DATA_PATH) -> None:
    df = df.drop(columns=["customerID"])
    df.to_csv(data_path + "deleted_columns.csv", index=False)


def get_numerical_features(df, data_path=DATA_PATH) -> None:
    for col in df:
        if df[col].dtype == "str":
            df[col] = df[col].astype("category").cat.codes
    df.to_csv(data_path + "numerical_features.csv", index=False)
def get_X_y(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

def scale_features(
    df,
    scaler,
    fit=False,
) -> pd.DataFrame:

    df_np = np.array(df)

    if fit:
        df_np = scaler.fit_transform(df_np)
    else:
        df_np = scaler.transform(df_np)

    return pd.DataFrame(
        df_np,
        columns=df.columns,
        index=df.index,
    )
