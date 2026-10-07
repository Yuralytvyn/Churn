import os
import pandas as pd

from init import DATA_PATH
from data_processing import *

df = pd.read_csv(DATA_PATH + "WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not os.path.exists(DATA_PATH + "deleted_columns.csv"):
    delete_columns(df)

if not os.path.exists(DATA_PATH + "numerical_features.csv"):
    get_numerical_features(df)

