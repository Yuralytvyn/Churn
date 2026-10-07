import pandas as pd
from sklearn.model_selection import train_test_split


import pandas as pd
from sklearn.model_selection import train_test_split


def divide_df_into_train_validation_test_set(
    df, ratio: list[float] = [0.7, 0.15, 0.15]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_set, temp_set = train_test_split(df, test_size=ratio[1] + ratio[2])

    val_set, test_set = train_test_split(
        temp_set, test_size=ratio[2] / (ratio[1] + ratio[2])
    )

    return train_set, val_set, test_set
