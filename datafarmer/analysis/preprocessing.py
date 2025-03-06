import pandas as pd


def get_features_info(df: pd.DataFrame) -> pd.DataFrame:
    """return the features information of the given DataFrame.
    the information includes the data type, total unique value, and list of unique values

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: features information dataframe
    """

    features = [
        [column, df[column].dtypes, df[column].nunique(), df[column].unique().tolist()]
        for column in df.columns
    ]
    df_features = pd.DataFrame(
        features, columns=["Feature", "Dtypes", "Unique Values", "Values"]
    )

    return df_features


def get_null_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """return the null proportions information of the given DataFrame.
    it only returns the columns that have null values proportion greater than 0.0

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: null information dataframe
    """

    null_samples = df.isnull().sum()
    null_propoportion = null_samples / len(df)
    null_info = pd.DataFrame(
        {"Null Samples": null_samples, "Null Proportion": null_propoportion}
    )

    return null_info[null_propoportion > 0.0]
