import pandas as pd

def get_features_info(df: pd.DataFrame) -> pd.DataFrame:
    """return the features information of the given DataFrame. 
    the information includes the data type, total unique value, and list of unique values

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: features information dataframe
    """

    features = [[column, df[column].dtypes, df[column].nunique(), df[column].unique().tolist()] for column in df.columns]
    df_features = pd.DataFrame(features, columns=['Feature', 'Dtypes', 'Unique Values', 'Values'])

    return df_features