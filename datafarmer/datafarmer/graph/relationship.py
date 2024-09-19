import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

def plot_single_pair(
    ax: plt.axes, 
    feature_i_1: np.ndarray, 
    feature_i_2: np.ndarray, 
    X: np.ndarray, 
    y: np.ndarray, 
    features: list
) -> None:
    """

    Args:
        ax (plt.axes): matplotlib axis
        feature_i_1 (np.ndarray): index of first feature to be plotted
        feature_i_2 (np.ndarray): index of second feature to be plotted
        X (np.ndarray): feature dataset of shape m x n
        y (np.ndarray): feature dataset of shape 1 x n
        features (list): list of n feature title
    """

    # plot distribution histogram if the feature are the same (diagonal of the pair-plot)
    if feature_i_1 == feature_i_2:
        tdf = pd.DataFrame(
        X[:, [feature_i_1]], 
        columns=[features[feature_i_1]])
        tdf["target"] = y
        ax[feature_i_1, feature_i_2].hist(tdf[features[feature_i_1]], bins=30)
    
    else:
        # otherwise plot the pair-wise scatter plot
        tdf = pd.DataFrame(
        X[:, [feature_i_1, feature_i_2]], 
        columns=[features[feature_i_1], features[feature_i_2]])
        tdf["target"] = y

        # calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
        tdf[features[feature_i_2]].astype(float),
        tdf[features[feature_i_1]].astype(float))
        line = slope*tdf[features[feature_i_2]]+intercept

        # plot scatter plot and regression line
        ax[feature_i_1, feature_i_2].scatter(
        x=tdf[features[feature_i_2]],
        y=tdf[features[feature_i_1]]
        )
        ax[feature_i_1, feature_i_2].plot(tdf[features[feature_i_2]],line,"r-",label="Regression line")

        # add r-quared and the equation value label
        r_squared = r_value**2
        equation = f"y = {slope:.2f}x + {intercept:.2f}"
        ax[feature_i_1, feature_i_2].annotate(f"R-squared = {r_squared:.2f}\n{equation}", xy=(0.05, 0.8), xycoords="axes fraction")
        
    # printing the feature labels only on the left side and the bottom side
    if feature_i_1 == len(features) - 1:
        ax[feature_i_1, feature_i_2].set(xlabel=features[feature_i_2], ylabel="")
    if feature_i_2 == 0:
        if feature_i_1 == len(features) - 1:
            ax[feature_i_1, feature_i_2].set(
                xlabel=features[feature_i_2], 
                ylabel=features[feature_i_1]
            )
        else:
            ax[feature_i_1, feature_i_2].set(
                xlabel="", 
                ylabel=features[feature_i_1]
            )

def my_plot_grid(X:np.ndarray, y:np.ndarray, features:list) -> plt.figure:
    """plots a pair grid of the given features. then return the figure

    Args:
        X (np.ndarray): dataset of shape m x n
        y (np.ndarray): target list of shape 1 x n
        features (list): list of n features title
    """

    feature_len = len(features)
    
    # create a matplot subplot area with the size of [feature len x feature len]
    fig, axis = plt.subplots(nrows=feature_len, ncols=feature_len)

    # setting figure size helps to optimize the figure size according to the feature len
    fig.set_size_inches(feature_len * 4, feature_len * 4)

    # iterate through features to plot pairwaise
    for i in range(0, feature_len):
        for j in range(0, feature_len):
            plot_single_pair(axis, i, j, X, y, features)
    
    return fig

def generate_pair_plot(df: pd.DataFrame) -> plt.figure:
    """generate pair plot of the given dataframe

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        plt.figure: pair plot figure
    """

    assert isinstance(df, pd.DataFrame), "data must be a pandas DataFrame"

    # force a target column with 1 value
    if "target" not in df.columns:
        df["target"] = 1

    # separate the features and target
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()

    # get the features list
    features = X.columns.tolist()
    X = X.to_numpy()

    # plot the pair grid
    fig = my_plot_grid(X, y, features)

    return fig
