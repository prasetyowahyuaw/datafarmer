import matplotlib.pyplot as plt
import pandas as pd

def generate_box_plot(df: pd.DataFrame, labels_name: str= None, values_name: str= None) -> plt.Figure:
    """
    Generate a boxplot from a pandas DataFrame.

    Args:
        data (pd.DataFrame): A pandas DataFrame.
        labels_name (str): The name of the column containing the labels.
        values_name (str): The name of the column containing the values.

    Returns:
        plt.Figure: A matplotlib figure object.
    """

    assert isinstance(df, pd.DataFrame), "data must be a pandas DataFrame"
    assert len(df.select_dtypes(include=["float64", "int64"]).columns) > 0, "data must contain at least one numerical column"

    # validation
    df["labels"] = df[labels_name] if labels_name else "all" # if none, it will get "all" as the label
    df["values"] = df[values_name] if values_name else df[df.select_dtypes(include=["float64", "int64"]).columns[0]] # if none, it will take the first numerical column

    categories = df["labels"].sort_values().unique()
    values = [df[df["labels"] == label]["values"] for label in categories]

    fig, ax = plt.subplots(figsize=(
        len(categories)*2, 
        7
    ))

    box_props = ax.boxplot(
        values, 
        labels=categories, 
        patch_artist=True, 
        showfliers=False,
        boxprops=dict(facecolor="lightblue", color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="red", linewidth=2)
    )

    for i, box in enumerate(box_props["boxes"]):

        # Get box properties
        median = box_props["medians"][i].get_ydata()[0]
        lower_whisker = box_props["whiskers"][i*2].get_ydata()[1]
        upper_whisker = box_props["whiskers"][i*2+1].get_ydata()[1]
        lower_quartile = box_props["boxes"][i].get_path().vertices[0,1]
        upper_quartile = box_props["boxes"][i].get_path().vertices[2,1]

        def add_annotation(y, text, color, offset=0.2):
            """Helper function to add annotations with dynamic positioning."""
            ax.text(
                i + 1, 
                y + offset, 
                text, 
                ha='center', 
                va='bottom', 
                color=color, 
                fontsize=8, 
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3'))

        # Add text annotations
        add_annotation(median, f'Median: {median:.2f}', 'darkred', offset=0.1)
        add_annotation(lower_quartile, f'Q1: {lower_quartile:.2f}', 'darkblue', offset=-0.2)
        add_annotation(upper_quartile, f'Q3: {upper_quartile:.2f}', 'darkblue', offset=0.1)
        add_annotation(lower_whisker, f'Lower : {lower_whisker:.2f}', 'grey', offset=-0.2)
        add_annotation(upper_whisker, f'Upper : {upper_whisker:.2f}', 'grey', offset=0.1)
    
    ax.set_title(f"Boxplot by {labels_name} and {values_name}")
    ax.set_xlabel(labels_name)
    ax.set_ylabel(values_name)
    ax.grid(True)

    return fig