from datafarmer.graph import generate_box_plot
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

def open_figure(file_path):
    if os.path.exists(file_path):
        subprocess.run(["open", file_path])
    else:
        print(f"File not found: {file_path}")

def test_generate_pair_plot(tmp_path):
    df =  pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C'], size=100),
        'Value': np.random.randn(100)
    })

    fig = generate_box_plot(df=df, labels_name='Category', values_name='Value')
    assert isinstance(fig, plt.Figure)

    file_path = tmp_path / "box_plot.png"
    fig.savefig(file_path)
    assert file_path.exists()
    open_figure(file_path)