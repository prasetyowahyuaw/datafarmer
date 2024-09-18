from datafarmer.graph import generate_pair_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

def open_figure(file_path):
    if os.path.exists(file_path):
        subprocess.run(["open", file_path])
    else:
        print(f"File not found: {file_path}")

def test_generate_pair_plot(tmp_path):
    df = pd.DataFrame({
        'feature_1': np.random.randint(1, 100, 100),         # Random integers between 1 and 100
        'feature_2': np.random.randint(1, 1000, 100),        # Random integers between 1 and 1000
        'feature_3': np.random.randint(1, 5000, 100),        # Random integers between 1 and 5000
        'feature_4': np.random.randn(100) * 1000,            # Normally distributed random values            # Random floats between 0 and 200
    })

    fig = generate_pair_plot(df)
    assert isinstance(fig, plt.Figure)

    file_path = tmp_path / "pair_plot.png"
    fig.savefig(file_path)
    assert file_path.exists()
    open_figure(file_path)