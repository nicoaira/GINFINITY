# src/hp/generate_grid.py
import itertools
import pandas as pd

learning_rates = [0.01, 0.001, 0.0001]
hidden_dims = [64, 128, 256, 512]
gin_layers_list = [3, 4, 5]

grid = list(itertools.product(learning_rates, hidden_dims, gin_layers_list))
df = pd.DataFrame(grid, columns=["lr", "hidden_dim", "gin_layers"])
df.to_csv("src/hp/grid.csv", index=False)
