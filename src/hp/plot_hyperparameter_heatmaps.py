import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Ruta del archivo de resultados
CSV_PATH = "src/hp/hp_results/grid_search_results.csv"
assert os.path.exists(CSV_PATH), f"No se encontró el archivo: {CSV_PATH}"

# Cargar resultados
df = pd.read_csv(CSV_PATH)

# Asegurar que no haya datos faltantes
df = df.dropna(subset=["val_auc", "val_triplet_loss"])

# Crear carpeta de gráficos
os.makedirs("src/hp/hp_results/plots", exist_ok=True)

# Variables únicas
learning_rates = sorted(df["learning_rate"].unique())
metrics = ["val_auc", "val_triplet_loss"]

# Loop por cada métrica
for metric in metrics:
    for lr in learning_rates:
        subset = df[df["learning_rate"] == lr]
        if subset.empty:
            continue

        pivot_table = subset.pivot(index="gin_layers", columns="hidden_dim", values=metric)

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".3f")
        plt.title(f"{metric.replace('_', ' ').title()} (lr = {lr})")
        plt.ylabel("GIN Layers")
        plt.xlabel("Hidden Dim")
        plt.tight_layout()

        plot_filename = f"src/hp/hp_results/plots/heatmap_{metric}_lr{lr}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"📊 Guardado: {plot_filename}")


# 🎯 Scatterplot: AUC vs Triplet Loss
plt.figure(figsize=(8, 6))

# Asignar un color distinto a cada learning rate
lr_values = sorted(df["learning_rate"].unique())
colors = cm.get_cmap("Set1", len(lr_values))

for i, lr in enumerate(lr_values):
    subset = df[df["learning_rate"] == lr]
    plt.scatter(subset["val_triplet_loss"], subset["val_auc"], 
                label=f"lr={lr}", color=colors(i), s=80, edgecolor='k', alpha=0.8)

plt.xlabel("Validation Triplet Loss")
plt.ylabel("Validation AUC")
plt.title("AUC vs Triplet Loss por combinación de hiperparámetros")
plt.legend(title="Learning Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("src/hp/hp_results/plots/scatter_auc_vs_triplet_loss.png")
plt.close()
print("📊 Guardado: src/hp/hp_results/plots/scatter_auc_vs_triplet_loss.png")
