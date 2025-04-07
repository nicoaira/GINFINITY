import os
import torch
import itertools
import pandas as pd
from torch import optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from src.gin_rna_dataset import GINRNADataset
from src.model.gin_model import GINModel
from src.triplet_loss import TripletLoss
from train_model_v2 import (
    train_model_with_early_stopping,
    remove_invalid_structures,
    log_setup,
    log_information
)

# Hiperparámetros a optimizar (fase 1)
learning_rates = [0.01, 0.001, 0.0001]
hidden_dims = [64, 128, 256, 512]
gin_layers_list = [3, 4, 5]

# Configuraciones fijas
OUTPUT_DIM = 128
BATCH_SIZE = 100
NUM_EPOCHS = 10
PATIENCE = 5
MIN_DELTA = 0.001
DECAY_RATE = 0.01
POOLING_TYPE = "global_add_pool"
DROPOUT = 0.0
GRAPH_ENCODING = "standard"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rutas a los datasets
TRAIN_PATH = "/Users/mercedescastro/tesis_meki/GINFINITY/example_data/train.csv"
# TRAIN_PATH = "/Users/mercedescastro/tesis_meki/GINFINITY/example_data/dummy_train.csv"
VAL_PATH = "/Users/mercedescastro/tesis_meki/GINFINITY/example_data/val_dataset.csv"
# VAL_PATH = "/Users/mercedescastro/tesis_meki/GINFINITY/example_data/dummy_train.csv"

assert os.path.exists(TRAIN_PATH), f"train.csv no encontrado"
assert os.path.exists(VAL_PATH), f"val_dataset.csv no encontrado"

# Cargar datasets
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

train_df = remove_invalid_structures(train_df)
val_df = remove_invalid_structures(val_df)

# Crear datasets y dataloaders
train_dataset = GINRNADataset(train_df, graph_encoding=GRAPH_ENCODING)
val_dataset = GINRNADataset(val_df, graph_encoding=GRAPH_ENCODING)

train_loader = GeoDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = GeoDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

result_folder = "src/hp/hp_results/"
# Crear carpeta de resultados
os.makedirs(result_folder, exist_ok=True)
results = []

# Búsqueda en grilla
for lr, hidden_dim, gin_layers in itertools.product(learning_rates, hidden_dims, gin_layers_list):
    model_id = f"gin_lr{lr}_dim{hidden_dim}_layers{gin_layers}"
    output_dir = f"output/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = f"{output_dir}/train.log"
    log_setup(log_path)

    model = GINModel(
        hidden_dim=hidden_dim,
        output_dim=OUTPUT_DIM,
        graph_encoding=GRAPH_ENCODING,
        gin_layers=gin_layers,
        pooling_type=POOLING_TYPE,
        dropout=DROPOUT
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TripletLoss(margin=1.0)

    log_information(log_path, {
        "Hyperparameters": {
            "learning_rate": lr,
            "hidden_dim": hidden_dim,
            "gin_layers": gin_layers
        }
    }, log_name="Hyperparameter Combo")

    print(f"\n🔍 Entrenando modelo: {model_id}")
    
    # Entrenamiento y validación
    train_model_with_early_stopping(
        model=model,
        model_id=model_id,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        device=DEVICE,
        log_path=log_path,
        save_best_weights=True,
        decay_rate=DECAY_RATE
    )

    # Extraer métricas del log
    val_auc = None
    val_loss = None
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "Validation AUC" in line and val_auc is None:
                try:
                    val_auc = float(line.split(":")[-1].strip())
                except:
                    val_auc = None
            if "Validation Triplet Loss" in line and val_loss is None:
                try:
                    val_loss = float(line.split(":")[-1].strip())
                except:
                    val_loss = None
            if val_auc is not None and val_loss is not None:
                break

    results.append({
        "learning_rate": lr,
        "hidden_dim": hidden_dim,
        "gin_layers": gin_layers,
        "val_triplet_loss": val_loss,
        "val_auc": val_auc
    })

# Guardar la grilla
df_results = pd.DataFrame(results)
df_results.to_csv(f"{result_folder}/grid_search_results.csv", index=False)
print(f"\n✅ Resultados guardados en {result_folder}/grid_search_results.csv")
