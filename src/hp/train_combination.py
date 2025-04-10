import argparse
import os
import time
import pandas as pd
import torch
from torch import optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from src.model.gin_model import GINModel
from src.gin_rna_dataset import GINRNADataset
from src.triplet_loss import TripletLoss
from train_model_v2 import (
    train_model_with_early_stopping,
    remove_invalid_structures,
    log_setup,
    log_information
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--gin_layers", type=int, required=True)
    parser.add_argument("--input_path", type=str, required=True, default="example_data/train.csv")
    parser.add_argument("--val_path", type=str, required=True, default="example_data/val_dataset.csv")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=0.005)
    parser.add_argument("--graph_encoding", type=str, default="standard")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pooling_type", type=str, default="global_add_pool")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decay_rate", type=float, default=0.01)
    parser.add_argument("--results_csv", type=str, default="hp_results/parallel_grid_results.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("hp_results", exist_ok=True)

    model_id = f"gin_lr{args.lr}_dim{args.hidden_dim}_layers{args.gin_layers}"
    output_dir = f"output/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = f"{output_dir}/train.log"
    log_setup(log_path)

    log_information(log_path, vars(args), log_name="Combo Parameters")

    # Cargar datos
    train_df = pd.read_csv(args.input_path)
    val_df = pd.read_csv(args.val_path)
    train_df = remove_invalid_structures(train_df)
    val_df = remove_invalid_structures(val_df)

    train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
    val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)

    train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = GINModel(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        graph_encoding=args.graph_encoding,
        gin_layers=args.gin_layers,
        pooling_type=args.pooling_type,
        dropout=args.dropout
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = TripletLoss(margin=1.0)

    start_time = time.time()

    # Entrenamiento
    train_model_with_early_stopping(
        model=model,
        model_id=model_id,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        device=args.device,
        log_path=log_path,
        save_best_weights=True,
        decay_rate=args.decay_rate
    )

    # Extraer métricas desde el log
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

    elapsed = round((time.time() - start_time) / 60, 3)

    # Guardar resultados
    results = {
        "model_id": model_id,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "gin_layers": args.gin_layers,
        "val_triplet_loss": val_loss,
        "val_auc": val_auc,
        "time_minutes": elapsed
    }

    df_result = pd.DataFrame([results])
    if not os.path.exists(args.results_csv):
        df_result.to_csv(args.results_csv, index=False)
    else:
        df_result.to_csv(args.results_csv, mode='a', index=False, header=False)

    print(f"✅ {model_id} terminado. AUC={val_auc}, Loss={val_loss}, Tiempo={elapsed} min")

if __name__ == "__main__":
    main()
