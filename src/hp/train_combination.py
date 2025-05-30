import argparse
import json
import os
import time
import pandas as pd
import torch
from torch import optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from src.benchmark.benchmark import run_benchmark
from src.model.gin_model import GINModel
from src.gin_rna_dataset import GINRNADataset
from src.triplet_loss import TripletLoss
from src.utils import get_project_root
from train_model import (
    train_model_with_early_stopping,
    remove_invalid_structures,
    log_setup,
    log_information
)
from filelock import FileLock

project_root = get_project_root()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--gin_layers", type=int, required=True)
    parser.add_argument("--input_path", type=str, required=True, default="example_data/train.csv")
    parser.add_argument("--val_path", type=str, required=True, default="example_data/val_dataset.csv")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=0.005)
    parser.add_argument("--graph_encoding", type=str, default="standard")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pooling_type", type=str, default="global_add_pool")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decay_rate", type=float, default=0.95)
    parser.add_argument("--results_csv", type=str, default="hp_results/parallel_grid_results.csv")
    parser.add_argument('--benchmark_datasets', type=str, default='rna_artificial_clusters_benchmark-evo-3', help='Benchmark datasets to use for evaluation.')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("hp_results", exist_ok=True)

    model_id = f"gin_lr{args.lr}_dim{args.hidden_dim}__out{args.output_dim}_layers{args.gin_layers}"
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

    # Load benchmark metadata
    with open(os.path.join(project_root, 'data/benchmark_datasets/benchmark_datasets.json'), 'r') as f:
        benchmark_metadata = json.load(f)

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
    triplet_loss = train_model_with_early_stopping(
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

    # Run benchmark and get average AUCs
    model_checkpoint_path = f"{output_dir}/{model_id}.pth"
    benchmark_results_path = os.path.join(output_dir, "benchmark_",model_id)
    average_aucs = run_benchmark(
        embeddings_script=os.path.join(project_root, "predict_embedding.py"),
        benchmark_datasets=[args.benchmark_datasets],  # Pass as a list
        benchmark_metadata=benchmark_metadata,
        benchmark_metadata_path=os.path.join(project_root, 'data/benchmark_datasets/benchmark_datasets.json'),
        datasets_dir=os.path.join(project_root, 'data/benchmark_datasets'),
        save_embeddings=False,
        emb_output_path=os.path.join(benchmark_results_path, "embeddings"),
        model_weights_path=model_checkpoint_path,
        structure_column_name="secondary_structure",
        structure_column_num=None,
        header=True,
        skip_barplot=False,
        skip_auc_curve=True,
        results_path=benchmark_results_path,
        save_distances=False,
        no_save=False,
        only_needed_embeddings=True,
        no_log=False,
        device=args.device,
        # num_workers=args.num_workers,
        distance_batch_size=1000,
        rna_types=[],
        # quiet=args.quiet_benchmark,
        # retries=args.retries  # Pass retries argument
    )

    elapsed = round((time.time() - start_time) / 60, 3)

    val_auc = sum(average_aucs) / len(average_aucs)
    # Guardar resultados
    results = {
        "model_id": model_id,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "gin_layers": args.gin_layers,
        "val_triplet_loss": triplet_loss,
        "val_auc": val_auc,
        "time_minutes": elapsed
    }

    lock_path = args.results_csv + '.lock'
    with FileLock(lock_path):
        df_result = pd.DataFrame([results])
        if not os.path.exists(args.results_csv):
            df_result.to_csv(args.results_csv, index=False)
        else:
            df_result.to_csv(args.results_csv, mode='a', index=False, header=False)
    print(f"✅ {model_id} terminado. AUC={val_auc}, Loss={triplet_loss}, Tiempo={elapsed} min")
    log_information(log_path, results, "Results")
if __name__ == "__main__":
    main()
