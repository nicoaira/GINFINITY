import argparse
import random
import torch
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os

from tqdm import tqdm

from src.embedding_evaluation.utils import square_dist
from src.training.gin_rna_dataset import GINRNADataset
from src.model.gin_model import GINModel
from src.utils import is_valid_dot_bracket

def remove_invalid_structures(df):
    valid_structures = (
        df["anchor_structure"].apply(is_valid_dot_bracket) & 
        df["positive_structure"].apply(is_valid_dot_bracket) & 
        df["negative_structure"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def load_trained_model(model_path, device='cpu'):
    """Load trained model from checkpoint with metadata"""
    model = GINModel.load_from_checkpoint(model_path, device)
    model.to(device)  # Move model to the specified device
    model.eval()
    return model

def get_dataset_loader(val_df):
    val_dataset = GINRNADataset(val_df)
    val_loader = GeoDataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

    return val_loader

def generate_validation_embeddings(model, validation_loader):
   
  anchor_embeddings = []
  positive_embeddings = []
  negative_embeddings = []

  with torch.no_grad():
      progress_bar_val = tqdm(enumerate(validation_loader), total=len(validation_loader))
      for _, batch in progress_bar_val:
          anchor, positive, negative = batch

          # Forward pass
          anchor_out, positive_out, negative_out = model(
              anchor, positive, negative)

          # Collect embeddings
          anchor_embeddings.append(anchor_out)
          positive_embeddings.append(positive_out)
          negative_embeddings.append(negative_out)

  # Concatenate the results into single tensors
  anchor_embeddings = torch.cat(anchor_embeddings)
  positive_embeddings = torch.cat(positive_embeddings)
  negative_embeddings = torch.cat(negative_embeddings)
  return anchor_embeddings, positive_embeddings, negative_embeddings

def save_val_embeddings(anchor_embeddings, positive_embeddings, negative_embeddings, embeddings_path):
    # Convert the tensors to NumPy arrays for saving
    anchor_embeddings_np = anchor_embeddings.detach().cpu().numpy()
    positive_embeddings_np = positive_embeddings.detach().cpu().numpy()
    negative_embeddings_np = negative_embeddings.detach().cpu().numpy()

    # Create a DataFrame for the embeddings
    embeddings_df = pd.DataFrame({
        'Type': ['Anchor'] * len(anchor_embeddings_np) + ['Positive'] * len(positive_embeddings_np) + ['Negative'] * len(negative_embeddings_np),
        'Embedding': list(anchor_embeddings_np) + list(positive_embeddings_np) + list(negative_embeddings_np)
    })

    # Expand the embeddings into separate columns
    embedding_cols = pd.DataFrame(embeddings_df['Embedding'].to_list())
    embeddings_df = embeddings_df.drop(columns=['Embedding']).join(embedding_cols)

    # Save the DataFrame to a CSV file
    embeddings_df.to_csv(embeddings_path, index=False)

def save_histogram(output_folder, anchor_embeddings, positive_embeddings, negative_embeddings, metric, square_dist_range = None):
    
    if metric == 'cosine':
        anchor_positive_similarity = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
        anchor_negative_similarity = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)
        hist_range = (0,1)
    elif metric == 'square_dist':
        anchor_positive_similarity = square_dist(anchor_embeddings, positive_embeddings)
        anchor_negative_similarity = square_dist(anchor_embeddings, negative_embeddings)
        hist_range = square_dist_range
 
    # Plot the histograms
    plt.figure(figsize=(10, 6))

    # Plot for anchor-positive distances (blue)
    plt.hist(anchor_positive_similarity.numpy(), bins=30, alpha=0.5, range=hist_range, label='Anchor-Positive', color='blue')

    # Plot for anchor-negative distances (red)
    plt.hist(anchor_negative_similarity.numpy(), bins=30, alpha=0.5, range=hist_range, label='Anchor-Negative', color='red')

    plt.ylim(0, 25000)  # Set y-axis limits


    # Add labels and title
    # Add labels and title
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Anchor-Positive and Anchor-Negative Distances ({metric.capitalize()} Metric)')
    plt.legend()

    if metric == 'cosine' or square_dist_range:
        output_path_png = os.path.join(output_folder, f"histogram_{metric}.png")
        output_path_svg = os.path.join(output_folder, f"histogram_{metric}.svg")
    else:
        output_path_png = os.path.join(output_folder, f"histogram_{metric}_original_scale.png")
        output_path_svg = os.path.join(output_folder, f"histogram_{metric}_original_scale.svg")

    plt.savefig(output_path_png)
    plt.savefig(output_path_svg)
    plt.close()  # Close the plot to free memory
    print(f"Saved {metric} histogram to {output_path_png}")
    print(f"Saved {metric} histogram to {output_path_svg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_id', default = "gin_2", type=str)
    parser.add_argument('--graph_encoding', type=str, choices=['standard', 'forgi'], default='standard',
                        help='Encoding to use for the transformation to graph. Only used in case of gin modeling')
    parser.add_argument('--val_dataset_path', default ="data/example_data/val_dataset.csv", type=str)
    parser.add_argument('--val_embeddings_path', type=str)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--save_embeddings', type=bool, default=True)
    args = parser.parse_args()

    output_folder = f"output/{args.model_id}"
    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

    if not args.val_embeddings_path:
        val_df = pd.read_csv(args.val_dataset_path)
        val_df = remove_invalid_structures(val_df)

        if args.samples:
            random_indices = random.sample(range(len(val_df)), args.samples)
            val_df = val_df.iloc[random_indices].copy()

        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
        val_loader = GeoDataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
        
        model = load_trained_model(
            args.model_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        anchor_embeddings, positive_embeddings, negative_embeddings = generate_validation_embeddings(model, val_loader)
        if args.save_embeddings:
            embeddings_path = f"{output_folder}/validation_embeddings.csv"
            save_val_embeddings(anchor_embeddings, positive_embeddings, negative_embeddings, embeddings_path)
            print(f"Saved embedding to {embeddings_path}")

    else:
        embeddings_df = pd.read_csv(args.val_embeddings_path)

        # Split the DataFrame based on the 'Type' column
        anchor_embeddings_df = embeddings_df[embeddings_df['Type'] == 'Anchor']
        positive_embeddings_df = embeddings_df[embeddings_df['Type'] == 'Positive']
        negative_embeddings_df = embeddings_df[embeddings_df['Type'] == 'Negative']

        # Drop the 'Type' column and convert back to tensors
        anchor_embeddings = torch.tensor(anchor_embeddings_df.drop(columns=['Type']).values, dtype=torch.float32)
        positive_embeddings = torch.tensor(positive_embeddings_df.drop(columns=['Type']).values, dtype=torch.float32)
        negative_embeddings = torch.tensor(negative_embeddings_df.drop(columns=['Type']).values, dtype=torch.float32)

    save_histogram(output_folder, anchor_embeddings, positive_embeddings, negative_embeddings, 'cosine')
    save_histogram(output_folder, anchor_embeddings, positive_embeddings, negative_embeddings, 'square_dist', square_dist_range=(0,1000))
    save_histogram(output_folder, anchor_embeddings, positive_embeddings, negative_embeddings, 'square_dist')

