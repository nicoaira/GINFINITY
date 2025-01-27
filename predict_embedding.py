#!/usr/bin/env python3

import random
import time
import torch
import pandas as pd
from tqdm import tqdm
import argparse
from src.model.gin_model import GINModel
from src.utils import dotbracket_to_forgi_graph, forgi_graph_to_tensor, log_information, log_setup, dotbracket_to_graph, graph_to_tensor
import os
import subprocess
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method, Manager

# Load the trained model


def load_trained_model(model_path, device='cpu'):
    """Load trained model from checkpoint with metadata"""
    model = GINModel.load_from_checkpoint(model_path, device)
    model.to(device)  # Move model to the specified device
    model.eval()
    return model

# Function to get embedding from graph


def get_gin_embedding(model, graph_encoding, structure, device):
    if graph_encoding == "standard":
        graph = dotbracket_to_graph(structure)
        tg = graph_to_tensor(graph)
    elif graph_encoding == "forgi":
        graph = dotbracket_to_forgi_graph(structure)
        tg = forgi_graph_to_tensor(graph)

    tg = tg.to(device)  # Move tensor to the specified device
    model.eval()
    with torch.no_grad():
        embedding = model.forward_once(tg)
    return ','.join(f'{x:.6f}' for x in embedding.cpu().numpy().flatten())


# Function to validate dot-bracket structure
def validate_structure(structure):
    if not isinstance(structure, str):
        raise ValueError(
            "The secondary structure must be a string containing valid characters for dot-bracket notation.")
    valid_characters = "()[]{}<>AaBbCcDd."
    if not all(char in valid_characters for char in structure):
        raise ValueError(f"Invalid characters found in the column used for secondary structure: '{structure}'. Valid characters are: {valid_characters}")

# Function to generate embeddings for a single row
def generate_embedding_for_row(args):
    idx, row, model, structure_column, device, graph_encoding = args
    
    structure = row[structure_column]
    validate_structure(structure)
    embedding = get_gin_embedding(model, graph_encoding, structure, device)
    return idx, embedding

# Main function to generate embeddings from CSV or TSV


def generate_embeddings(
        input_df,
        output_path,
        model_path,
        log_path,
        structure_column,
        device='cpu',
        num_workers=4
):
    # Load the trained model once - simplified as parameters are loaded from checkpoint
    model = load_trained_model(model_path, device)
    graph_encoding = model.metadata['graph_encoding']

    # Initialize list for storing embeddings
    embeddings = [None] * len(input_df)

    # Prepare arguments for multiprocessing
    args_list = [(idx, row, model, structure_column, device, graph_encoding) for idx, row in input_df.iterrows()]

    # Use multiprocessing to generate embeddings
    with Pool(num_workers) as pool:
        for idx, embedding in tqdm(pool.imap_unordered(generate_embedding_for_row, args_list), total=len(input_df), desc="Processing Embeddings"):
            embeddings[idx] = embedding

    # Add the embeddings to the DataFrame
    input_df['embedding_vector'] = embeddings

    # Save the output TSV
    input_df.to_csv(output_path, sep='\t', index=False)
    print(f"Embeddings saved to {output_path}")
    save_log = {
        "Embeddings saved path": output_path
    }
    log_information(log_path, save_log)

def read_input_data(input, samples, structure_column_num, header):
    delimiter = '\t' if input.endswith('.tsv') else ','

    # Load the input CSV based on whether there is a header or not
    if header:
        df = pd.read_csv(input, delimiter=delimiter)
    else:
        if structure_column_num is None:
            raise ValueError(
                "When header is False, structure_column_num must be specified.")
        df = pd.read_csv(input, delimiter=delimiter, header=None)
        
    if samples:
        random_indices = random.sample(range(len(df)), samples)
        df = df.iloc[random_indices].copy()
    return df

def get_structure_column_name(input_df, header,structure_column_name, structure_column_num):
    if header:
        if structure_column_name:
            structure_column = structure_column_name
        elif args.structure_column_num is not None and not structure_column_name:
            structure_column = input_df.columns[structure_column_num]
        else:
            # default value = secondary_structure
            structure_column = "secondary_structure"
    return structure_column

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Generate embeddings from RNA secondary structures using a trained GIN model.")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--samples', type=int)
    
    parser.add_argument('--output', type=str, help='Output path of the embedding')
    parser.add_argument('--model_id', type=str, help='If output path not defined, store in output/{model_id}/{model_id}_embedding.tsv')
    
    parser.add_argument('--structure_column_name', type=str,
                        help='Name of the column with the RNA secondary structures.')
    parser.add_argument('--structure_column_num', type=int,
                        help='Column number of the RNA secondary structures (0-indexed). If both column name and number are provided, column number will be ignored.')

    # Allows the default model_path to be dynamic
    script_directory = Path(__file__).resolve().parent
    default_model_path = script_directory / 'saved_model' / 'ResNet-Secondary.pth'
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file.')

    parser.add_argument('--header', type=str, default='True',
                        help='Specify whether the input CSV file has a header (default: True). Use "True" or "False".')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to run the model on (default: "cuda" if available, otherwise "cpu").')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes to use for multiprocessing (default: 4).')
    args = parser.parse_args()

    # Validate the header argument
    if args.header.lower() not in ['true', 'false']:
        raise ValueError(
            "Invalid value for --header. Please use 'True' or 'False'.")
    args.header = args.header.lower() == 'true'

    if args.output:
        output_path = args.output
    elif args.model_id:
        output_path = f"output/{args.model_id}/{args.model_id}_embeddings.tsv"
    else:
        raise "Either output path or output name must be defined"
    
    input_df = read_input_data(args.input, args.samples, args.structure_column_num, args.header)

    # Determine which column to use for structure
    structure_column = get_structure_column_name(input_df, args.header, args.structure_column_name, args.structure_column_num)
    
    device = args.device
    
    output_folder = os.path.dirname(output_path) 
    os.makedirs(output_folder, exist_ok=True)

    log_path = f"{output_folder}/predict_embedding.log"
    log_setup(log_path)

    predict_params = {
        "model_path": args.model_path,
        "device": device,
        "test_data_path": args.input,
        "samples_test_data": input_df.shape[0],
        "num_workers": args.num_workers
    }
    
    # Model metadata will be logged automatically when loading the model
    log_information(log_path, predict_params, "Predict params")
    
    start_time = time.time()
    # Generate embeddings
    generate_embeddings(
        input_df,
        output_path,
        args.model_path,
        log_path,
        structure_column,
        device=device,
        num_workers=args.num_workers
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    print(f"Finished. Total execution time: {execution_time_minutes:.6f} minutes")
    execution_time = {
        "Total execution time" : f"{execution_time_minutes:.6f} minutes"
    }
    log_information(log_path, execution_time, "Execution time")
