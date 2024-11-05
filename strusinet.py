#!/usr/bin/env python3

import torch
import pandas as pd
from tqdm import tqdm
import argparse
from src.model.siamese_model import SiameseResNetLSTM
from src.model.GINModel import GINModel
from src.model.utils import pad_and_convert_to_contact_matrix, dotbracket_to_graph, graph_to_tensor
import re
import os
import subprocess
from pathlib import Path

# Load the trained model
def load_trained_model(model_path, model_type = "siamese", input_channels=1, hidden_dim=256, lstm_layers=1, device='cpu'):
    # Check if the model file exists, if not provide instruction to download it
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Attempting to download...")
        model_url = "https://drive.google.com/uc?export=download&id=1ltrAQ2OfmvrRx8cKxeNKK_oebwVRClEW"
        download_command = f"wget -O {model_path} \"{model_url}\""
        try:
            subprocess.run(download_command, shell=True, check=True)
            print(f"Model downloaded successfully and saved at {model_path}")
        except subprocess.CalledProcessError:
            raise FileNotFoundError(f"Failed to download the model file. Please download it manually from {model_url} and place it in the 'saved_model/' directory.")
    
    # Instantiate the model
    if model_type == "siamese":
        model = SiameseResNetLSTM(input_channels=input_channels, hidden_dim=hidden_dim, lstm_layers=lstm_layers)
    elif model_type == "gin":
        model = GINModel(input_dim=2, hidden_dim=256, output_dim=128)

    # Load the checkpoint that contains multiple states (epoch, optimizer, and model state_dict)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Load only the model's state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to the appropriate device (CPU or GPU)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to get embedding from contact matrix
def get_siamese_embedding(contact_matrix, model, device='cpu'):
    contact_tensor = torch.tensor(contact_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, max_len, max_len)
    
    with torch.no_grad():
        embedding = model.forward_once(contact_tensor)
    return ','.join(f'{x:.6f}' for x in embedding.cpu().numpy().flatten())

# Function to get embedding from graph
def get_gin_embedding(graph, model):
    tg = graph_to_tensor(graph)
    
    model.eval()
    with torch.no_grad():
        embedding = model(tg)
    return ','.join(f'{x:.6f}' for x in embedding.cpu().numpy().flatten())



# Function to validate dot-bracket structure
def validate_structure(structure):
    if not isinstance(structure, str):
        raise ValueError("The secondary structure must be a string containing valid characters for dot-bracket notation.")
    valid_characters = "()[]{}<>AaBbCcDd."
    if not all(char in valid_characters for char in structure):
        raise ValueError(f"Invalid characters found in the column used for secondary structure: '{structure}'. Valid characters are: {valid_characters}")

# Main function to generate embeddings from CSV or TSV
def generate_embeddings(input, output, model_type, model_path, structure_column_name='secondary_structure', structure_column_num=None, max_len=641, device='cpu', header=True):
    # Load the trained model
    model = load_trained_model(model_path, model_type, device=device)
    
    # Determine delimiter based on file extension
    delimiter = '\t' if input.endswith('.tsv') else ','
    
    # Load the input CSV based on whether there is a header or not
    if header:
        df = pd.read_csv(input, delimiter=delimiter)
    else:
        if structure_column_num is None:
            raise ValueError("When header is False, structure_column_num must be specified.")
        df = pd.read_csv(input, delimiter=delimiter, header=None)
        structure_column = df.columns[structure_column_num]
    
    # Determine which column to use for structure
    if header:
        if args.structure_column_name:
            structure_column = structure_column_name
        elif structure_column_num is not None and not args.structure_column_name :
            structure_column = df.columns[structure_column_num]
        else:
            # default value = secondary_structure
            structure_column = "secondary_structure"
    
    # Initialize list for storing embeddings
    embeddings = []

    # Iterate over rows and calculate embeddings using tqdm for progress bar
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Embeddings"):
            structure = row[structure_column]
            # Validate the dot-bracket structure
            validate_structure(structure)
            if model_type == "siamese":
                # Convert dot-bracket structure to contact matrix
                contact_matrix = pad_and_convert_to_contact_matrix(structure, max_len)
                # Get the embedding using the neural network
                embedding = get_siamese_embedding(contact_matrix, model, device=device)
            elif model_type == "gin":
                graph = dotbracket_to_graph(structure)
                embedding = get_gin_embedding(graph, model)

            embeddings.append(embedding)  # Convert list to comma-separated string

    # Add the embeddings to the DataFrame
    df['embedding_vector'] = embeddings
    
    # Save the output TSV
    df.to_csv(output, sep='\t', index=False)
    print(f"Embeddings saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from RNA secondary structures using a trained Siamese model.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output TSV file with embeddings.')
    parser.add_argument('--structure_column_name', type=str, help='Name of the column with the RNA secondary structures.')
    parser.add_argument('--structure_column_num', type=int, help='Column number of the RNA secondary structures (0-indexed). If both column name and number are provided, column number will be ignored.')

    # Allows the default model_path to be dynamic
    script_directory = Path(__file__).resolve().parent
    default_model_path = script_directory / 'saved_model' / 'ResNet-Secondary.pth'
    parser.add_argument('--model_path', type=str, default=str(default_model_path), help=f'Path to the trained model file (default: {default_model_path}).')
    
    parser.add_argument('--model_type', type=str, default='siamese', help='Model type to run (e.g., "siamese" or "gin").')

    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu" or "cuda").')
    parser.add_argument('--header', type=str, default='True', help='Specify whether the input CSV file has a header (default: True). Use "True" or "False".')
    args = parser.parse_args()

    # Validate the header argument
    if args.header.lower() not in ['true', 'false']:
        raise ValueError("Invalid value for --header. Please use 'True' or 'False'.")
    args.header = args.header.lower() == 'true'

    # Generate embeddings
    generate_embeddings(
        args.input, 
        args.output, 
        args.model_type, 
        args.model_path, 
        structure_column_name=args.structure_column_name, 
        structure_column_num=args.structure_column_num, 
        device=args.device,
        header=args.header
    )
