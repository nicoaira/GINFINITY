import argparse
import os
import torch
import numpy as np
import pandas as pd
import torch
import numpy as np
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import random
from sklearn.manifold import TSNE

def str_to_tensor(embedding_str):
    embedding_list = embedding_str.split(',')

    # Convert the list of strings to a NumPy array of floats
    embedding_array = np.array([float(x) for x in embedding_list])

    #Convert the NumPy array back to a PyTorch tensor
    return torch.tensor(embedding_array)

def project_embeddings(df):
    embeddings = [str_to_tensor(e) for e in df['embedding_vector']]

    # Ensure the embeddings are 2D (i.e., shape is [num_samples, 256])
    embeddings = torch.stack(embeddings).numpy()

    # Check if the embeddings are already flattened to (num_samples, 256)
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(1)  # Remove the singleton dimension if necessary

    # Extract RNA class labels
    rfam = df['rfam'].values

    ### 1. t-SNE ###
    # Perform t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=3, random_state=42)
    embedding_tsne = tsne.fit_transform(embeddings)

    return embedding_tsne

def save_df_with_tsne_embedding(df, embedding_tsne, output_path):
    df['tSNE_1'] = embedding_tsne[:, 0]
    df['tSNE_2'] = embedding_tsne[:, 1]
    df['tSNE_3'] = embedding_tsne[:, 2]
    df.to_csv(output_path)

def save_scatter_2d_3plots(output_folder, df, embedding_tsne, column='rfam'):
    column_values = df[column].values

    df_tsne = pd.DataFrame({
        'tSNE_1': embedding_tsne[:, 0],
        'tSNE_2': embedding_tsne[:, 1],
        'tSNE_3': embedding_tsne[:, 2],
        column: column_values
    })

    # Create the three scatter plots using Plotly Express
    fig_tsne12 = px.scatter(
        df_tsne,
        x='tSNE_1',
        y='tSNE_2',
        color=column,
        labels={'tSNE_1': 't-SNE Dimension 1', 'tSNE_2': 't-SNE Dimension 2', 'color': 'RNA Class'},
        title='t-SNE 1 vs 2',
        hover_data=[column],
      #  color_discrete_sequence=qualitative.Vivid
    )

    fig_tsne13 = px.scatter(
        df_tsne,
        x='tSNE_1',
        y='tSNE_3',
        color=column,
        labels={'tSNE_1': 't-SNE Dimension 1', 'tSNE_3': 't-SNE Dimension 3', 'color': 'RNA Class'},
        title='t-SNE 1 vs 3',
        hover_data=[column],
      #  color_discrete_sequence=qualitative.Vivid
    )

    fig_tsne23 = px.scatter(
        df_tsne,
        x='tSNE_2',
        y='tSNE_3',
        color=column,
        labels={'tSNE_2': 't-SNE Dimension 2', 'tSNE_3': 't-SNE Dimension 3', 'color': 'RNA Class'},
        title='t-SNE 2 vs 3',
        hover_data=[column],
      #  color_discrete_sequence=qualitative.Vivid
    )

    # Create a subplot layout with 1 row and 3 columns
    fig_combined = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("t-SNE 1 vs 2", "t-SNE 1 vs 3", "t-SNE 2 vs 3"),
        horizontal_spacing=0.1
    )

    # Add the traces from each individual figure to the combined figure
    for trace in fig_tsne12.data:
        fig_combined.add_trace(trace, row=1, col=1)

    for trace in fig_tsne13.data:
        trace.showlegend = False
        fig_combined.add_trace(trace, row=1, col=2)

    for trace in fig_tsne23.data:
        trace.showlegend = False
        fig_combined.add_trace(trace, row=1, col=3)

    # Update the layout of the combined figure
    fig_combined.update_layout(
        title_text='t-SNE Subplots of RNA embeddings',
        height=600,
        width=1800,
        showlegend=True  # Keeps a single shared legend
    )

    # Update x and y axis titles for each subplot
    fig_combined.update_xaxes(title_text='t-SNE Dimension 1', row=1, col=1)
    fig_combined.update_yaxes(title_text='t-SNE Dimension 2', row=1, col=1)

    fig_combined.update_xaxes(title_text='t-SNE Dimension 1', row=1, col=2)
    fig_combined.update_yaxes(title_text='t-SNE Dimension 3', row=1, col=2)

    fig_combined.update_xaxes(title_text='t-SNE Dimension 2', row=1, col=3)
    fig_combined.update_yaxes(title_text='t-SNE Dimension 3', row=1, col=3)

    # Define output file paths
    html_output_path = os.path.join(output_folder, f"scatter_tsne_subplots_{column}.html")
    png_output_path = os.path.join(output_folder, f"scatter_tsne_subplots_{column}.png")
    svg_output_path = os.path.join(output_folder, f"scatter_tsne_subplots_{column}.svg")

    # Save the combined plot as an interactive HTML file
    fig_combined.write_html(html_output_path)
    print(f"Combined scatter plot saved as HTML to {html_output_path}")

    # Save the combined plot as a PNG file
    fig_combined.write_image(png_output_path, format='png', width=1800, height=600)
    print(f"Combined scatter plot saved as PNG to {png_output_path}")

    # Save the combined plot as an SVG file
    fig_combined.write_image(svg_output_path, format='svg', width=1800, height=600)
    print(f"Combined scatter plot saved as SVG to {svg_output_path}")

def save_scatter_2d(output_folder, df, embedding_tsne, column = 'rfam'):
    column_values = df[column].values

    df_tsne = pd.DataFrame({
        'tSNE_1': embedding_tsne[:, 0],
        'tSNE_2': embedding_tsne[:, 1],
        # 'tSNE_3': embedding_tsne[:, 3],
        column: column_values
    })

    # Plot the t-SNE projection using Plotly
    fig_tsne12 = px.scatter(
        df_tsne,
        x='tSNE_1',
        y='tSNE_2',
        color=column,
        labels={'color': 'RNA Class'},
        title='t-SNE projection of RNA embeddings',
        hover_data=[column]
    )

    # Define file paths
    html_output_path = f"{output_folder}/scatter_tsne_{column}.html"
    png_output_path = f"{output_folder}/scatter_tsne_{column}.png"
    svg_output_path = f"{output_folder}/scatter_tsne_{column}.svg"

    # Save as an interactive HTML file
    fig_tsne12.write_html(html_output_path)
    print(f"Scatter plot saved as HTML to {html_output_path}")

    # Save as a static PNG image
    pio.write_image(fig_tsne12, png_output_path, format='png', width=800, height=600)
    print(f"Scatter plot saved as PNG to {png_output_path}")
    pio.write_image(fig_tsne12, svg_output_path, format='svg', width=800, height=600)
    print(f"Scatter plot saved as SVG to {svg_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--generate_embeddings_path', type=str, required=True)
    parser.add_argument('--model_id', required=True, type=str)
    parser.add_argument('--sample_num', type=int)
    args = parser.parse_args()


    df = pd.read_csv(args.generate_embeddings_path, sep="\t")

    if args.sample_num:
        random_indices = random.sample(range(len(df)), args.sample_num)
        df = df.iloc[random_indices].copy()
    
    # Filter the DataFrame to exclude the specified rna_types
    rna_types_to_remove = ['antisense_RNA', 'other', 'ribozyme', 'vault_RNA', 'vaultRNA', 'ncRNA', 'scaRNA', 'SRP_RNA', 'RNase_MRP_RNA', 'tRNA'] 
    df = df[~df['rna_type'].isin(rna_types_to_remove)]

    embedding_tsne = project_embeddings(df)

    output_folder = f"output/{args.model_id}"
    os.makedirs(output_folder, exist_ok=True)

    save_scatter_2d(output_folder, df, embedding_tsne, column = 'rfam')
    save_scatter_2d(output_folder, df, embedding_tsne, column = 'rna_type')

    save_scatter_2d_3plots(output_folder, df, embedding_tsne, column = 'rfam')
    save_scatter_2d_3plots(output_folder, df, embedding_tsne, column = 'rna_type')

    df['tSNE_1'] = embedding_tsne[:, 0]
    df['tSNE_2'] = embedding_tsne[:, 1]
    df['tSNE_3'] = embedding_tsne[:, 2]
    
    projected_embeddings_path = f"{output_folder}/projected_embeddings.csv"
    df.to_csv(projected_embeddings_path)
    print(f"Saved projections to {projected_embeddings_path}")
    