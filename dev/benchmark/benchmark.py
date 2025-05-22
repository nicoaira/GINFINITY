import sys
import os

# Add the parent directory of 'src' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import subprocess
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
from datetime import datetime
import pytz
import shutil
import json
import re
import platform
import time
from multiprocessing import Pool
from src.model.gin_model import GINModel
from src.utils import get_project_root

# Get the project root directory
project_root = get_project_root()

def check_device(verbose=False):
    """
    Check if CUDA is available. If it is, return "cuda". Otherwise, return "cpu".
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to print out the type of device found. Defaults to False.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_model = torch.cuda.get_device_name(0)
        if verbose:
            print(f"GPU is available. Model: {gpu_model}")
    else:
        device = "cpu"
        if verbose:
            print(f"No GPU found, using CPU...")

    return device

def get_time(timezone='Europe/Madrid'):
    """
    Get the current time in the specified timezone and format it as '-YYMMDD-HHMMSS'.

    Parameters
    ----------
    timezone : str, optional
        Timezone to use. Default is 'Europe/Madrid'.

    Returns
    -------
    str
        The current time as a string in '-YYMMDD-HHMMSS' format.
    """
    geo_tz = pytz.timezone(timezone)
    geo_time = datetime.now(geo_tz)
    formatted_time = geo_time.strftime('-%y%m%d-%H%M%SS')
    return formatted_time

def get_embeddings(embeddings_script,
                   sampled_rnas_path,
                   emb_output_path,
                   model_weights_path,
                   structure_column_name,
                   structure_column_num,
                   header,
                   device='cuda',
                   num_workers=4,
                   quiet=False,
                   retries=0):  # Add retries parameter
    """
    Generate embeddings by running the embeddings script as a subprocess.

    Parameters
    ----------
    embeddings_script : str
        Path to the script that generates embeddings.
    sampled_rnas_path : str
        Path to the input file containing RNA sequences and structures.
    emb_output_path : str
        Path where the embeddings output file will be saved.
    model_weights_path : str
        Path to the model weights.
    structure_column_name : str
        Name of the column containing RNA secondary structures, if provided.
    structure_column_num : int
        Column index of the RNA secondary structures if name is not provided.
    header : bool
        Whether the input file has a header.
    device : str
        Device to use for computation ('cuda' or 'cpu')
    num_workers : int
        Number of worker processes for parallel processing
    quiet : bool
        If True, suppress console output.
    retries : int
        Number of retries if the output file is not saved.

    Raises
    ------
    FileNotFoundError
        If the embeddings script is not found.
    subprocess.CalledProcessError
        If the subprocess call to generate embeddings fails.
    """
    if not os.path.isfile(embeddings_script):
        raise FileNotFoundError(f"Embeddings script '{embeddings_script}' not found.")

    command = [
        "python", embeddings_script,
        "--input", sampled_rnas_path,
        "--output", emb_output_path,
        "--model_path", model_weights_path,
        "--header", str(header),
        "--device", device,
        "--num_workers", str(num_workers),
        "--retries", str(retries)  # Pass retries parameter
    ]

    if structure_column_name:
        command.extend(["--structure_column_name", structure_column_name])
    elif structure_column_num is not None:
        command.extend(["--structure_column_num", str(structure_column_num)])
    else:
        command.extend(["--structure_column_name", "secondary_structure"])

    try:
        if quiet:
            with open(os.devnull, 'w') as devnull:
                subprocess.run(command, check=True, text=True, stdout=devnull, stderr=devnull)
        else:
            subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the embeddings script!")
        print(e)

def calculate_distance_batch(args):
    """
    Calculate distances for a batch of pairs using vectorized operations.
    
    Parameters
    ----------
    args : tuple
        Contains (batch_pairs, embeddings_tensor)
        batch_pairs is a list of (id1, id2) tuples
        embeddings_tensor is a dictionary mapping IDs to embedding tensors
    
    Returns
    -------
    list
        List of (id1, id2, distance) tuples
    """
    batch_pairs, embedding_dict = args
    results = []
    for id1, id2 in batch_pairs:
        vector1 = embedding_dict.get(id1)
        vector2 = embedding_dict.get(id2)
        if vector1 is not None and vector2 is not None:
            # Convert numpy arrays to torch tensors if they aren't already
            if not isinstance(vector1, torch.Tensor):
                vector1 = torch.tensor(vector1, dtype=torch.float32)
                vector2 = torch.tensor(vector2, dtype=torch.float32)
            distance = torch.sum((vector1 - vector2) ** 2).item()
        else:
            distance = float('nan')
        results.append((id1, id2, distance))
    return results

def count_initial_comment_lines(file_path):
    """
    Count how many initial lines in the given file start with '#' so we can skip them
    when reading the file with pandas without using comment='#'.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    int
        Number of initial lines that start with '#'.
    """
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                count += 1
            else:
                break
    return count

def load_benchmark_dataset(benchmark_path, expected_id):
    """
    Load the benchmark dataset, skipping commented lines at the top, and verify the Benchmark ID
    against the expected ID from the metadata.

    Parameters
    ----------
    benchmark_path : str
        Path to the benchmark dataset file.
    expected_id : str
        The expected benchmark ID (from the JSON metadata) to verify.

    Returns
    -------
    pd.DataFrame
        The loaded benchmark dataset.

    Raises
    ------
    ValueError
        If the benchmark ID in the file does not match the expected ID or is not found.
    """
    benchmark_id_pattern = re.compile(r'^#\s*Benchmark ID:\s*([A-Za-z0-9]+)')
    file_id = None
    skiprows = 0
    with open(benchmark_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                skiprows += 1
                match = benchmark_id_pattern.search(line)
                if match:
                    file_id = match.group(1)
            else:
                break

    if file_id is None:
        raise ValueError(f"No Benchmark ID found in {benchmark_path}.")
    if file_id != expected_id:
        raise ValueError(f"Benchmark ID mismatch in {benchmark_path}! Expected: {expected_id}, Found: {file_id}")

    benchmark_df = pd.read_csv(benchmark_path, sep='\t', skiprows=skiprows)
    return benchmark_df

def get_distances(embedding_dict,
                  benchmark_path,
                  benchmark_name,
                  benchmark_version,
                  benchmarking_results_path,
                  expected_id,
                  save_distances=False,
                  no_save=False,
                  batch_size=1000,
                  num_workers=4,
                  quiet=False,
                  id_colname='rnacentral_id'):  # Add id_colname parameter
    """
    Calculate the square distances between embeddings for all pairs in the benchmark dataset,
    and optionally save the results.

    Parameters
    ----------
    embedding_dict : dict
        RNAcentral ID -> embedding vector.
    benchmark_path : str
        Path to the benchmark dataset.
    benchmark_name : str
        Name of the benchmark.
    benchmark_version : str
        Version of the benchmark.
    benchmarking_results_path : str
        Directory for results.
    expected_id : str
        Expected benchmark ID.
    save_distances : bool
        Whether to save distances.
    no_save : bool
        If True, do not save any outputs.
    batch_size : int
        Number of pairs to process in each batch.
    num_workers : int
        Number of worker processes for parallel processing.
    id_colname : str
        Name of the column containing unique identifiers. Default: 'rnacentral_id'
    """
    benchmark_df = load_benchmark_dataset(benchmark_path, expected_id)

    # Prepare pairs for batch processing
    pairs = list(zip(benchmark_df[f'{id_colname}_1'], benchmark_df[f'{id_colname}_2']))
    
    # Split pairs into batches
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    args_list = [(batch, embedding_dict) for batch in batches]
    
    # Calculate distances using multiprocessing
    all_results = []
    with Pool(num_workers) as pool:
        if quiet:
            all_results = []
            for results in pool.imap_unordered(calculate_distance_batch, args_list):
                all_results.extend(results)
        else:
            all_results = []
            with tqdm(total=len(pairs), desc=f"Calculating distances for {benchmark_name} (v{benchmark_version})") as pbar:
                for results in pool.imap_unordered(calculate_distance_batch, args_list):
                    all_results.extend(results)
                    pbar.update(len(results))
    
    # Create a dictionary mapping (id1, id2) to distance
    distance_dict = {(id1, id2): dist for id1, id2, dist in all_results}
    
    # Add distances to the benchmark dataframe
    benchmark_df['square_distance'] = benchmark_df.apply(
        lambda row: distance_dict.get((row[f'{id_colname}_1'], row[f'{id_colname}_2']), np.nan),
        axis=1
    )

    if save_distances and (not no_save):
        benchmark_version_f = 'v'+'_'.join(str(benchmark_version).split('.'))
        benchmark_w_dist_file = benchmark_name + '_' + benchmark_version_f + '_w_dist.tsv'
        benchmark_w_dist_path = os.path.join(benchmarking_results_path, benchmark_w_dist_file)
        benchmark_df.to_csv(benchmark_w_dist_path, sep='\t', index=False)

    return benchmark_df[benchmark_df.square_distance.notna()]

def get_roc_auc(benchmark_name, benchmark_version,
                benchmark_df, target,
                benchmarking_results_path,
                skip_barplot=False, skip_auc_curve=False,
                no_save=False, quiet=False):
    """
    Calculate ROC AUC scores and optionally generate plots.

    Parameters
    ----------
    benchmark_name : str
    benchmark_version : str
    benchmark_df : pd.DataFrame
    target : str
        Target column.
    benchmarking_results_path : str
    skip_barplot : bool
    skip_auc_curve : bool
    no_save : bool
    quiet : bool
        If True, suppress console output.

    Returns
    -------
    dict
        A dictionary with AUC results and average AUC.
    """
    unique_rna_types = benchmark_df['analysis_rna_type'].unique()
    auc_results = {}

    for rna_type in unique_rna_types:
        rna_type_df = benchmark_df[benchmark_df['analysis_rna_type'] == rna_type]
        y_true = rna_type_df[target]
        y_scores = -rna_type_df['square_distance']

        if len(y_true.unique()) > 1:
            auc_val = roc_auc_score(y_true, y_scores)
        else:
            auc_val = float('nan')
        auc_results[rna_type] = auc_val

    average_auc = np.nanmean(list(auc_results.values()))

    if not quiet:
        print("Benchmark: " + benchmark_name)
        for rna_type, auc_val in auc_results.items():
            print(f"RNA Type: {rna_type}, AUC: {auc_val:.4f}")
        print(f"Average AUC across all RNA types: {average_auc:.4f}")

    if no_save:
        return {"auc_results": auc_results, "average_auc": average_auc}

    plot_dir = os.path.join(benchmarking_results_path, "plots")
    if not os.path.exists(plot_dir) and (not skip_barplot or not skip_auc_curve):
        os.makedirs(plot_dir)

    benchmark_version_f = 'v'+'_'.join(str(benchmark_version).split('.'))

    if not skip_barplot:
        plt.figure(figsize=(10, 6))
        plt.bar(auc_results.keys(), auc_results.values())
        plt.xlabel('RNA Type')
        plt.ylabel('AUC')
        plt.title('AUC by RNA Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        benchmark_plot_filename_png = 'AUC_benchmark_' + benchmark_name + '-' + benchmark_version_f + '.png'
        benchmark_plot_filename_svg = 'AUC_benchmark_' + benchmark_name + '-' + benchmark_version_f + '.svg'
        plt.savefig(os.path.join(plot_dir, benchmark_plot_filename_png), dpi=300)
        plt.savefig(os.path.join(plot_dir, benchmark_plot_filename_svg), dpi=300)
        plt.close()

    if not skip_auc_curve:
        n_types = len(unique_rna_types)
        n_cols = 3
        n_rows = int(np.ceil(n_types / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_types == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, rna_type in enumerate(unique_rna_types):
            ax = axes[idx]
            rna_type_df = benchmark_df[benchmark_df['analysis_rna_type'] == rna_type]
            y_true = rna_type_df[target]
            y_scores = -rna_type_df['square_distance']

            if len(y_true.unique()) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_val = roc_auc_score(y_true, y_scores)

                ax.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_title(f'RNA Type: {rna_type}')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc="lower right")
            else:
                ax.set_title(f'RNA Type: {rna_type}')
                ax.text(0.5, 0.5, 'Only one class present', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

        for idx in range(n_types, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        benchmark_plot_filename_png = 'ROC_curves_benchmark_' + benchmark_name + '-' + benchmark_version_f + '.png'
        benchmark_plot_filename_svg = 'ROC_curves_benchmark_' + benchmark_name + '-' + benchmark_version_f + '.svg'

        plt.savefig(os.path.join(plot_dir, benchmark_plot_filename_png), dpi=300)
        plt.savefig(os.path.join(plot_dir, benchmark_plot_filename_svg), dpi=300)
        plt.close()

    return {"auc_results": auc_results, "average_auc": average_auc}

def cleanup(files=[], directories=[], quiet=False):
    """
    Remove specified temporary files and directories.

    Parameters
    ----------
    files : list of str, optional
        List of file paths to remove.
    directories : list of str, optional
        List of directory paths to remove.
    """
    if len(files) == 0 and len(directories) == 0:
        return
    else:
        if not quiet:
            print('Removing temporary directories and files...')
        for d in directories:
            if os.path.exists(d):
                shutil.rmtree(d)
        for f in files:
            if os.path.exists(f):
                os.remove(f)

def parse_benchmarks(benchmark_args, benchmark_metadata):
    """
    Parse the benchmark dataset arguments and select appropriate entries.
    If only a name is provided, select latest version.
    If name-vX is provided, select that version.

    Parameters
    ----------
    benchmark_args : list of str
    benchmark_metadata : dict

    Returns
    -------
    list of dict
    """
    name_dict = {}
    for entry in benchmark_metadata["benchmark_datasets"]:
        name = entry["name"]
        version = entry["version"]
        if name not in name_dict:
            name_dict[name] = []
        name_dict[name].append(entry)

    for k in name_dict:
        name_dict[k].sort(key=lambda x: x['version'], reverse=False)

    selected_benchmarks = []

    for arg in benchmark_args:
        if '-v' in arg:
            match = re.match(r"^(.*)-v(\d+)$", arg)
            if not match:
                raise ValueError(f"Invalid benchmark argument format: {arg}")
            b_name = match.group(1)
            b_version = int(match.group(2))

            if b_name not in name_dict:
                raise ValueError(f"No benchmark named '{b_name}' found in metadata.")

            found = False
            for entry in name_dict[b_name]:
                if entry["version"] == b_version:
                    selected_benchmarks.append(entry)
                    found = True
                    break
            if not found:
                raise ValueError(f"No version '{b_version}' found for benchmark '{b_name}'.")
        else:
            if arg not in name_dict:
                raise ValueError(f"No benchmark named '{arg}' found in metadata.")
            latest_entry = name_dict[arg][-1]
            selected_benchmarks.append(latest_entry)

    return selected_benchmarks

def check_required_files(embeddings_script,
                         model_weights_path,
                         benchmark_metadata_path,
                         datasets_dir,
                         selected_benchmarks,
                         benchmark_metadata):
    """
    Check all required files exist.

    Parameters
    ----------
    embeddings_script, model_weights_path, benchmark_metadata_path : str
    datasets_dir : str
    selected_benchmarks : list of dict
    benchmark_metadata : dict

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if not os.path.isfile(embeddings_script):
        raise FileNotFoundError(f"Embeddings script not found: {embeddings_script}")

    if not os.path.isfile(model_weights_path):
        raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")

    if not os.path.isfile(benchmark_metadata_path):
        raise FileNotFoundError(f"Benchmark metadata JSON not found: {benchmark_metadata_path}")

    needed_primary_ids = set([bm["primary_sampled_dataset_id"] for bm in selected_benchmarks])
    primary_map = {p["unique_id"]: p for p in benchmark_metadata["primary_sampled_datasets"]}

    for pid in needed_primary_ids:
        if pid not in primary_map:
            raise ValueError(f"No primary sampled dataset found for ID: {pid}")
        primary_filename = primary_map[pid]["filename"]
        primary_path = os.path.join(datasets_dir, primary_filename)
        if not os.path.isfile(primary_path):
            raise FileNotFoundError(f"Primary sampled dataset not found: {primary_path}")

    for bm in selected_benchmarks:
        bm_filename = bm["filename"]
        bm_path = os.path.join(datasets_dir, bm_filename)
        if not os.path.isfile(bm_path):
            raise FileNotFoundError(f"Benchmark dataset not found: {bm_path}")

def log_information(log_path, info_dict):
    """
    Log information to a specified log file.

    Parameters
    ----------
    log_path : str
        Path to the log file.
    info_dict : dict
        Dictionary with information to log.
    """
    with open(log_path, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        for key, value in info_dict.items():
            f.write(f"{key}: {value}\n")

def filter_rna_types(data, rna_types):
    data = data.copy()
    rna_types = [rt.lower() for rt in rna_types]
    def pick_analysis_type(row):
        t1 = row['rna_type_1'].lower()
        t2 = row['rna_type_2'].lower()
        if t1 in rna_types:
            return row['rna_type_1']
        elif t2 in rna_types:
            return row['rna_type_2']
        return None

    data['analysis_rna_type'] = data.apply(pick_analysis_type, axis=1)
    # Keep only rows with a known analysis_rna_type
    return data[data['analysis_rna_type'].notna()]

def run_benchmark(embeddings_script,
                  benchmark_datasets,
                  benchmark_metadata,
                  benchmark_metadata_path,
                  datasets_dir,
                  save_embeddings,
                  emb_output_path,
                  model_weights_path,
                  structure_column_name,
                  structure_column_num,
                  header,
                  skip_barplot,
                  skip_auc_curve,
                  results_path,
                  save_distances,
                  no_save,
                  only_needed_embeddings,
                  no_log,
                  device='cuda',
                  num_workers=4,
                  distance_batch_size=1000,
                  quiet=False,
                  retries=0,
                  rna_types=None,
                  id_colname='rnacentral_id'):  # Add id_colname parameter
    
    # Load model to get metadata
    model = GINModel.load_from_checkpoint(model_weights_path, device)
    model_metadata = model.metadata

    start_time = time.time()
    bm_start_time = get_time(timezone='Europe/Madrid')

    selected_benchmarks = parse_benchmarks(benchmark_datasets, benchmark_metadata)

    primary_map = {p["unique_id"]: p for p in benchmark_metadata["primary_sampled_datasets"]}
    needed_primary_ids = set([bm["primary_sampled_dataset_id"] for bm in selected_benchmarks])

    check_required_files(
        embeddings_script=embeddings_script,
        model_weights_path=model_weights_path,
        benchmark_metadata_path=benchmark_metadata_path,
        datasets_dir=datasets_dir,
        selected_benchmarks=selected_benchmarks,
        benchmark_metadata=benchmark_metadata
    )

    # Collect needed IDs if only_needed_embeddings
    primary_datasets_needed_ids = {pid: set() for pid in needed_primary_ids}
    if only_needed_embeddings:
        for bm in selected_benchmarks:
            benchmark_filename = bm["filename"]
            benchmark_id = bm["id"]
            benchmark_path = os.path.join(datasets_dir, benchmark_filename)
            bm_df = load_benchmark_dataset(benchmark_path, benchmark_id)
            pid = bm["primary_sampled_dataset_id"]
            primary_datasets_needed_ids[pid].update(bm_df[f'{id_colname}_1'].unique())
            primary_datasets_needed_ids[pid].update(bm_df[f'{id_colname}_2'].unique())

    if not no_save:
        benchmarking_results_path = results_path
        if not os.path.exists(benchmarking_results_path):
            os.makedirs(benchmarking_results_path)
    else:
        benchmarking_results_path = results_path

    # Set up logging
    log_path = None
    if not no_log and not no_save:
        log_path = os.path.join(benchmarking_results_path, 'benchmark.log')

    # Log basic info
    if log_path:
        log_info = {
            "Date and Time": str(datetime.now()),
            "Command Run": " ".join(sys.argv),
            "Platform": platform.platform(),
            "Python Version": platform.python_version(),
            "CUDA Available": str(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            log_info["GPU"] = torch.cuda.get_device_name(0)
        else:
            log_info["GPU"] = "No GPU"
        log_information(log_path, log_info)

    primary_embeddings_map = {}

    # Embedding generation timing
    embedding_start = time.time()

    for pid in needed_primary_ids:
        primary_info = primary_map[pid]
        primary_filename = primary_info["filename"]
        primary_path = os.path.join(datasets_dir, primary_filename)
        skiprows = count_initial_comment_lines(primary_path)

        if not no_save:
            if save_embeddings:
                emb_output_dir = os.path.join(benchmarking_results_path, "embeddings_" + pid)
                if not os.path.exists(emb_output_dir):
                    os.makedirs(emb_output_dir)
            else:
                emb_output_dir = "tmp"
                if not os.path.exists(emb_output_dir):
                    os.makedirs(emb_output_dir)
        else:
            emb_output_dir = "tmp"
            if not os.path.exists(emb_output_dir):
                os.makedirs(emb_output_dir)

        if only_needed_embeddings:
            needed_ids = primary_datasets_needed_ids[pid]
            primary_df = pd.read_csv(primary_path, sep='\t', header=0 if header else None, skiprows=skiprows)
            if id_colname not in primary_df.columns:
                raise ValueError(f"Primary dataset does not have '{id_colname}' column.")
            filtered_df = primary_df[primary_df[id_colname].isin(needed_ids)]
            temp_filtered_path = os.path.join(emb_output_dir, primary_info["name"] + '_filtered' + bm_start_time + '.tsv')
            filtered_df.to_csv(temp_filtered_path, sep='\t', index=False)
            embedding_input_path = temp_filtered_path
            expected_ids = set(filtered_df[id_colname].unique())
            n_rows = filtered_df.shape[0]
        else:
            full_df = pd.read_csv(primary_path, sep='\t', header=0 if header else None, skiprows=skiprows)
            if id_colname not in full_df.columns:
                raise ValueError(f"Primary dataset does not have '{id_colname}' column.")
            embedding_input_path = primary_path
            expected_ids = set(full_df[id_colname].unique())
            n_rows = full_df.shape[0]

        emb_filename = '/' + primary_info["name"] + '_w_emb' + bm_start_time + '.tsv'
        curr_emb_output_path = emb_output_dir + emb_filename

        emb_gen_start = time.time()
        get_embeddings(
            embeddings_script=embeddings_script,
            sampled_rnas_path=embedding_input_path,
            emb_output_path=curr_emb_output_path,
            model_weights_path=model_weights_path,
            structure_column_name=structure_column_name,
            structure_column_num=structure_column_num,
            header=header,
            device=device,
            num_workers=num_workers,
            quiet=quiet,
            retries=retries  # Pass retries parameter
        )
        emb_gen_end = time.time()

        embeddings_df = pd.read_csv(curr_emb_output_path, sep='\t')
        embeddings_df['embedding_vector'] = embeddings_df['embedding_vector'].apply(
            lambda x: np.array(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan
        )
        embedding_dict = dict(zip(embeddings_df[id_colname], embeddings_df['embedding_vector']))

        if save_embeddings:
            for rid in expected_ids:
                if rid not in embedding_dict:
                    embedding_dict[rid] = np.nan

        primary_embeddings_map[pid] = embedding_dict

        if log_path:
            emb_time = emb_gen_end - emb_gen_start
            emb_log_info = {
                f"Embedding Generation for PID {pid}": f"{emb_time:.4f} seconds total, {emb_time/n_rows if n_rows>0 else 'N/A'} per row"
            }
            log_information(log_path, emb_log_info)

        if not save_embeddings:
            cleanup(directories=[emb_output_dir], quiet=quiet)

    embedding_end = time.time()

    average_aucs = []

    # Benchmarking (distances and AUC)
    for bm in selected_benchmarks:
        benchmark_name = bm["name"]
        benchmark_filename = bm["filename"]
        benchmark_path = os.path.join(datasets_dir, benchmark_filename)
        benchmark_target = bm["target"]
        benchmark_version = bm["version"]
        benchmark_id = bm["id"]

        pid = bm["primary_sampled_dataset_id"]
        embedding_dict = primary_embeddings_map[pid]

        # Distance calculation timing
        dist_start = time.time()
        benchmark_df = get_distances(
            embedding_dict=embedding_dict,
            benchmark_path=benchmark_path,
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            benchmarking_results_path=benchmarking_results_path,
            expected_id=benchmark_id,
            save_distances=save_distances,
            no_save=no_save,
            num_workers=num_workers,
            batch_size=distance_batch_size,
            quiet=quiet,
            id_colname=id_colname  # Pass id_colname parameter
        )
        dist_end = time.time()

        if rna_types:
            benchmark_df = filter_rna_types(benchmark_df, rna_types)

        n_pairs = benchmark_df.shape[0]

        # AUC calculation timing
        auc_start = time.time()
        auc_info = get_roc_auc(
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            benchmark_df=benchmark_df,
            target=benchmark_target,
            benchmarking_results_path=benchmarking_results_path,
            skip_barplot=skip_barplot,
            skip_auc_curve=skip_auc_curve,
            no_save=no_save,
            quiet=quiet
        )
        auc_end = time.time()

        average_aucs.append(auc_info['average_auc'])

        if log_path:
            dist_time = dist_end - dist_start
            auc_log_info = {
                f"Distance Calculation for {benchmark_name}": f"{dist_time:.4f} seconds total, {dist_time/n_pairs if n_pairs>0 else 'N/A'} per pair"
            }
            if auc_info is not None:
                auc_log_info[f"AUC Results for {benchmark_name}"] = f"AUC by RNA Type: {auc_info['auc_results']}, Average AUC: {auc_info['average_auc']:.4f}"

            log_information(log_path, auc_log_info)

    end_time = time.time()

    if log_path:
        # General info about datasets
        dataset_info = {
            "Primary Datasets Used": str([primary_map[pid]["filename"] for pid in needed_primary_ids]),
            "Benchmark Datasets Used": str([bm["filename"] for bm in selected_benchmarks])
        }

        total_time = end_time - start_time
        dataset_info["Total Execution Time"] = f"{total_time:.4f} seconds"
        log_information(log_path, dataset_info)

    return average_aucs

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from RNA secondary structures using a trained model and benchmark them.")

    parser.add_argument('--embeddings-script', dest='embeddings_script', type=str,
                        default=os.path.join(project_root, "scripts/predict_embedding.py"),
                        help='Path to the embeddings generation script. Default: "scripts/predict_embedding.py".')

    parser.add_argument('--benchmark-metadata', dest='benchmark_metadata_path', type=str,
                        default=os.path.join(project_root, 'data/benchmark_datasets/benchmark_datasets.json'),
                        help="Name of the JSON file containing benchmark dataset information (in --datasets-dir). Default: 'benchmarking_datasets.json'")

    parser.add_argument('--datasets-dir', dest='datasets_dir', type=str,
                        default=os.path.join(project_root, 'data/benchmark_datasets'),
                        help="Directory containing the benchmark metadata JSON, primary sampled datasets, and benchmark datasets. Default: './datasets'")

    parser.add_argument('--benchmark-datasets', dest='benchmark_datasets',
                        nargs='+',
                        help="Specify one or more benchmark datasets by name or name-version. If not provided, uses all latest versions.")

    parser.add_argument('--save-embeddings', dest='save_embeddings',
                        action='store_true',
                        help="If set, embeddings will be saved. Ignored if --no-save is given.")

    parser.add_argument('--emb-output-path', dest='emb_output_path', type=str,
                        default=os.path.join(project_root, 'output/benchmarking_results/embeddings'),
                        help='Output path for embeddings if save_embeddings is set.')

    parser.add_argument('--structure-column-name', dest='structure_column_name', type=str,
                        help='Name of the column with RNA secondary structures. Default: "secondary_structure"')

    parser.add_argument('--structure-column-num', dest='structure_column_num', type=int,
                        help='Column number of the RNA secondary structures (0-indexed). If both name and num provided, name takes precedence.')

    parser.add_argument('--model-path', dest='model_path', type=str, 
                        default=os.path.join(project_root, 'output/saved_model/ResNet-Secondary.pth'),
                        help='Path to the trained model file. Default: "saved_model/ResNet-Secondary.pth".')

    parser.add_argument('--header', type=str, default='True',
                        help='Specify whether input CSV files have a header (True/False). Default: True.')

    parser.add_argument('--skip-barplot', action='store_true', dest='skip_barplot', 
                        help='Skip generating the AUC barplot.')

    parser.add_argument('--skip-auc-curve', action='store_true', dest='skip_auc_curve', 
                        help='Skip generating the ROC curve.')

    parser.add_argument('--results-path', dest='results_path', type=str,
                        default=os.path.join(project_root, 'output/benchmarking_results'),
                        help='Path to save results. Time-stamp appended unless --no-save is specified. Default: "./benchmarking_results".')

    parser.add_argument('--save-distances', action='store_true', dest='save_distances',
                        help='Save the benchmark dataframes with the distance column.')

    parser.add_argument('--no-save', action='store_true', dest='no_save',
                        help='Do not save any output files.')

    parser.add_argument('--only-needed-embeddings', dest='only_needed_embeddings', action='store_true',
                        help='If set, only generate embeddings for RNAcentral IDs required by the benchmarks.')

    parser.add_argument('--no-log', dest='no_log', action='store_true',
                        help='If set, no log file will be created.')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for parallel processing. Default: 4')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation (cuda or cpu).')

    parser.add_argument('--distance-batch-size', dest='distance_batch_size', 
                       type=int, default=1000,
                       help='Batch size for distance calculations. Default: 1000')

    parser.add_argument('--quiet', action='store_true', help='Suppress console output.')

    parser.add_argument('--retries', type=int, default=0, help='Number of retries if the output file is not saved (default: 0).')
    parser.add_argument('--rna_type', nargs='+', help='List of RNA types to include in the benchmark. Attention: the results of easy benchmarks could differe if rna_types are filter out, since the negative pairs with non selected rna_types could have different dificulty.')
    parser.add_argument('--id-column', dest='id_colname', type=str,
                        default='rnacentral_id',
                        help='Name of the column containing unique identifiers. Default: "rnacentral_id"')
    args = parser.parse_args()

    if args.header.lower() not in ['true', 'false']:
        raise ValueError("Invalid value for --header. Use 'True' or 'False'.")
    args.header = (args.header.lower() == 'true')

    benchmark_metadata_fullpath = os.path.join(args.datasets_dir, args.benchmark_metadata_path)

    with open(benchmark_metadata_fullpath, 'r') as file:
        benchmark_metadata = json.load(file)

    if not args.benchmark_datasets:
        all_names = {}
        for entry in benchmark_metadata["benchmark_datasets"]:
            name = entry["name"]
            version = entry["version"]
            if name not in all_names:
                all_names[name] = []
            all_names[name].append(version)
        benchmark_datasets = []
        for name, versions in all_names.items():
            max_version = max(versions)
            benchmark_datasets.append(f"{name}-v{max_version}")
    else:
        benchmark_datasets = args.benchmark_datasets

    run_benchmark(
        embeddings_script=args.embeddings_script,
        benchmark_datasets=benchmark_datasets,
        benchmark_metadata=benchmark_metadata,
        benchmark_metadata_path=benchmark_metadata_fullpath,
        datasets_dir=args.datasets_dir,
        save_embeddings=args.save_embeddings,
        emb_output_path=args.emb_output_path,
        model_weights_path=args.model_path,
        structure_column_name=args.structure_column_name,
        structure_column_num=args.structure_column_num,
        header=args.header,
        skip_barplot=args.skip_barplot,
        skip_auc_curve=args.skip_auc_curve,
        results_path=args.results_path,
        save_distances=args.save_distances,
        no_save=args.no_save,
        only_needed_embeddings=args.only_needed_embeddings,
        no_log=args.no_log,
        device=args.device,
        num_workers=args.num_workers,
        distance_batch_size=args.distance_batch_size,
        quiet=args.quiet,
        retries=args.retries,
        rna_types=args.rna_type if args.rna_type else None,
        id_colname=args.id_colname
    )

if __name__ == "__main__":
    main()
