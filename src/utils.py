from datetime import datetime
import platform
import sys
import GPUtil
import numpy as np
import psutil
import torch
import networkx as nx
import os
from torch_geometric.data import Data
import forgi.graph.bulge_graph as fgb
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd # Added import

# ==============================================================================
# Logging & System Utilities
# ==============================================================================

def get_system_info():
    # Operating System
    os_name = platform.system()
    os_version = platform.version()
    os_release = platform.release()

    # CPU Info
    cpu = platform.processor()
    physical_cores = psutil.cpu_count(logical=False)
    total_cores = psutil.cpu_count(logical=True)
    # Memory Info
    svmem = psutil.virtual_memory()
    total_memory = svmem.total / (1024 ** 3)  # Convert to GB

    # Disk Info
    total_disk_space = psutil.disk_usage('/').total / (1024 ** 3)  # Convert to GB

    # GPU Info
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "Name": gpu.name,
            "Load": f"{gpu.load * 100:.2f}%",
            "Memory Total": f"{gpu.memoryTotal}MB",
            "Memory Used": f"{gpu.memoryUsed}MB",
            "Memory Free": f"{gpu.memoryFree}MB",
            "Driver Version": gpu.driver
        })

    # Collect all info into a dictionary
    system_info = {
        "Operating System": f"{os_name} {os_release} (Version: {os_version})",
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "CPU": {
            "Name": cpu,
            "Physical Cores": physical_cores,
            "Total Cores": total_cores,
        },
        "Memory": f"{total_memory:.2f} GB",
        "Disk Space": f"{total_disk_space:.2f} GB",
        "GPU": gpu_info if gpu_info else "No GPU Detected",
    }

    return system_info

def log_setup(log_path, print_log=True):
    """
    Sets up and logs basic run information to a specified log file.

    This function records essential details about the runtime environment,
    including the date and time, the command run, platform information, 
    Python version, and GPU details (if available). The information is 
    logged in a specified file in write mode, overwriting any existing content.

    Parameters
    ----------
    log_path : str
        Path to the log file where the runtime information will be stored.
    """
    log_info = {
        "Date and Time": str(datetime.now()),
        "Command Run": " ".join(sys.argv),
    }
    log_information(log_path, log_info, "Run Info", 'w')

    system_info = get_system_info()
    log_information(log_path, system_info, "System Info", print_log = print_log)

def log_information(log_path, info_dict, log_name = None, open_type='a', print_log = False):
    """
    Logs detailed information to a specified log file.

    This function writes a log entry to a file, including a header and 
    key-value pairs from the provided dictionary. Each log entry is separated
    by a line of equal signs for clarity.

    Parameters
    ----------
    log_path : str
        Path to the log file where the information will be logged.
    log_name : str
        A header name for the log entry to provide context about the information.
    info_dict : dict
        A dictionary containing the information to be logged. The dictionary
        is automatically augmented with the `log_name` as the first entry.
    open_type : str, optional
        File open mode. Defaults to 'a' (append mode). Use 'w' to overwrite 
        the log file.
    """
    with open(log_path, open_type) as f:
        sep = "\n" + "="*50 + "\n"
        f.write(sep)
        if print_log:
            print(sep)
        if log_name:
            log_name = f"{log_name}\n"
            f.write(log_name)
            if print_log:
                print(log_name)
        for key, value in info_dict.items():
            to_log = f"{key}: {value}\n"
            f.write(to_log)
            if print_log:
                print(to_log)

def get_project_root(marker=".git"):
    """
    Finds the root directory of the project by locating the specified marker file.
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if marker in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if current_dir == parent_dir:
            raise FileNotFoundError(f"Project root marker '{marker}' not found.")
        current_dir = parent_dir

# ==============================================================================
# RNA Structure Validation & Graph Processing
# ==============================================================================

def is_valid_dot_bracket(structure):
    """
    Check if a dot-bracket RNA structure has matching open and closing parentheses.
    It does not accepts pseudoknots or other types of base pairs.
    """
    stack = []
    for char in structure:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    
    # If stack is empty, all parentheses are matched
    return len(stack) == 0

def dotbracket_to_graph(dotbracket):
    G = nx.Graph()
    bases = []

    # Add nodes and edges based on dot-bracket structure
    # TODO: Node information is redundant, should it be removed?
    for i, c in enumerate(dotbracket):
        if c == '(':
            bases.append(i)
            G.add_node(i, label='unpaired')
        elif c == ')':
            if bases:
                neighbor = bases.pop()
                G.add_edge(i, neighbor, edge_type='base_pair')
                G.nodes[i]['label'] = 'paired'
                G.nodes[neighbor]['label'] = 'paired'
            else:
                print("Mismatched parentheses in input!")
                return None
        elif c == '.':
            G.add_node(i, label='unpaired')
        else:
            print("Input is not in dot-bracket notation!")
            return None

        # Adding sequential (adjacent) edges
        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent')
    
    return G

def generate_slices(G, L, keep_paired_neighbors=True):
    slices = []
    nodes = sorted(G.nodes())
    n = len(nodes)
    for start in range(n - L + 1):
        window_nodes = list(range(start, start + L))
        sub_nodes = set(window_nodes)
        if keep_paired_neighbors:
            for node in window_nodes:
                for neighbor in G.neighbors(node):
                    if G.edges[node, neighbor].get('edge_type') == 'base_pair' and neighbor not in window_nodes:
                        sub_nodes.add(neighbor)
        H = G.subgraph(sub_nodes).copy()
        if keep_paired_neighbors:
            for node in list(H.nodes()):
                if node not in window_nodes:
                    for neighbor in list(H.neighbors(node)):
                        if H.edges[node, neighbor].get('edge_type') == 'adjacent':
                            H.remove_edge(node, neighbor)
        slices.append((start, H))
    return slices

def graph_to_tensor(G):
    nodes = sorted(G.nodes())
    node_features = [[1.0] if G.nodes[node]['label'] == 'paired' else [0.0] for node in nodes]
    x = torch.tensor(node_features, dtype=torch.float)
    edge_indices = [[u, v] for u, v in G.edges()]
    edge_attrs = [[1.0, 0.0] if G.edges[u, v]['edge_type'] == 'adjacent' else [0.0, 1.0] for u, v in G.edges()]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def dotbracket_to_forgi_graph(dotbracket):
    bulge_graph = fgb.BulgeGraph.from_dotbracket(dotbracket)

    g = nx.Graph()
    
    nodes = list(bulge_graph.edges.keys())

    for i,n in enumerate(nodes):
        g.add_node(i, structure = n[0], length = bulge_graph.element_length(n))
        for c in bulge_graph.edges[n]:
            g.add_edge(i,nodes.index(c))
    return g

def forgi_graph_to_tensor(g):
    element_type_map = {
        'f': [1, 0, 0, 0, 0, 0],
        't': [0, 1, 0, 0, 0, 0],
        's': [0, 0, 1, 0, 0, 0],
        'i': [0, 0, 0, 1, 0, 0],
        'm': [0, 0, 0, 0, 1, 0],
        'h': [0, 0, 0, 0, 0, 1]
    }
    x = torch.Tensor([element_type_map.get(g.nodes[node]["structure"]) + [g.nodes[node]["length"]] for node in g.nodes])
    edge_index = torch.LongTensor(list(g.edges())).t().contiguous()

    # Graph to Data object
    data = Data(x=x, edge_index=edge_index)

    return data

def should_skip_window_due_to_low_complexity(window_sequence, mask_threshold):
    """
    Determines if a window should be skipped based on its fraction of paired bases.
    Assumes '(' and ')' are the primary pairing characters.
    """
    if mask_threshold <= 0:  # No masking if threshold is zero or negative
        return False
    
    # Count standard paired bases. Extend this if other characters like [], {} are used for pairs.
    paired_bases = window_sequence.count('(') + window_sequence.count(')')
    
    total_bases = len(window_sequence)
    if total_bases == 0:
        return True  # Skip empty windows
    
    fraction_paired = paired_bases / total_bases
    return fraction_paired < mask_threshold

# ==============================================================================
# Data Handling / I/O Utilities
# ==============================================================================

def read_input_data(input_path, samples=None, structure_column_num=None, header=True, id_column_for_validation=None):
    """Reads input data from CSV or TSV file."""
    sep_char = '\t' if input_path.endswith('.tsv') else ','
    if header:
        df = pd.read_csv(input_path, sep=sep_char, low_memory=False)
    else:
        if structure_column_num is None and id_column_for_validation is None: # if no header, a column number is usually needed for structure or id
            pass # Allow reading without specific column if not strictly needed by caller immediately
        df = pd.read_csv(input_path, sep=sep_char, header=None)
    
    if samples:
        df = df.sample(n=samples, random_state=42)
    
    # Validate ID column if provided
    if header and id_column_for_validation and id_column_for_validation not in df.columns:
        raise ValueError(f"ID column '{id_column_for_validation}' not found in input file columns: {list(df.columns)}")
    # Duplication check for ID column should be done by the caller if critical for the specific script

    return df

def get_structure_column_name(df, header, col_name=None, col_num=None, default_name="secondary_structure"):
    """Determines the name of the structure column."""
    if header:
        if col_name:
            if col_name not in df.columns:
                raise ValueError(f"Specified structure column name '{col_name}' not found in DataFrame columns: {list(df.columns)}")
            return col_name
        elif col_num is not None:
            if col_num >= len(df.columns):
                raise ValueError(f"Specified structure column number {col_num} is out of bounds for DataFrame columns: {list(df.columns)}")
            return df.columns[col_num]
        else:
            # If default_name is not in columns, it's an issue, but predict_embedding handles this by checking existence.
            # For now, just return default_name; caller should verify.
            return default_name
    else: # No header
        if col_num is None:
            raise ValueError("When header is False, structure_column_num must be specified.")
        if col_num >= len(df.columns):
            raise ValueError(f"Specified structure column number {col_num} is out of bounds for DataFrame columns (no header): {len(df.columns)} columns exist.")
        return df.columns[col_num] # This will be the integer index itself if no header

# ==============================================================================
# Other Utilities / Domain-Specific
# ==============================================================================

def calculate_distance_batch(args):
    # Moved from sample_and_pair.py
    batch, embeddings_tensor, metric = args
    results = []
    for i, j in batch:
        if metric == 'cosine':
            distance = 1 - torch.nn.functional.cosine_similarity(embeddings_tensor[i], embeddings_tensor[j], dim=0).item()
        else:  # squared distance
            distance = torch.sum((embeddings_tensor[i] - embeddings_tensor[j]) ** 2).item()
        results.append((i, j, distance))
    return results

def calculate_distances(embeddings, metric='squared', num_workers=1, batch_size=1000):
    # Moved from sample_and_pair.py
    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    num_embeddings = embeddings_tensor.shape[0]
    
    total_pairs = num_embeddings * (num_embeddings - 1) // 2
    pairs = [(i, j) for i in range(num_embeddings) for j in range(i + 1, num_embeddings)]
    
    # Split pairs into batches
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    args_list = [(batch, embeddings_tensor, metric) for batch in batches]
    
    distances = []
    with Pool(num_workers) as pool:
        with tqdm(total=total_pairs, desc="Calculating distances") as pbar:
            for result in pool.imap_unordered(calculate_distance_batch, args_list):
                distances.extend(result)
                pbar.update(len(result))
    
    return distances