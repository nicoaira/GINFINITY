from datetime import datetime
import platform
import sys
import GPUtil
import psutil
import torch
import networkx as nx
import os
from torch_geometric.data import Data
import pandas as pd

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

def graph_to_tensor(G):
    nodes = sorted(G.nodes())
    node_features = [[1.0] if G.nodes[node]['label'] == 'paired' else [0.0] for node in nodes]
    x = torch.tensor(node_features, dtype=torch.float)
    edge_indices = [[u, v] for u, v in G.edges()]
    edge_attrs = [[1.0, 0.0] if G.edges[u, v]['edge_type'] == 'adjacent' else [0.0, 1.0] for u, v in G.edges()]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ==============================================================================
# Input Data Setup and Validation
# ==============================================================================

def setup_and_read_input(args, need_model=False):
    # Logging setup
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + '.log'
    quiet = getattr(args, "quiet", False)
    log_setup(log_path, print_log=not quiet)
    log_information(log_path, vars(args), "Arguments", print_log=not quiet)

    # Read data
    sep_char = '\t' if args.input.endswith('.tsv') else ','
    df = pd.read_csv(args.input, sep=sep_char, low_memory=False)

    # Check that the structure column name exists in df, else raise a ValueError
    if args.structure_column_name not in df.columns:
        raise ValueError(f"Structure column '{args.structure_column_name}' not found in input data.")
    # Ensure the ID column exists
    if args.id_column not in df.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in input data.")
    # Check for duplicate IDs
    if df[args.id_column].duplicated().any():
        log_information(log_path, {"warning": "duplicate IDs"}, "Warning")

    # Model path existence check (only if needed)
    if need_model:
        if not hasattr(args, "model_path"):
            raise ValueError("need_model=True but args has no model_path attribute.")
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model path '{args.model_path}' does not exist.")

    # Determine extra cols to propagate
    if args.keep_cols:
        requested_cols = [c.strip() for c in args.keep_cols.split(',')]
        missing_cols = [col for col in requested_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns specified in --keep-cols do not exist in the input file: {missing_cols}")
        propagate = requested_cols
    else:
        propagate = [
            col for col in df.columns
            if col not in [args.id_column, args.structure_column_name]
        ]
    return df, log_path, propagate