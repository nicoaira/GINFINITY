from collections import defaultdict
from datetime import datetime
import platform
import sys
from typing import Dict, Iterable, List, Optional, Tuple

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
    """Validate that an extended dot-bracket string is well-formed.

    Supports classical pairs ``()`` and common pseudoknot annotations using
    paired symbols (``[]``, ``{}``, ``<>``) as well as matching upper/lowercase
    letter pairs such as ``A``/``a``.
    """
    bracket_pairs = {')': '(', ']': '[', '}': '{', '>': '<'}
    stacks = defaultdict(list)

    for pos, char in enumerate(structure):
        if char == '.':
            continue
        if char in bracket_pairs.values():
            stacks[char].append(pos)
            continue
        if char in bracket_pairs:
            opener = bracket_pairs[char]
            if not stacks[opener]:
                return False
            stacks[opener].pop()
            continue
        if 'A' <= char <= 'Z':
            stacks[char].append(pos)
            continue
        if 'a' <= char <= 'z':
            opener = char.upper()
            if not stacks[opener]:
                return False
            stacks[opener].pop()
            continue
        return False

    return all(len(stack) == 0 for stack in stacks.values())


def _extract_loop_segments(dotbracket: str) -> Tuple[Dict[int, Dict[str, float]], List[List[int]]]:
    """Return metadata for unpaired loop positions and contiguous segments."""

    seq_len = len(dotbracket)
    loop_meta: Dict[int, Dict[str, float]] = {}
    segments: List[List[int]] = []
    current_loop: List[int] = []

    for idx, char in enumerate(dotbracket):
        if char == '.':
            current_loop.append(idx)
            continue

        if current_loop:
            segments.append(current_loop.copy())
            loop_size = len(current_loop)
            norm_denom = max(1, seq_len)
            loop_size_norm = loop_size / norm_denom
            for pos_in_loop, node_idx in enumerate(current_loop):
                rel_pos = pos_in_loop / (loop_size - 1) if loop_size > 1 else 0.5
                loop_meta[node_idx] = {
                    'loop_size': loop_size,
                    'loop_pos': pos_in_loop,
                    'loop_size_norm': loop_size_norm,
                    'loop_pos_norm': rel_pos,
                }
            current_loop = []

    if current_loop:
        segments.append(current_loop.copy())
        loop_size = len(current_loop)
        norm_denom = max(1, seq_len)
        loop_size_norm = loop_size / norm_denom
        for pos_in_loop, node_idx in enumerate(current_loop):
            rel_pos = pos_in_loop / (loop_size - 1) if loop_size > 1 else 0.5
            loop_meta[node_idx] = {
                'loop_size': loop_size,
                'loop_pos': pos_in_loop,
                'loop_size_norm': loop_size_norm,
                'loop_pos_norm': rel_pos,
            }

    return loop_meta, segments


def _build_pair_map(dotbracket: str) -> Tuple[List[int], Dict[int, int]]:
    """Compute partner indices and branch counts for each paired position."""

    n = len(dotbracket)
    pair_map: List[int] = [-1] * n
    pair_stacks: Dict[str, List[int]] = defaultdict(list)
    bracket_pairs = {')': '(', ']': '[', '}': '{', '>': '<'}

    for idx, char in enumerate(dotbracket):
        if char == '.':
            continue
        if char in bracket_pairs.values():
            pair_stacks[char].append(idx)
            continue
        if char in bracket_pairs:
            opener = bracket_pairs[char]
            if not pair_stacks[opener]:
                return pair_map, {}
            start_idx = pair_stacks[opener].pop()
        elif 'A' <= char <= 'Z':
            pair_stacks[char].append(idx)
            continue
        elif 'a' <= char <= 'z':
            opener = char.upper()
            if not pair_stacks[opener]:
                return pair_map, {}
            start_idx = pair_stacks[opener].pop()
        else:
            return pair_map, {}

        pair_map[idx] = start_idx
        pair_map[start_idx] = idx

    branch_count: Dict[int, int] = {}
    for start, end in enumerate(pair_map):
        if end <= start:
            continue
        branches = 0
        pos = start + 1
        while pos < end:
            partner = pair_map[pos]
            if partner > pos:
                branches += 1
                pos = partner + 1
            else:
                pos += 1
        branch_count[start] = branches

    return pair_map, branch_count


def dotbracket_to_graph(dotbracket, sequence=None):
    """Convert an extended dot-bracket string (and optional sequence) into a graph.

    Parameters
    ----------
    dotbracket : str
        Dot-bracket representation of the RNA secondary structure. Supports
        classical ``()`` pairs as well as pseudoknot annotations denoted by
        ``[]``, ``{}``, ``<>`` and matching upper-/lowercase letter pairs
        such as ``A``/``a``.
    sequence : str, optional
        Nucleotide sequence corresponding to ``dotbracket``. If provided,
        each node in the resulting graph will contain a ``base`` attribute
        with the nucleotide character at that position.
    """
    G = nx.Graph()
    pair_stacks = defaultdict(list)
    bracket_pairs = {')': '(', ']': '[', '}': '{', '>': '<'}

    loop_meta, _ = _extract_loop_segments(dotbracket)

    # Add nodes and edges based on dot-bracket structure
    for i, c in enumerate(dotbracket):
        base = sequence[i] if sequence is not None and i < len(sequence) else None
        loop_info = loop_meta.get(i)
        node_attrs = {
            'label': 'unpaired',
            'base': base,
            'loop_size': loop_info['loop_size'] if loop_info else 0,
            'loop_pos': loop_info['loop_pos'] if loop_info else 0,
            'loop_size_norm': loop_info['loop_size_norm'] if loop_info else 0.0,
            'loop_pos_norm': loop_info['loop_pos_norm'] if loop_info else 0.0,
        }
        G.add_node(i, **node_attrs)

        if c == '.':
            pass
        elif c in bracket_pairs.values():
            pair_stacks[c].append(i)
        elif c in bracket_pairs:
            opener = bracket_pairs[c]
            if not pair_stacks[opener]:
                print("Mismatched base-pair symbols in input!")
                return None
            neighbor = pair_stacks[opener].pop()
            G.add_edge(i, neighbor, edge_type='base_pair')
            G.nodes[i]['label'] = 'paired'
            G.nodes[neighbor]['label'] = 'paired'
            G.nodes[i]['base'] = base
        elif 'A' <= c <= 'Z':
            pair_stacks[c].append(i)
        elif 'a' <= c <= 'z':
            opener = c.upper()
            if not pair_stacks[opener]:
                print("Mismatched pseudoknot symbols in input!")
                return None
            neighbor = pair_stacks[opener].pop()
            G.add_edge(i, neighbor, edge_type='base_pair')
            G.nodes[i]['label'] = 'paired'
            G.nodes[neighbor]['label'] = 'paired'
            G.nodes[i]['base'] = base
        else:
            print("Input is not in dot-bracket notation!")
            return None

        # Adding sequential (adjacent) edges
        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent')

    return G

def _one_hot_base(base):
    mapping = {
        'A': [1.0, 0.0, 0.0, 0.0],
        'C': [0.0, 1.0, 0.0, 0.0],
        'G': [0.0, 0.0, 1.0, 0.0],
        'U': [0.0, 0.0, 0.0, 1.0],
    }
    if base is None:
        return [0.0, 0.0, 0.0, 0.0]
    return mapping.get(base.upper(), [0.0, 0.0, 0.0, 0.0])

def graph_to_tensor(G, seq_weight: float = 0.0):
    """Convert a NetworkX graph to torch_geometric Data with weighted features.

    Parameters
    ----------
    G : nx.Graph
        Graph produced by :func:`dotbracket_to_graph`.
    seq_weight : float, optional
        Weight of the one-hot encoded nucleotide sequence relative to the
        pairing state. Must be between ``0`` and ``1``. If ``0`` (default),
        only the pairing information is used.
    """

    nodes = sorted(G.nodes())
    node_features = []
    use_sequence = seq_weight > 0
    pair_weight = 1.0 - seq_weight
    for node in nodes:
        node_data = G.nodes[node]
        pair_val = 1.0 if node_data['label'] == 'paired' else 0.0
        loop_size_norm = float(node_data.get('loop_size_norm', 0.0))
        loop_pos_norm = float(node_data.get('loop_pos_norm', 0.0))

        features = [pair_weight * pair_val, loop_size_norm, loop_pos_norm]
        if use_sequence:
            base_vec = _one_hot_base(node_data.get('base'))
            features.extend(seq_weight * b for b in base_vec)
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'adjacent')
        attr_base = [1.0 if edge_type == 'adjacent' else 0.0,
                     1.0 if edge_type == 'base_pair' else 0.0]

        for src, dst in ((u, v), (v, u)):
            is_forward = 1.0 if src < dst else 0.0
            is_backward = 1.0 - is_forward
            edge_indices.append([src, dst])
            edge_attrs.append(attr_base + [is_forward, is_backward])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _classify_loop_segments(
    dotbracket: str,
    loop_segments: Iterable[List[int]],
    pair_map: List[int],
    branch_count: Dict[int, int],
) -> Dict[int, str]:
    """Assign loop categories inspired by FORGI to unpaired nucleotides."""

    categories: Dict[int, str] = {}
    for segment in loop_segments:
        if not segment:
            continue
        start_idx = segment[0]
        end_idx = segment[-1]

        enclosing = None
        search_pos = start_idx - 1
        while search_pos >= 0:
            partner = pair_map[search_pos]
            if partner > search_pos and partner >= end_idx:
                enclosing = search_pos
                break
            search_pos -= 1

        if enclosing is None:
            loop_type = 'external'
        else:
            branches = branch_count.get(enclosing, 0)
            if branches <= 0:
                loop_type = 'hairpin'
            elif branches == 1:
                loop_type = 'internal'
            else:
                loop_type = 'multi'

        for idx in segment:
            categories[idx] = loop_type

    return categories


def _build_base_edges(pair_map: List[int]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for idx, partner in enumerate(pair_map):
        if partner > idx:
            edges.append((idx, partner))
    return edges


def _structure_to_data_forgi(
    dotbracket: str,
    sequence: Optional[str] = None,
    seq_weight: float = 0.0,
) -> Optional[Data]:
    """Efficient PyG graph construction using FORGI-inspired node encodings."""

    num_nodes = len(dotbracket)
    if num_nodes == 0:
        return Data()

    loop_meta, loop_segments = _extract_loop_segments(dotbracket)
    pair_map, branch_count = _build_pair_map(dotbracket)
    loop_categories = _classify_loop_segments(dotbracket, loop_segments, pair_map, branch_count)
    base_pair_edges = _build_base_edges(pair_map)

    node_features: List[List[float]] = []
    pair_weight = max(0.0, min(1.0, 1.0 - seq_weight))
    use_sequence = seq_weight > 0

    category_to_index = {
        'hairpin': 0,
        'internal': 1,
        'multi': 2,
        'external': 3,
    }

    for idx in range(num_nodes):
        base = sequence[idx] if sequence is not None and idx < len(sequence) else None
        loop_info = loop_meta.get(idx)
        loop_size_norm = float(loop_info.get('loop_size_norm', 0.0)) if loop_info else 0.0
        loop_pos_norm = float(loop_info.get('loop_pos_norm', 0.0)) if loop_info else 0.0

        is_paired = 1.0 if pair_map[idx] != -1 else 0.0
        features = [pair_weight * is_paired]

        if is_paired:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            category = loop_categories.get(idx, 'external')
            one_hot = [0.0, 0.0, 0.0, 0.0]
            one_hot[category_to_index.get(category, 3)] = 1.0
            features.extend(one_hot)

        features.extend([loop_size_norm, loop_pos_norm])

        if use_sequence:
            features.extend(seq_weight * value for value in _one_hot_base(base))

        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    edge_pairs: List[Tuple[int, int, int]] = []
    for idx in range(1, num_nodes):
        edge_pairs.append((idx - 1, idx, 0))
        edge_pairs.append((idx, idx - 1, 0))
    for src, dst in base_pair_edges:
        edge_pairs.append((src, dst, 1))
        edge_pairs.append((dst, src, 1))

    if edge_pairs:
        edge_index = torch.tensor(
            [[src for src, _, _ in edge_pairs], [dst for _, dst, _ in edge_pairs]],
            dtype=torch.long,
        )
        edge_attr = torch.zeros((len(edge_pairs), 4), dtype=torch.float)
        for edge_idx, (src, dst, etype) in enumerate(edge_pairs):
            if etype == 0:
                edge_attr[edge_idx, 0] = 1.0
            else:
                edge_attr[edge_idx, 1] = 1.0
            if src < dst:
                edge_attr[edge_idx, 2] = 1.0
            else:
                edge_attr[edge_idx, 3] = 1.0
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def structure_to_data(
    dotbracket: str,
    sequence: Optional[str] = None,
    graph_encoding: str = "standard",
    seq_weight: float = 0.0,
) -> Optional[Data]:
    """Create a :class:`torch_geometric.data.Data` instance from a dot-bracket string."""

    if graph_encoding == "forgi":
        return _structure_to_data_forgi(dotbracket, sequence=sequence, seq_weight=seq_weight)

    graph = dotbracket_to_graph(dotbracket, sequence)
    if graph is None:
        return None
    return graph_to_tensor(graph, seq_weight)

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
