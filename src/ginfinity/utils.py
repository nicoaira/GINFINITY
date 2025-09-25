from collections import defaultdict
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
# Graph encoding constants
# ==============================================================================

FORGI_NODE_TYPES = [
    "five_prime",
    "stem",
    "hairpin",
    "internal",
    "multiloop",
    "three_prime",
    "other",
]

FORGI_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(FORGI_NODE_TYPES)}

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

def dotbracket_to_graph(dotbracket, sequence=None, graph_encoding: str = "standard"):
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
    encoding = (graph_encoding or "standard").lower()
    G = nx.Graph()
    G.graph['graph_encoding'] = encoding
    G.graph['base_node_count'] = len(dotbracket)
    pair_stacks = defaultdict(list)
    bracket_pairs = {')': '(', ']': '[', '}': '{', '>': '<'}

    # Pre-compute loop membership metadata for unpaired positions
    seq_len = len(dotbracket)
    loop_meta = {}
    current_loop = []
    for idx, char in enumerate(dotbracket):
        if char == '.':
            current_loop.append(idx)
            continue
        if current_loop:
            loop_size = len(current_loop)
            norm_denom = max(1, seq_len)
            loop_size_norm = loop_size / norm_denom
            for pos_in_loop, node_idx in enumerate(current_loop):
                if loop_size > 1:
                    rel_pos = pos_in_loop / (loop_size - 1)
                else:
                    rel_pos = 0.5
                loop_meta[node_idx] = {
                    'loop_size': loop_size,
                    'loop_pos': pos_in_loop,
                    'loop_size_norm': loop_size_norm,
                    'loop_pos_norm': rel_pos,
                }
            current_loop = []
    if current_loop:
        loop_size = len(current_loop)
        norm_denom = max(1, seq_len)
        loop_size_norm = loop_size / norm_denom
        for pos_in_loop, node_idx in enumerate(current_loop):
            if loop_size > 1:
                rel_pos = pos_in_loop / (loop_size - 1)
            else:
                rel_pos = 0.5
            loop_meta[node_idx] = {
                'loop_size': loop_size,
                'loop_pos': pos_in_loop,
                'loop_size_norm': loop_size_norm,
                'loop_pos_norm': rel_pos,
            }

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
            'node_kind': 'base',
            'forgi_type': None,
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

    if encoding == 'standard':
        return G
    if encoding == 'forgi':
        return _augment_graph_with_forgi(G, dotbracket, sequence)
    raise ValueError(f"Unsupported graph_encoding '{graph_encoding}'")


def _build_forgi_type_map(bg):
    """Generate a mapping from Forgi element id to human-readable type."""
    type_map = {}
    for node in bg.stem_iterator():
        type_map[node] = 'stem'
    for node in getattr(bg, 'hloop_iterator', lambda: [])():
        type_map[node] = 'hairpin'
    for node in getattr(bg, 'iloop_iterator', lambda: [])():
        type_map[node] = 'internal'
    for node in getattr(bg, 'mloop_iterator', lambda: [])():
        type_map[node] = 'multiloop'
    for node in getattr(bg, 'floop_iterator', lambda: [])():
        type_map[node] = 'five_prime'
    for node in getattr(bg, 'tloop_iterator', lambda: [])():
        type_map[node] = 'three_prime'
    return type_map


def _augment_graph_with_forgi(G, dotbracket, sequence):
    """Add Forgi structural element nodes and edges to an existing base graph."""
    try:
        import forgi.graph.bulge_graph as fgb
    except ImportError as exc:
        raise RuntimeError(
            "Forgi graph encoding requires the 'forgi' package. Install it to use graph_encoding='forgi'."
        ) from exc

    bg = fgb.BulgeGraph.from_dotbracket(dotbracket, sequence) if sequence is not None else fgb.BulgeGraph.from_dotbracket(dotbracket)
    type_map = _build_forgi_type_map(bg)
    base_count = G.graph.get('base_node_count', len(dotbracket))

    forgi_indices = {}
    forgi_nodes = sorted(bg.defines.keys())
    for offset, node_name in enumerate(forgi_nodes):
        node_index = base_count + offset
        members = [pos - 1 for pos in bg.define_residue_num_iterator(node_name)]
        members = sorted({idx for idx in members if 0 <= idx < base_count})
        node_type = type_map.get(node_name, 'other')
        member_count = len(members)
        member_fraction = member_count / base_count if base_count > 0 else 0.0

        G.add_node(
            node_index,
            node_kind='forgi',
            forgi_id=node_name,
            forgi_type=node_type,
            members=members,
            member_count=member_count,
            member_fraction=member_fraction,
        )
        forgi_indices[node_name] = node_index

        for member in members:
            G.add_edge(node_index, member, edge_type='forgi_membership')

    for node_name, neighbors in getattr(bg, 'edges', {}).items():
        src_idx = forgi_indices.get(node_name)
        if src_idx is None:
            continue
        for neighbor in neighbors:
            dst_idx = forgi_indices.get(neighbor)
            if dst_idx is None or dst_idx == src_idx:
                continue
            if src_idx < dst_idx:
                G.add_edge(src_idx, dst_idx, edge_type='forgi_connection')

    G.graph['forgi_node_count'] = len(forgi_nodes)
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

def graph_to_tensor(G, seq_weight: float = 0.0, graph_encoding: str = None):
    """Convert a NetworkX graph to torch_geometric Data with weighted features."""

    encoding = graph_encoding or G.graph.get('graph_encoding', 'standard')
    encoding = (encoding or 'standard').lower()

    if encoding == 'standard':
        return _graph_to_tensor_standard(G, seq_weight)
    if encoding == 'forgi':
        return _graph_to_tensor_forgi(G, seq_weight)
    raise ValueError(f"Unsupported graph_encoding '{encoding}'")


def _graph_to_tensor_standard(G, seq_weight: float):
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


def _graph_to_tensor_forgi(G, seq_weight: float):
    nodes = sorted(G.nodes())
    node_features = []
    use_sequence = seq_weight > 0
    pair_weight = 1.0 - seq_weight
    forgi_dim = len(FORGI_NODE_TYPES)

    for node in nodes:
        node_data = G.nodes[node]
        node_kind = node_data.get('node_kind', 'base')
        is_base = 1.0 if node_kind == 'base' else 0.0

        pair_val = 1.0 if (node_kind == 'base' and node_data.get('label') == 'paired') else 0.0
        loop_size_norm = float(node_data.get('loop_size_norm', 0.0)) if node_kind == 'base' else 0.0
        loop_pos_norm = float(node_data.get('loop_pos_norm', 0.0)) if node_kind == 'base' else 0.0

        base_features = [pair_weight * pair_val, loop_size_norm, loop_pos_norm]

        if use_sequence and node_kind == 'base':
            base_vec = _one_hot_base(node_data.get('base'))
            seq_features = [seq_weight * b for b in base_vec]
        else:
            seq_features = [0.0, 0.0, 0.0, 0.0]

        features = base_features + seq_features
        features.append(is_base)

        forgi_type_vec = [0.0] * forgi_dim
        if node_kind == 'forgi':
            forgi_type = node_data.get('forgi_type', 'other')
            type_idx = FORGI_TYPE_TO_INDEX.get(forgi_type, FORGI_TYPE_TO_INDEX['other'])
            forgi_type_vec[type_idx] = 1.0
        features.extend(forgi_type_vec)

        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for u, v, data in G.edges(data=True):
        base_edge_type = data.get('edge_type', 'adjacent')
        for src, dst in ((u, v), (v, u)):
            attr_vec = [0.0, 0.0, 0.0, 0.0, 0.0]
            src_kind = G.nodes[src].get('node_kind', 'base')
            dst_kind = G.nodes[dst].get('node_kind', 'base')

            if base_edge_type == 'adjacent':
                attr_vec[0] = 1.0
            elif base_edge_type == 'base_pair':
                attr_vec[1] = 1.0
            elif base_edge_type == 'forgi_connection':
                attr_vec[4] = 1.0
            elif base_edge_type == 'forgi_membership':
                if src_kind == 'forgi' and dst_kind == 'base':
                    attr_vec[2] = 1.0  # parent -> child
                elif src_kind == 'base' and dst_kind == 'forgi':
                    attr_vec[3] = 1.0  # child -> parent
            else:
                attr_vec[4] = 1.0

            is_forward = 1.0 if src < dst else 0.0
            is_backward = 1.0 - is_forward
            edge_indices.append([src, dst])
            edge_attrs.append(attr_vec + [is_forward, is_backward])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
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
