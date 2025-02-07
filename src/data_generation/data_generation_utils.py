#!/usr/bin/env python
"""
data_generation_utils.py

Utility functions for RNA triplet generation.
This module uses forgi and ViennaRNA to load and handle RNA secondary structures.
It includes functions to apply local modifications to stems and loops (hairpin, internal, bulge, multiloop)
subject to size constraints. For each eligible node the modification action (insertion vs deletion)
is chosen at random (with restrictions if the current size is at or near a boundary).
Detailed logging is provided to help with debugging.
"""

import random
import logging
import matplotlib.pyplot as plt
import time

import forgi.graph.bulge_graph as fgb
from ViennaRNA import fold  # Assume the ViennaRNA Python binding is installed

logger = logging.getLogger(__name__)

def generate_random_rna(length):
    """Return a random RNA sequence of the given length."""
    seq = ''.join(random.choices("ACGU", k=length))
    logger.debug("Generated random RNA (length %d): %s", length, seq)
    return seq

def predict_structure(seq):
    """Predict the MFE secondary structure for the RNA sequence using ViennaRNA."""
    structure, mfe = fold(seq)
    logger.debug("Predicted structure for sequence %s: %s (MFE: %s)", seq, structure, mfe)
    return structure

def get_node_mapping(structure, seq):
    try:
        bg = fgb.BulgeGraph.from_dotbracket(structure, seq)
    except TypeError:
        bg = fgb.BulgeGraph.from_dotbracket(structure)
    bg_str = bg.to_bg_string()
    mapping = {}
    for line in bg_str.splitlines():
        if line.startswith("define"):
            parts = line.split()
            node_name = parts[1]
            indices = [int(x) for x in parts[2:]]
            mapping[node_name] = indices
    logger.debug("Node mapping for structure %s with sequence %s: %s", structure, seq, mapping)
    return mapping

def get_base_pairing(structure):
    """Compute and return a list with the base pairing (0-indexed) for the dot-bracket structure."""
    stack = []
    pairing = [None] * len(structure)
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairing[i] = j
                pairing[j] = i
    logger.debug("Base pairing for structure %s: %s", structure, pairing)
    return pairing

def choose_action(current_size, min_size, max_size):
    """
    Return 'insert' if current_size is at or below min_size,
    'delete' if current_size is at or above max_size,
    otherwise randomly choose between "insert" and "delete".
    """
    if current_size <= min_size:
        return "insert"
    elif current_size >= max_size:
        return "delete"
    else:
        return random.choice(["insert", "delete"])

def modify_stem_insert(seq, structure):
    mapping = get_node_mapping(structure, seq)
    stem_nodes = [node for node in mapping if node.startswith("s")]
    if not stem_nodes:
        logging.warning("No stem nodes available for modification.")
        return seq, structure, mapping
    node = random.choice(stem_nodes)
    indices = sorted(mapping[node])
    if len(indices) < 2:
        logging.warning("Stem node %s is too short to insert.", node)
        return seq, structure, mapping
    insertion_index = indices[len(indices) // 2]
    pairing = get_base_pairing(structure)
    partner_index = pairing[insertion_index - 1]
    if partner_index is None:
        logging.warning("No pairing found for stem base at position %d", insertion_index)
        return seq, structure, mapping
    partner_index = partner_index + 1
    if insertion_index > partner_index:
        insertion_index, partner_index = partner_index, insertion_index
    pair_options = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]
    base1, base2 = random.choice(pair_options)
    seq_list = list(seq)
    struct_list = list(structure)
    seq_list.insert(insertion_index - 1, base1)
    struct_list.insert(insertion_index - 1, '(')
    seq_list.insert(partner_index, base2)
    struct_list.insert(partner_index, ')')
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    return new_seq, new_structure, new_mapping

def modify_stem_delete(seq, structure):
    mapping = get_node_mapping(structure, seq)
    pairing = get_base_pairing(structure)
    stem_nodes = [node for node in mapping if node.startswith("s")]
    if not stem_nodes:
        logging.warning("No stem nodes available for deletion.")
        return seq, structure, mapping
    node = random.choice(stem_nodes)
    indices = mapping[node]
    base_pairs = []
    for idx in indices:
        partner = pairing[idx - 1]
        if partner is not None and (partner + 1) in indices and idx < (partner + 1):
            base_pairs.append((idx, partner + 1))
    if not base_pairs:
        logging.warning("No base pairs found in stem %s for deletion.", node)
        return seq, structure, mapping
    pair_to_delete = random.choice(base_pairs)
    seq_list = [base for i, base in enumerate(seq, start=1) if i not in pair_to_delete]
    struct_list = [char for i, char in enumerate(structure, start=1) if i not in pair_to_delete]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    return new_seq, new_structure, new_mapping

def modify_stem(seq, structure, min_size, max_size, max_modifications, mod_counts):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("s") and len(mapping[node]) >= min_size 
                and mod_counts.get(node, 0) < max_modifications]
    if not eligible:
        logging.warning("No eligible stem nodes (min size %d and < %d modifications) for modification.", min_size, max_modifications)
        return seq, structure, mapping
    node = random.choice(eligible)
    current_size = len(mapping[node])
    action = choose_action(current_size, min_size, max_size)
    logging.debug("Modifying stem node %s (current size %d, modifications %d) with action: %s",
                  node, current_size, mod_counts.get(node, 0), action)
    if action == "insert":
        result = modify_stem_insert(seq, structure)
    else:
        result = modify_stem_delete(seq, structure)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    return result

def modify_hairpin(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("h") and 
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and 
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logging.warning("No eligible hairpin nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    current_size = len(indices)
    if action is None:
        action = choose_action(current_size, min_size if min_size is not None else 0,
                               max_size if max_size is not None else current_size+5)
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if current_size <= (min_size if min_size is not None else 1):
            logging.warning("Hairpin node %s is at minimum size; cannot delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    return new_seq, new_structure, new_mapping

def modify_internal_loop(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("i") and 
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logging.warning("No eligible internal loop nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    current_size = len(indices)
    if action is None:
        action = choose_action(current_size, min_size if min_size is not None else 0,
                               max_size if max_size is not None else current_size+5)
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if current_size <= (min_size if min_size is not None else 1):
            logging.warning("Internal loop node %s is at minimum size; cannot delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    return new_seq, new_structure, new_mapping

def modify_multiloop(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("m") and 
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and 
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logging.warning("No eligible multiloop nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    current_size = len(indices)
    if action is None:
        action = choose_action(current_size, min_size if min_size is not None else 0,
                               max_size if max_size is not None else current_size+5)
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if current_size <= (min_size if min_size is not None else 1):
            logging.warning("Multiloop node %s is at minimum size; cannot delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    return new_seq, new_structure, new_mapping

def modify_bulge(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("i") and len(mapping[node]) == 1 and
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logging.warning("No eligible bulge nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    action = action if action is not None else "insert"
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = indices[0]
        nucleotide = random.choice(["A", "C", "G", "U"])
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        logging.warning("Bulge node %s is of minimal size; cannot delete.", node)
        return seq, structure, mapping
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    return new_seq, new_structure, new_mapping

def dinuc_shuffle(seq):
    """Return a new sequence that preserves the original dinucleotide frequencies."""
    if len(seq) <= 1:
        logger.warning("Sequence length <= 1. Returning sequence unchanged.")
        return seq
    graph = {}
    for i in range(len(seq)-1):
        src = seq[i]
        dst = seq[i+1]
        graph.setdefault(src, []).append(dst)
    for src in graph:
        random.shuffle(graph[src])
    trail = []
    stack = [seq[0]]
    while stack:
        current = stack[-1]
        if current in graph and graph[current]:
            next_node = graph[current].pop()
            stack.append(next_node)
        else:
            trail.append(stack.pop())
    trail.reverse()
    if len(trail) != len(seq):
        logger.error("Eulerian trail length (%d) != sequence length (%d). Returning original sequence.", len(trail), len(seq))
        return seq
    new_seq = "".join(trail)
    logger.debug("Dinucleotide shuffled sequence: %s", new_seq)
    return new_seq

def generate_negative_sample(seq, allowed_variation=0):
    """Generate a negative sample by dinucleotide shuffling and optional length variation, then folding."""
    shuffled_seq = dinuc_shuffle(seq)
    if allowed_variation > 0:
        variation = random.choice(range(-allowed_variation, allowed_variation+1))
        if variation > 0:
            logger.debug("Increasing negative sample length by %d", variation)
            for _ in range(variation):
                pos = random.randint(1, len(shuffled_seq)+1)
                shuffled_seq = shuffled_seq[:pos-1] + random.choice("ACGU") + shuffled_seq[pos-1:]
        elif variation < 0 and len(shuffled_seq) + variation > 0:
            logger.debug("Decreasing negative sample length by %d", abs(variation))
            for _ in range(abs(variation)):
                pos = random.randint(1, len(shuffled_seq))
                shuffled_seq = shuffled_seq[:pos-1] + shuffled_seq[pos:]
    neg_structure = predict_structure(shuffled_seq)
    logger.debug("Generated negative sample: %s with structure %s", shuffled_seq, neg_structure)
    return shuffled_seq, neg_structure

def plot_rna_structure(seq, structure, ax=None):
    """
    Plot the RNA secondary structure using forgi’s matplotlib tools.
    If an Axes object is provided, plot into it; otherwise, use the current Axes.
    """
    try:
        import forgi.visual.mplotlib as fvm
        bg = fgb.BulgeGraph.from_dotbracket(structure, seq)
        if ax is None:
            ax = plt.gca()
        fvm.plot_rna(bg, ax=ax)
        ax.set_title(f"Sequence: {seq}")
        ax.axis("off")
    except Exception as e:
        logger.exception("Error while plotting RNA structure: %s", e)

def generate_triplet(seq_min_len, seq_max_len, seq_len_distribution, seq_len_mean, seq_len_sd,
                     neg_len_variation,
                     n_stem_indels, stem_min_size, stem_max_size, stem_max_n_modifications,
                     n_hloop_indels, hloop_min_size, hloop_max_size, hloop_max_n_modifications,
                     n_iloop_indels, iloop_min_size, iloop_max_size, iloop_max_n_modifications,
                     n_bulge_indels, bulge_min_size, bulge_max_size, bulge_max_n_modifications,
                     n_mloop_indels, mloop_min_size, mloop_max_size, mloop_max_n_modifications):
    if seq_len_distribution == "unif":
        length = random.randint(seq_min_len, seq_max_len)
    else:
        length = int(random.gauss(seq_len_mean, seq_len_sd))
        length = max(seq_min_len, min(seq_max_len, length))
    logger.debug("Selected sequence length: %d", length)
    anchor_seq = generate_random_rna(length)
    anchor_structure = predict_structure(anchor_seq)
    anchor_mapping = get_node_mapping(anchor_structure, anchor_seq)
    logger.info("Generated anchor sequence and structure.")
    
    pos_seq, pos_structure, pos_mapping = anchor_seq, anchor_structure, anchor_mapping
    
    mod_counts = {}
    
    for _ in range(n_stem_indels):
        pos_seq, pos_structure, pos_mapping = modify_stem(pos_seq, pos_structure,
                                                          stem_min_size, stem_max_size, stem_max_n_modifications,
                                                          mod_counts)
    
    for _ in range(n_hloop_indels):
        pos_seq, pos_structure, pos_mapping = modify_hairpin(pos_seq, pos_structure,
                                                             min_size=hloop_min_size,
                                                             max_size=hloop_max_size,
                                                             max_modifications=hloop_max_n_modifications,
                                                             mod_counts=mod_counts)
    for _ in range(n_iloop_indels):
        pos_seq, pos_structure, pos_mapping = modify_internal_loop(pos_seq, pos_structure,
                                                                   min_size=iloop_min_size,
                                                                   max_size=iloop_max_size,
                                                                   max_modifications=iloop_max_n_modifications,
                                                                   mod_counts=mod_counts)
    for _ in range(n_bulge_indels):
        pos_seq, pos_structure, pos_mapping = modify_bulge(pos_seq, pos_structure,
                                                           min_size=bulge_min_size,
                                                           max_size=bulge_max_size,
                                                           max_modifications=bulge_max_n_modifications,
                                                           mod_counts=mod_counts)
    for _ in range(n_mloop_indels):
        pos_seq, pos_structure, pos_mapping = modify_multiloop(pos_seq, pos_structure,
                                                               min_size=mloop_min_size,
                                                               max_size=mloop_max_size,
                                                               max_modifications=mloop_max_n_modifications,
                                                               mod_counts=mod_counts)
    neg_seq, neg_structure = generate_negative_sample(anchor_seq, neg_len_variation)
    
    triplet = {
        "anchor_seq": anchor_seq,
        "anchor_structure": anchor_structure,
        "positive_seq": pos_seq,
        "positive_structure": pos_structure,
        "negative_seq": neg_seq,
        "negative_structure": neg_structure
    }
    logger.info("Triplet generated.")
    return triplet

def generate_triplet_thread(thread_size, *args, **kwargs):
    return [generate_triplet(*args, **kwargs) for _ in range(thread_size)]

def split_dataset(df, train_fraction):
    train_df = df.sample(frac=train_fraction, random_state=42)
    val_df = df.drop(train_df.index)
    logger.debug("Dataset split: %d training, %d validation samples", len(train_df), len(val_df))
    return train_df, val_df
