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
logging.getLogger("forgi").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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
    """Compute and return a list with the base pairing (0-indexed) for the dot‐bracket structure."""
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
        choice = random.choice(["insert", "delete"])
        logger.debug("Chose action '%s' for current_size=%d, min=%d, max=%d", choice, current_size, min_size, max_size)
        return choice

# --- Stem Modification Functions ---

def modify_stem_insert(seq, structure):
    mapping = get_node_mapping(structure, seq)
    stem_nodes = [node for node in mapping if node.startswith("s")]
    if not stem_nodes:
        logger.warning("No stem nodes available for insertion modification.")
        return seq, structure, mapping
    node = random.choice(stem_nodes)
    indices = sorted(mapping[node])
    logger.debug("Stem insert: Chosen node %s with indices %s", node, indices)
    if len(indices) < 2:
        logger.warning("Stem node %s is too short for insertion.", node)
        return seq, structure, mapping
    insertion_index = indices[len(indices) // 2]
    pairing = get_base_pairing(structure)
    partner_index = pairing[insertion_index - 1]
    if partner_index is None:
        logger.warning("No pairing found for stem base at position %d", insertion_index)
        return seq, structure, mapping
    partner_index = partner_index + 1
    if insertion_index > partner_index:
        insertion_index, partner_index = partner_index, insertion_index
    pair_options = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]
    base1, base2 = random.choice(pair_options)
    logger.debug("Stem insert: Inserting bases %s and %s at positions %d and %d in node %s", base1, base2, insertion_index, partner_index, node)
    seq_list = list(seq)
    struct_list = list(structure)
    seq_list.insert(insertion_index - 1, base1)
    struct_list.insert(insertion_index - 1, '(')
    seq_list.insert(partner_index, base2)
    struct_list.insert(partner_index, ')')
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    logger.debug("Stem insert: New sequence length %d, new structure: %s", len(new_seq), new_structure)
    return new_seq, new_structure, new_mapping

def modify_stem_delete(seq, structure):
    mapping = get_node_mapping(structure, seq)
    pairing = get_base_pairing(structure)
    stem_nodes = [node for node in mapping if node.startswith("s")]
    if not stem_nodes:
        logger.warning("No stem nodes available for deletion modification.")
        return seq, structure, mapping
    node = random.choice(stem_nodes)
    indices = mapping[node]
    logger.debug("Stem delete: Chosen node %s with indices %s", node, indices)
    base_pairs = []
    for idx in indices:
        partner = pairing[idx - 1]
        if partner is not None and (partner + 1) in indices and idx < (partner + 1):
            base_pairs.append((idx, partner + 1))
    if not base_pairs:
        logger.warning("No base pairs found in stem node %s for deletion.", node)
        return seq, structure, mapping
    pair_to_delete = random.choice(base_pairs)
    logger.debug("Stem delete: Deleting base pair at positions %s in node %s", pair_to_delete, node)
    seq_list = [base for i, base in enumerate(seq, start=1) if i not in pair_to_delete]
    struct_list = [char for i, char in enumerate(structure, start=1) if i not in pair_to_delete]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    logger.debug("Stem delete: New sequence length %d, new structure: %s", len(new_seq), new_structure)
    return new_seq, new_structure, new_mapping

def modify_stem(seq, structure, min_size, max_size, max_modifications, mod_counts):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("s") and len(mapping[node]) >= min_size 
                and mod_counts.get(node, 0) < max_modifications]
    if not eligible:
        logger.warning("No eligible stem nodes (min size %d and < %d modifications) for modification.", min_size, max_modifications)
        return seq, structure, mapping
    node = random.choice(eligible)
    current_size = len(mapping[node])
    action = choose_action(current_size, min_size, max_size)
    logger.debug("Stem modification: Chosen node %s with size %d; action=%s; current mod count=%d", 
                 node, current_size, action, mod_counts.get(node, 0))
    if action == "insert":
        result = modify_stem_insert(seq, structure)
    else:
        result = modify_stem_delete(seq, structure)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    return result

# --- Hairpin Loop Modification ---

def modify_hairpin(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("h") and 
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and 
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logger.warning("No eligible hairpin nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    current_size = len(indices)
    if action is None:
        action = choose_action(current_size, min_size if min_size is not None else 0,
                               max_size if max_size is not None else current_size+5)
    logger.debug("Hairpin modification: Node %s with size %d; chosen action: %s", node, current_size, action)
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        logger.debug("Hairpin insertion: Inserting %s at position %d in node %s", nucleotide, insertion_position, node)
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if current_size <= (min_size if min_size is not None else 1):
            logger.warning("Hairpin node %s is at minimum size; cannot delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        logger.debug("Hairpin deletion: Deleting base at position %d in node %s", deletion_position, node)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    logger.debug("Hairpin modification: New sequence length %d", len(new_seq))
    return new_seq, new_structure, new_mapping

# --- Internal Loop Modification ---

def modify_internal_loop(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("i") and 
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logger.warning("No eligible internal loop nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    current_size = len(indices)
    if action is None:
        action = choose_action(current_size, min_size if min_size is not None else 0,
                               max_size if max_size is not None else current_size+5)
    logger.debug("Internal loop modification: Node %s with size %d; action: %s", node, current_size, action)
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        logger.debug("Internal loop insertion: Inserting %s at position %d in node %s", nucleotide, insertion_position, node)
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if current_size <= (min_size if min_size is not None else 1):
            logger.warning("Internal loop node %s is at minimum size; cannot delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        logger.debug("Internal loop deletion: Deleting base at position %d in node %s", deletion_position, node)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    logger.debug("Internal loop modification: New sequence length %d", len(new_seq))
    return new_seq, new_structure, new_mapping

# --- Multiloop Modification ---

def modify_multiloop(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("m") and 
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and 
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logger.warning("No eligible multiloop nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    current_size = len(indices)
    if action is None:
        action = choose_action(current_size, min_size if min_size is not None else 0,
                               max_size if max_size is not None else current_size+5)
    logger.debug("Multiloop modification: Node %s with size %d; action: %s", node, current_size, action)
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        logger.debug("Multiloop insertion: Inserting %s at position %d in node %s", nucleotide, insertion_position, node)
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if current_size <= (min_size if min_size is not None else 1):
            logger.warning("Multiloop node %s is at minimum size; cannot delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        logger.debug("Multiloop deletion: Deleting base at position %d in node %s", deletion_position, node)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    logger.debug("Multiloop modification: New sequence length %d", len(new_seq))
    return new_seq, new_structure, new_mapping

# --- Bulge Modification ---

def modify_bulge(seq, structure, action=None, min_size=None, max_size=None, max_modifications=None, mod_counts=None):
    mapping = get_node_mapping(structure, seq)
    eligible = [node for node in mapping if node.startswith("i") and len(mapping[node]) == 1 and
                (min_size is None or len(mapping[node]) >= min_size) and 
                (max_size is None or len(mapping[node]) <= max_size) and
                (max_modifications is None or mod_counts.get(node, 0) < max_modifications)]
    if not eligible:
        logger.warning("No eligible bulge nodes for modification.")
        return seq, structure, mapping
    node = random.choice(eligible)
    indices = mapping[node]
    action = action if action is not None else "insert"
    logger.debug("Bulge modification: Chosen node %s; action: %s", node, action)
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = indices[0]
        nucleotide = random.choice(["A", "C", "G", "U"])
        logger.debug("Bulge insertion: Inserting %s at position %d in node %s", nucleotide, insertion_position, node)
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        logger.warning("Bulge node %s is of minimal size; deletion not permitted.", node)
        return seq, structure, mapping
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    mod_counts[node] = mod_counts.get(node, 0) + 1
    logger.debug("Bulge modification: New sequence length %d", len(new_seq))
    return new_seq, new_structure, new_mapping

# --- Dinucleotide Shuffling and Negative Sample ---

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

# --- Plotting ---

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

# --- Triplet Generation Pipeline ---

def generate_triplet(seq_min_len, seq_max_len, seq_len_distribution, seq_len_mean, seq_len_sd,
                     neg_len_variation,
                     n_stem_indels, stem_min_size, stem_max_size, stem_max_n_modifications,
                     n_hloop_indels, hloop_min_size, hloop_max_size, hloop_max_n_modifications,
                     n_iloop_indels, iloop_min_size, iloop_max_size, iloop_max_n_modifications,
                     n_bulge_indels, bulge_min_size, bulge_max_size, bulge_max_n_modifications,
                     n_mloop_indels, mloop_min_size, mloop_max_size, mloop_max_n_modifications,
                     appending_event_probability, both_sides_appending_probability,
                     linker_min, linker_max, appending_size_factor,
                     mod_normalization, normalization_len):
    # Choose sequence length.
    if seq_len_distribution == "unif":
        length = random.randint(seq_min_len, seq_max_len)
    else:
        length = int(random.gauss(seq_len_mean, seq_len_sd))
        length = max(seq_min_len, min(seq_max_len, length))
    logger.debug("Selected sequence length: %d", length)
    # Generate anchor.
    anchor_seq = generate_random_rna(length)
    anchor_structure = predict_structure(anchor_seq)
    anchor_mapping = get_node_mapping(anchor_structure, anchor_seq)
    logger.info("Generated anchor sequence and structure.")
    
    pos_seq, pos_structure, pos_mapping = anchor_seq, anchor_structure, anchor_mapping

    # --- Modification Normalization ---
    if mod_normalization:
        factor = len(anchor_seq) / normalization_len
        if factor < 1:
            factor = 1
        # Scale modification counts if greater than 1.
        if n_stem_indels > 1:
            n_stem_indels = max(1, round(n_stem_indels * factor))
        if n_hloop_indels > 1:
            n_hloop_indels = max(1, round(n_hloop_indels * factor))
        if n_iloop_indels > 1:
            n_iloop_indels = max(1, round(n_iloop_indels * factor))
        if n_bulge_indels > 1:
            n_bulge_indels = max(1, round(n_bulge_indels * factor))
        if n_mloop_indels > 1:
            n_mloop_indels = max(1, round(n_mloop_indels * factor))
        logger.debug("Normalized mod counts with factor %.2f: stem=%d, hloop=%d, iloop=%d, bulge=%d, mloop=%d",
                     factor, n_stem_indels, n_hloop_indels, n_iloop_indels, n_bulge_indels, n_mloop_indels)
    
    mod_counts = {}
    
    # --- Apply Stem Modifications ---
    for i in range(n_stem_indels):
        logger.debug("Stem modification cycle %d", i+1)
        pos_seq, pos_structure, pos_mapping = modify_stem(pos_seq, pos_structure,
                                                          stem_min_size, stem_max_size, stem_max_n_modifications,
                                                          mod_counts)
    
    # --- Apply Hairpin Loop Modifications ---
    for i in range(n_hloop_indels):
        logger.debug("Hairpin loop modification cycle %d", i+1)
        pos_seq, pos_structure, pos_mapping = modify_hairpin(pos_seq, pos_structure,
                                                             min_size=hloop_min_size,
                                                             max_size=hloop_max_size,
                                                             max_modifications=hloop_max_n_modifications,
                                                             mod_counts=mod_counts)
    # --- Apply Internal Loop Modifications ---
    for i in range(n_iloop_indels):
        logger.debug("Internal loop modification cycle %d", i+1)
        pos_seq, pos_structure, pos_mapping = modify_internal_loop(pos_seq, pos_structure,
                                                                   min_size=iloop_min_size,
                                                                   max_size=iloop_max_size,
                                                                   max_modifications=iloop_max_n_modifications,
                                                                   mod_counts=mod_counts)
    # --- Apply Bulge Modifications ---
    for i in range(n_bulge_indels):
        logger.debug("Bulge modification cycle %d", i+1)
        pos_seq, pos_structure, pos_mapping = modify_bulge(pos_seq, pos_structure,
                                                           min_size=bulge_min_size,
                                                           max_size=bulge_max_size,
                                                           max_modifications=bulge_max_n_modifications,
                                                           mod_counts=mod_counts)
    # --- Apply Multiloop Modifications ---
    for i in range(n_mloop_indels):
        logger.debug("Multiloop modification cycle %d", i+1)
        pos_seq, pos_structure, pos_mapping = modify_multiloop(pos_seq, pos_structure,
                                                               min_size=mloop_min_size,
                                                               max_size=mloop_max_size,
                                                               max_modifications=mloop_max_n_modifications,
                                                               mod_counts=mod_counts)
    # Generate negative sample from anchor.
    neg_seq, neg_structure = generate_negative_sample(anchor_seq, neg_len_variation)
    
    # --- Appending Event ---
    if random.random() < appending_event_probability:
        logger.debug("Appending event triggered.")
        r = random.random()
        p_both = both_sides_appending_probability
        p_left = (1 - p_both) / 2
        p_right = p_left
        mean_append = len(anchor_seq) * appending_size_factor
        sigma_append = mean_append / 2
        def sample_append_length():
            L_app = int(random.gauss(mean_append, sigma_append))
            return max(1, L_app)
        linker_length = random.randint(linker_min, linker_max)
        linker_seq = generate_random_rna(linker_length)
        linker_structure = '.' * linker_length
        logger.debug("Linker: length=%d, sequence=%s", linker_length, linker_seq)
        if r < p_left:
            logger.debug("Appending to left only.")
            L_app = sample_append_length()
            appended_seq = generate_random_rna(L_app)
            appended_structure = predict_structure(appended_seq)
            pos_seq = appended_seq + linker_seq + pos_seq
            pos_structure = appended_structure + linker_structure + pos_structure
            neg_seq = appended_seq + linker_seq + neg_seq
            neg_structure = appended_structure + linker_structure + neg_structure
        elif r < p_left + p_right:
            logger.debug("Appending to right only.")
            L_app = sample_append_length()
            appended_seq = generate_random_rna(L_app)
            appended_structure = predict_structure(appended_seq)
            pos_seq = pos_seq + linker_seq + appended_seq
            pos_structure = pos_structure + linker_structure + appended_structure
            neg_seq = neg_seq + linker_seq + appended_seq
            neg_structure = neg_structure + linker_structure + appended_structure
        else:
            logger.debug("Appending to both sides.")
            L_app_left = sample_append_length()
            appended_seq_left = generate_random_rna(L_app_left)
            appended_structure_left = predict_structure(appended_seq_left)
            L_app_right = sample_append_length()
            appended_seq_right = generate_random_rna(L_app_right)
            appended_structure_right = predict_structure(appended_seq_right)
            pos_seq = appended_seq_left + linker_seq + pos_seq + linker_seq + appended_seq_right
            pos_structure = appended_structure_left + linker_structure + pos_structure + linker_structure + appended_structure_right
            neg_seq = appended_seq_left + linker_seq + neg_seq + linker_seq + appended_seq_right
            neg_structure = appended_structure_left + linker_structure + neg_structure + linker_structure + appended_structure_right
    
    triplet = {
        "anchor_seq": anchor_seq,
        "anchor_structure": anchor_structure,
        "positive_seq": pos_seq,
        "positive_structure": pos_structure,
        "negative_seq": neg_seq,
        "negative_structure": neg_structure
    }
    logger.info("Triplet generated (anchor length %d).", len(anchor_seq))
    return triplet

# NEW: Function to generate an anchor sequence and structure.
def generate_anchor(seq_min_len, seq_max_len, seq_len_distribution, seq_len_mean, seq_len_sd):
    if seq_len_distribution == "unif":
        length = random.randint(seq_min_len, seq_max_len)
    else:
        length = int(random.gauss(seq_len_mean, seq_len_sd))
        length = max(seq_min_len, min(seq_max_len, length))
    anchor_seq = generate_random_rna(length)
    anchor_structure = predict_structure(anchor_seq)
    anchor_mapping = get_node_mapping(anchor_structure, anchor_seq)
    logger.debug("Generated anchor (length %d)", len(anchor_seq))
    return anchor_seq, anchor_structure, anchor_mapping

# NEW: Function to generate a triplet from a pre-created anchor.
def generate_triplet_from_anchor(anchor_seq, anchor_structure,
                                 neg_len_variation,
                                 n_stem_indels, stem_min_size, stem_max_size, stem_max_n_modifications,
                                 n_hloop_indels, hloop_min_size, hloop_max_size, hloop_max_n_modifications,
                                 n_iloop_indels, iloop_min_size, iloop_max_size, iloop_max_n_modifications,
                                 n_bulge_indels, bulge_min_size, bulge_max_size, bulge_max_n_modifications,
                                 n_mloop_indels, mloop_min_size, mloop_max_size, mloop_max_n_modifications,
                                 appending_event_probability, both_sides_appending_probability,
                                 linker_min, linker_max, appending_size_factor,
                                 mod_normalization, normalization_len):
    pos_seq, pos_structure = anchor_seq, anchor_structure
    # Modification normalization.
    if mod_normalization:
        factor = len(anchor_seq) / normalization_len
        if factor < 1:
            factor = 1
        if n_stem_indels > 1:
            n_stem_indels = max(1, round(n_stem_indels * factor))
        if n_hloop_indels > 1:
            n_hloop_indels = max(1, round(n_hloop_indels * factor))
        if n_iloop_indels > 1:
            n_iloop_indels = max(1, round(n_iloop_indels * factor))
        if n_bulge_indels > 1:
            n_bulge_indels = max(1, round(n_bulge_indels * factor))
        if n_mloop_indels > 1:
            n_mloop_indels = max(1, round(n_mloop_indels * factor))
        logger.debug("Normalized mod counts with factor %.2f", factor)
    mod_counts = {}
    # Apply Stem Modifications.
    for i in range(n_stem_indels):
        logger.debug("Stem modification cycle %d", i+1)
        pos_seq, pos_structure, _ = modify_stem(pos_seq, pos_structure,
                                                stem_min_size, stem_max_size, stem_max_n_modifications,
                                                mod_counts)
    # Apply Hairpin Loop Modifications.
    for i in range(n_hloop_indels):
        logger.debug("Hairpin loop modification cycle %d", i+1)
        pos_seq, pos_structure, _ = modify_hairpin(pos_seq, pos_structure,
                                                   min_size=hloop_min_size,
                                                   max_size=hloop_max_size,
                                                   max_modifications=hloop_max_n_modifications,
                                                   mod_counts=mod_counts)
    # Apply Internal Loop Modifications.
    for i in range(n_iloop_indels):
        logger.debug("Internal loop modification cycle %d", i+1)
        pos_seq, pos_structure, _ = modify_internal_loop(pos_seq, pos_structure,
                                                         min_size=iloop_min_size,
                                                         max_size=iloop_max_size,
                                                         max_modifications=iloop_max_n_modifications,
                                                         mod_counts=mod_counts)
    # Apply Bulge Modifications.
    for i in range(n_bulge_indels):
        logger.debug("Bulge modification cycle %d", i+1)
        pos_seq, pos_structure, _ = modify_bulge(pos_seq, pos_structure,
                                                 min_size=bulge_min_size,
                                                 max_size=bulge_max_size,
                                                 max_modifications=bulge_max_n_modifications,
                                                 mod_counts=mod_counts)
    # Apply Multiloop Modifications.
    for i in range(n_mloop_indels):
        logger.debug("Multiloop modification cycle %d", i+1)
        pos_seq, pos_structure, _ = modify_multiloop(pos_seq, pos_structure,
                                                     min_size=mloop_min_size,
                                                     max_size=mloop_max_size,
                                                     max_modifications=mloop_max_n_modifications,
                                                     mod_counts=mod_counts)
    neg_seq, neg_structure = generate_negative_sample(anchor_seq, neg_len_variation)
    # Appending Event.
    if random.random() < appending_event_probability:
        logger.debug("Appending event triggered.")
        r = random.random()
        p_both = both_sides_appending_probability
        p_left = (1 - p_both) / 2
        p_right = p_left
        mean_append = len(anchor_seq) * appending_size_factor
        sigma_append = mean_append / 2
        def sample_append_length():
            L_app = int(random.gauss(mean_append, sigma_append))
            return max(1, L_app)
        linker_length = random.randint(linker_min, linker_max)
        linker_seq = generate_random_rna(linker_length)
        linker_structure = '.' * linker_length
        logger.debug("Linker: length=%d", linker_length)
        if r < p_left:
            logger.debug("Appending to left only.")
            L_app = sample_append_length()
            appended_seq = generate_random_rna(L_app)
            appended_structure = predict_structure(appended_seq)
            pos_seq = appended_seq + linker_seq + pos_seq
            pos_structure = appended_structure + linker_structure + pos_structure
            neg_seq = appended_seq + linker_seq + neg_seq
            neg_structure = appended_structure + linker_structure + neg_structure
        elif r < p_left + p_right:
            logger.debug("Appending to right only.")
            L_app = sample_append_length()
            appended_seq = generate_random_rna(L_app)
            appended_structure = predict_structure(appended_seq)
            pos_seq = pos_seq + linker_seq + appended_seq
            pos_structure = pos_structure + linker_structure + appended_structure
            neg_seq = neg_seq + linker_seq + appended_seq
            neg_structure = neg_structure + linker_structure + appended_structure
        else:
            logger.debug("Appending to both sides.")
            L_app_left = sample_append_length()
            appended_seq_left = generate_random_rna(L_app_left)
            appended_structure_left = predict_structure(appended_seq_left)
            L_app_right = sample_append_length()
            appended_seq_right = generate_random_rna(L_app_right)
            appended_structure_right = predict_structure(appended_seq_right)
            pos_seq = appended_seq_left + linker_seq + pos_seq + linker_seq + appended_seq_right
            pos_structure = appended_structure_left + linker_structure + pos_structure + linker_structure + appended_structure_right
            neg_seq = appended_seq_left + linker_seq + neg_seq + linker_seq + appended_seq_right
            neg_structure = appended_structure_left + linker_structure + neg_structure + linker_structure + appended_structure_right
    triplet = {
        "anchor_seq": anchor_seq,
        "anchor_structure": anchor_structure,
        "positive_seq": pos_seq,
        "positive_structure": pos_structure,
        "negative_seq": neg_seq,
        "negative_structure": neg_structure
    }
    logger.info("Triplet generated from anchor (anchor length %d)", len(anchor_seq))
    return triplet

# NEW: Update to process a batch of anchors.
def generate_triplet_thread(anchors, *args, **kwargs):
    # 'anchors' is a list of tuples (anchor_seq, anchor_structure, anchor_mapping)
    return [generate_triplet_from_anchor(a_seq, a_struct, *args, **kwargs) for a_seq, a_struct, _ in anchors]

def split_dataset(df, train_fraction):
    train_df = df.sample(frac=train_fraction, random_state=42)
    val_df = df.drop(train_df.index)
    logger.debug("Dataset split: %d training, %d validation samples", len(train_df), len(val_df))
    return train_df, val_df
