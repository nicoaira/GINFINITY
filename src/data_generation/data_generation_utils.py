#!/usr/bin/env python
"""
data_generation_utils.py

Utility functions for RNA triplet generation.
This module uses forgi and ViennaRNA to load and handle RNA secondary structures,
and implements local modifications by directly updating both the RNA sequence and 
its dot-bracket structure without refolding.
Detailed logging is provided to help with debugging.
"""

import random
import json
import logging
import matplotlib.pyplot as plt

import forgi.graph.bulge_graph as fgb
from ViennaRNA import fold  # Using ViennaRNA's Python bindings for structure prediction

# Set up a module-level logger.
logger = logging.getLogger(__name__)


# ===============================
# Basic helper functions
# ===============================
def generate_random_rna(length):
    """Return a random RNA sequence of given length."""
    seq = ''.join(random.choices("ACGU", k=length))
    logger.debug("Generated random RNA (length %d): %s", length, seq)
    return seq


def predict_structure(seq):
    """
    Predict the MFE secondary structure for the RNA sequence using ViennaRNA Python bindings.
    (This is only used for the initial anchor and for negative sample prediction.)
    """
    structure, mfe = fold(seq)
    logger.debug("Predicted structure for sequence %s: %s (MFE: %s)", seq, structure, mfe)
    return structure


def get_node_mapping(structure, seq):
    # Pass the actual sequence to forgi so that it does not default to N's.
    bg = fgb.BulgeGraph.from_dotbracket(structure, seq)
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
    """
    Compute the base pairing for a given dot-bracket structure.
    Returns a list where pairing[i] is the index (0-indexed) of the nucleotide paired with position i,
    or None if unpaired.
    """
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


# ===============================
# Graph-based rearrangement
# ===============================
def rearrange_nodes(seq, structure, node1, node2):
    """
    Swap the sequences corresponding to two selected nodes (given by their names)
    by swapping the corresponding segments in both the RNA sequence and its structure.
    No refolding is performed.
    """
    mapping = get_node_mapping(structure, seq)
    if node1 not in mapping or node2 not in mapping:
        logger.warning("Nodes %s or %s not found in mapping. No rearrangement applied.", node1, node2)
        return seq, structure, mapping

    indices1 = sorted(mapping[node1])
    indices2 = sorted(mapping[node2])
    segment1_seq = "".join(seq[i - 1] for i in indices1)
    segment2_seq = "".join(seq[i - 1] for i in indices2)
    segment1_struct = "".join(structure[i - 1] for i in indices1)
    segment2_struct = "".join(structure[i - 1] for i in indices2)
    logger.debug("Swapping segments: node %s (%s/%s) with node %s (%s/%s)",
                 node1, segment1_seq, segment1_struct,
                 node2, segment2_seq, segment2_struct)

    # Convert sequence and structure to lists for mutability.
    seq_list = list(seq)
    struct_list = list(structure)
    # Replace positions in node1 with segment2.
    for idx, pos in enumerate(indices1):
        if idx < len(segment2_seq):
            seq_list[pos - 1] = segment2_seq[idx]
            struct_list[pos - 1] = segment2_struct[idx]
        else:
            # If segment lengths differ, remove extra positions.
            seq_list[pos - 1] = ''
            struct_list[pos - 1] = ''
    # Replace positions in node2 with segment1.
    for idx, pos in enumerate(indices2):
        if idx < len(segment1_seq):
            seq_list[pos - 1] = segment1_seq[idx]
            struct_list[pos - 1] = segment1_struct[idx]
        else:
            seq_list[pos - 1] = ''
            struct_list[pos - 1] = ''
    new_seq = "".join(ch for ch in seq_list if ch != '')
    new_structure = "".join(ch for ch in struct_list if ch != '')
    new_mapping = get_node_mapping(new_structure, new_seq)
    logger.debug("Rearranged sequence: %s\nNew structure: %s", new_seq, new_structure)
    return new_seq, new_structure, new_mapping


# ===============================
# Local modifications (indels) without refolding
# ===============================
def modify_hairpin(seq, structure, action="insert"):
    """
    Modify a hairpin loop (node starting with "h") by inserting or deleting one nucleotide.
    The dot-bracket structure is updated manually—no refolding is performed.
    """
    mapping = get_node_mapping(structure, seq)
    hairpin_nodes = [node for node in mapping if node.startswith("h")]
    if not hairpin_nodes:
        logger.warning("No hairpin nodes found for modification.")
        return seq, structure, mapping

    node = random.choice(hairpin_nodes)
    indices = mapping[node]
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        logger.debug("Inserting nucleotide %s at hairpin position %d (node %s)", nucleotide, insertion_position, node)
        seq_list.insert(insertion_position - 1, nucleotide)
        # In hairpins, unpaired positions are represented as dots.
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if len(indices) <= 1:
            logger.warning("Not enough nucleotides in hairpin %s to delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        logger.debug("Deleting nucleotide at hairpin position %d (node %s)", deletion_position, node)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    else:
        logger.error("Invalid action %s for hairpin modification", action)
        return seq, structure, mapping

    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    return new_seq, new_structure, new_mapping


def modify_internal_loop(seq, structure, action="insert"):
    """
    Modify an internal loop (node starting with "i") by inserting or deleting one nucleotide.
    The dot-bracket structure is updated manually.
    """
    mapping = get_node_mapping(structure, seq)
    internal_nodes = [node for node in mapping if node.startswith("i")]
    if not internal_nodes:
        logger.warning("No internal loop nodes found for modification.")
        return seq, structure, mapping

    node = random.choice(internal_nodes)
    indices = mapping[node]
    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        logger.debug("Inserting nucleotide %s at internal loop position %d (node %s)", nucleotide, insertion_position, node)
        seq_list.insert(insertion_position - 1, nucleotide)
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if len(indices) <= 1:
            logger.warning("Not enough nucleotides in internal loop %s to delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        logger.debug("Deleting nucleotide at internal loop position %d (node %s)", deletion_position, node)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    else:
        logger.error("Invalid action %s for internal loop modification", action)
        return seq, structure, mapping

    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    return new_seq, new_structure, new_mapping


def modify_multiloop(seq, structure, action="insert"):
    """
    Modify a multiloop (node starting with "m") by inserting or deleting one nucleotide.
    The dot-bracket structure is updated manually.
    """
    mapping = get_node_mapping(structure, seq)
    multiloop_nodes = [node for node in mapping if node.startswith("m")]
    if not multiloop_nodes:
        logger.warning("No multiloop nodes found for modification.")
        return seq, structure, mapping

    node = random.choice(multiloop_nodes)
    indices = mapping[node]
    
    # Check if the selected node has any indices
    if not indices:
        logger.warning("Multiloop node %s has no indices; skipping modification.", node)
        return seq, structure, mapping

    seq_list = list(seq)
    struct_list = list(structure)
    if action == "insert":
        insertion_position = random.choice(indices)
        nucleotide = random.choice(["A", "C", "G", "U"])
        logger.debug("Inserting nucleotide %s at multiloop position %d (node %s)", nucleotide, insertion_position, node)
        seq_list.insert(insertion_position - 1, nucleotide)
        # In multiloops, unpaired positions are represented as dots.
        struct_list.insert(insertion_position - 1, '.')
    elif action == "delete":
        if len(indices) <= 1:
            logger.warning("Not enough nucleotides in multiloop %s to delete.", node)
            return seq, structure, mapping
        deletion_position = random.choice(indices)
        logger.debug("Deleting nucleotide at multiloop position %d (node %s)", deletion_position, node)
        del seq_list[deletion_position - 1]
        del struct_list[deletion_position - 1]
    else:
        logger.error("Invalid action %s for multiloop modification", action)
        return seq, structure, mapping

    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    return new_seq, new_structure, new_mapping



def modify_stem_delete(seq, structure):
    """
    In a randomly selected stem, delete one base pair.
    Both the sequence and its structure (i.e. the corresponding '(' and ')') are removed.
    """
    mapping = get_node_mapping(structure, seq)
    pairing = get_base_pairing(structure)
    stem_nodes = [node for node in mapping if node.startswith("s")]
    if not stem_nodes:
        logger.warning("No stem nodes found for deletion.")
        return seq, structure, mapping

    node = random.choice(stem_nodes)
    indices = mapping[node]
    base_pairs = []
    for idx in indices:
        partner = pairing[idx - 1]
        if partner is not None and (partner + 1) in indices and idx < (partner + 1):
            base_pairs.append((idx, partner + 1))
    if not base_pairs:
        logger.warning("No base pairs found in stem %s for deletion.", node)
        return seq, structure, mapping

    pair_to_delete = random.choice(base_pairs)
    logger.debug("Deleting stem base pair at positions %s (node %s)", pair_to_delete, node)
    seq_list = list(seq)
    struct_list = list(structure)
    # Remove the two nucleotides and their corresponding structure characters.
    for pos in sorted(pair_to_delete, reverse=True):
        del seq_list[pos - 1]
        del struct_list[pos - 1]
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    return new_seq, new_structure, new_mapping


def modify_stem_insert(seq, structure):
    """
    In a randomly selected stem, insert an extra base pair.
    The new base pair (with its corresponding '(' and ')') is inserted without refolding.
    """
    mapping = get_node_mapping(structure, seq)
    stem_nodes = [node for node in mapping if node.startswith("s")]
    if not stem_nodes:
        logger.warning("No stem nodes found for insertion.")
        return seq, structure, mapping

    node = random.choice(stem_nodes)
    indices = sorted(mapping[node])
    # Choose a position roughly in the middle of the stem.
    insertion_index = indices[len(indices) // 2]
    pairing = get_base_pairing(structure)
    partner_index = pairing[insertion_index - 1]
    if partner_index is None:
        logger.warning("No pairing found for base at position %d in stem %s.", insertion_index, node)
        return seq, structure, mapping
    partner_index = partner_index + 1  # convert to 1-indexed
    # Determine left/right order.
    if insertion_index < partner_index:
        left_index = insertion_index
        right_index = partner_index
    else:
        left_index = partner_index
        right_index = insertion_index
    pair_options = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]
    base1, base2 = random.choice(pair_options)
    logger.debug("Inserting stem base pair (%s, %s) at positions %d and %d (node %s)",
                 base1, base2, left_index, right_index, node)
    seq_list = list(seq)
    struct_list = list(structure)
    # Insert the left base at left_index.
    seq_list.insert(left_index - 1, base1)
    struct_list.insert(left_index - 1, '(')
    # After left insertion, right_index shifts by 1.
    seq_list.insert(right_index, base2)
    struct_list.insert(right_index, ')')
    new_seq = "".join(seq_list)
    new_structure = "".join(struct_list)
    new_mapping = get_node_mapping(new_structure, new_seq)
    return new_seq, new_structure, new_mapping


# ===============================
# Dinucleotide shuffling and negative sample generation
# ===============================
def dinuc_shuffle(seq):
    """
    Return a new sequence that preserves the original dinucleotide frequencies.
    This is implemented by building a directed multigraph from overlapping dinucleotides
    and then finding a random Eulerian trail.
    """
    if len(seq) <= 1:
        logger.warning("Sequence length <= 1. Returning sequence unchanged.")
        return seq
    graph = {}
    for i in range(len(seq) - 1):
        src = seq[i]
        dst = seq[i + 1]
        graph.setdefault(src, []).append(dst)
    # Randomize the order of outgoing edges.
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
    """
    Generate a negative sample by:
      (a) Dinucleotide-shuffling the anchor sequence,
      (b) Optionally altering the length slightly by inserting or deleting random nucleotides,
      (c) Refolding the shuffled sequence using ViennaRNA's fold.
    """
    shuffled_seq = dinuc_shuffle(seq)
    if allowed_variation > 0:
        variation = random.choice(range(-allowed_variation, allowed_variation + 1))
        if variation > 0:
            logger.debug("Increasing negative sample length by %d", variation)
            for _ in range(variation):
                pos = random.randint(1, len(shuffled_seq) + 1)
                shuffled_seq = shuffled_seq[: pos - 1] + random.choice("ACGU") + shuffled_seq[pos - 1:]
        elif variation < 0 and len(shuffled_seq) + variation > 0:
            logger.debug("Decreasing negative sample length by %d", abs(variation))
            for _ in range(abs(variation)):
                pos = random.randint(1, len(shuffled_seq))
                shuffled_seq = shuffled_seq[: pos - 1] + shuffled_seq[pos:]
    neg_structure = predict_structure(shuffled_seq)
    logger.debug("Generated negative sample: %s with structure %s", shuffled_seq, neg_structure)
    return shuffled_seq, neg_structure


# ===============================
# Random modification dispatcher
# ===============================
def apply_random_modification(seq, structure):
    """
    Randomly choose one of several modification types and apply it.
    The modification updates both the sequence and the dot-bracket structure manually.
    """
    modifications = [
        "stem_delete",
        "stem_insert",
        "hairpin_insert",
        "hairpin_delete",
        "internal_insert",
        "internal_delete",
        "multiloop_insert",
        "multiloop_delete",
    ]
    mod = random.choice(modifications)
    logger.debug("Applying random modification: %s", mod)
    if mod == "stem_delete":
        return modify_stem_delete(seq, structure)
    elif mod == "stem_insert":
        return modify_stem_insert(seq, structure)
    elif mod == "hairpin_insert":
        return modify_hairpin(seq, structure, action="insert")
    elif mod == "hairpin_delete":
        return modify_hairpin(seq, structure, action="delete")
    elif mod == "internal_insert":
        return modify_internal_loop(seq, structure, action="insert")
    elif mod == "internal_delete":
        return modify_internal_loop(seq, structure, action="delete")
    elif mod == "multiloop_insert":
        return modify_multiloop(seq, structure, action="insert")
    elif mod == "multiloop_delete":
        return modify_multiloop(seq, structure, action="delete")
    else:
        logger.error("No valid modification selected. Returning original sequence.")
        return seq, structure, get_node_mapping(structure, seq)


# ===============================
# Triplet Generation Pipeline
# ===============================
def generate_triplet(
    min_length=50,
    max_length=100,
    length_distribution="uniform",
    mean=None,
    std=None,
    modification_cycles=1,
    allowed_variation=0,
):
    """
    Generate an RNA triplet consisting of:
      - Anchor: A random RNA sequence and its MFE structure (obtained via ViennaRNA).
      - Positive: A version of the anchor that has undergone a series of local structural modifications
        (with manual updates to the dot-bracket structure).
      - Negative: A dinucleotide-shuffled version of the anchor, refolded to produce its structure.
    """
    # Choose sequence length.
    if length_distribution == "uniform":
        length = random.randint(min_length, max_length)
    elif length_distribution == "normal" and mean is not None and std is not None:
        length = int(random.gauss(mean, std))
        length = max(min_length, min(max_length, length))
    else:
        length = random.randint(min_length, max_length)
    logger.debug("Selected sequence length: %d", length)
    
    # Generate anchor sample.
    anchor_seq = generate_random_rna(length)
    anchor_structure = predict_structure(anchor_seq)
    anchor_mapping = get_node_mapping(anchor_structure, anchor_seq)
    logger.info("Generated anchor sequence and structure.")

    # Generate positive sample: apply modifications iteratively.
    pos_seq = anchor_seq
    pos_structure = anchor_structure
    pos_mapping = anchor_mapping
    for cycle in range(modification_cycles):
        logger.debug("Positive modification cycle %d", cycle + 1)
        pos_seq, pos_structure, pos_mapping = apply_random_modification(pos_seq, pos_structure)
    
    # Generate negative sample.
    neg_seq, neg_structure = generate_negative_sample(anchor_seq, allowed_variation)
    
    triplet = {
        "anchor_seq": anchor_seq,
        "anchor_structure": anchor_structure,
        "positive_seq": pos_seq,
        "positive_structure": pos_structure,
        "negative_seq": neg_seq,
        "negative_structure": neg_structure,
    }
    logger.info("Triplet generated.")
    return triplet


# ===============================
# Visualization and Dataset Splitting
# ===============================
def plot_rna_structure(seq, structure, output_file):
    """
    Plot the RNA secondary structure using forgi’s matplotlib tools.
    The plot is saved to output_file.
    """
    try:
        import forgi.visual.mplotlib as fvm
        bg = fgb.BulgeGraph.from_dotbracket(structure)
        fvm.plot_rna(bg)
        plt.title(f"RNA Structure\nSequence: {seq}")
        plt.savefig(output_file)
        plt.close()
        logger.debug("Saved RNA structure plot to %s", output_file)
    except Exception as e:
        logger.exception("Error while plotting RNA structure: %s", e)


def split_dataset(df, train_fraction=0.8):
    """
    Split a Pandas DataFrame into training and validation sets.
    """
    train_df = df.sample(frac=train_fraction, random_state=42)
    val_df = df.drop(train_df.index)
    logger.debug("Dataset split: %d training, %d validation samples", len(train_df), len(val_df))
    return train_df, val_df
