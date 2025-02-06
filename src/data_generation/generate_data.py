#!/usr/bin/env python
"""
generate_data.py

Main script for RNA triplet dataset generation.

This script now supports a --debug flag that not only prints detailed log messages to the console,
but also writes them to a file named "debug_<timestamp>.log" in the output directory.
It now assigns sequential triplet IDs (from 0 to n–1) and plots each triplet (anchor, positive, negative)
as a row of subplots in one figure, saving each figure as "triplet_<id>.png" in the plot directory.
"""

import argparse
import os
import json
import uuid
import datetime
import logging
import random
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import functions from the utility module.
from data_generation_utils import (
    generate_triplet,
    plot_rna_structure,  # We assume plot_rna_structure accepts an optional 'ax' argument
    split_dataset,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="RNA Structure Triplet Dataset Generator with forgi"
    )
    # Sequence generation parameters.
    parser.add_argument("--num_structures", type=int, default=100,
                        help="Number of RNA triplets to generate")
    parser.add_argument("--min_length", type=int, default=50,
                        help="Minimum RNA sequence length")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum RNA sequence length")
    parser.add_argument("--length_distribution", choices=["uniform", "normal"],
                        default="uniform", help="Type of length distribution")
    parser.add_argument("--mean", type=float, default=75,
                        help="Mean length for normal distribution")
    parser.add_argument("--std", type=float, default=10,
                        help="Standard deviation for normal distribution")
    # Modification settings.
    parser.add_argument("--modification_cycles", type=int, default=1,
                        help="Number of modification cycles for the positive sample")
    # Negative sample generation.
    parser.add_argument("--neg_allowed_variation", type=int, default=0,
                        help="Allowed length variation (in nucleotides) for negative sample")
    # Performance settings.
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for parallel generation")
    # Visualization options.
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save structure plots for a subset of triplets")
    parser.add_argument("--num_plots", type=int, default=5,
                        help="Number of triplets to plot")
    # Dataset splitting.
    parser.add_argument("--split", action="store_true",
                        help="Split the dataset into training and validation sets")
    parser.add_argument("--train_fraction", type=float, default=0.8,
                        help="Fraction of data to use for training (rest is validation)")
    # Output directory.
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output CSV, plots, and metadata")
    # Debug option.
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging for detailed output")
    return parser.parse_args()


def setup_logging(output_dir, debug_flag):
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # When debug is enabled, set level to DEBUG.
    # Otherwise, set level to ERROR so that warnings and lower messages are not printed.
    level = logging.DEBUG if debug_flag else logging.ERROR
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler.
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"debug_{timestamp}.log")
    fh = logging.FileHandler(log_filename)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("Logging is configured. Debug output will be saved to %s", log_filename)


def main():
    args = parse_arguments()

    # Create output directory if it does not exist.
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot:
        plot_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

    # Set up logging.
    setup_logging(args.output_dir, args.debug)
    logging.info("Starting RNA triplet generation with parameters: %s", vars(args))

    # Generate metadata.
    metadata = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": vars(args),
    }
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info("Metadata saved to %s", metadata_file)

    # Generate RNA triplets in parallel.
    triplets = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                generate_triplet,
                min_length=args.min_length,
                max_length=args.max_length,
                length_distribution=args.length_distribution,
                mean=args.mean,
                std=args.std,
                modification_cycles=args.modification_cycles,
                allowed_variation=args.neg_allowed_variation,
            )
            for _ in range(args.num_structures)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating triplets"):
            try:
                triplet = future.result()
                triplets.append(triplet)
            except Exception:
                logging.exception("Error generating triplet:")

    # Organize the triplets into a DataFrame.
    df = pd.DataFrame(triplets)
    # Instead of UUIDs, assign sequential triplet IDs (0, 1, 2, …)
    df["triplet_id"] = df.index

    # Save the full dataset CSV with a metadata header.
    output_csv = os.path.join(args.output_dir, "rna_triplets.csv")
    with open(output_csv, "w") as f:
        f.write("# Metadata: " + json.dumps(metadata) + "\n")
    df.to_csv(output_csv, mode="a", index=False)
    logging.info("Full dataset saved to %s", output_csv)

    # If splitting is enabled, create training and validation CSVs.
    if args.split:
        train_df, val_df = split_dataset(df, train_fraction=args.train_fraction)
        train_csv = os.path.join(args.output_dir, "rna_triplets_train.csv")
        val_csv = os.path.join(args.output_dir, "rna_triplets_val.csv")
        with open(train_csv, "w") as f:
            f.write("# Metadata: " + json.dumps(metadata) + "\n")
        train_df.to_csv(train_csv, mode="a", index=False)
        with open(val_csv, "w") as f:
            f.write("# Metadata: " + json.dumps(metadata) + "\n")
        val_df.to_csv(val_csv, mode="a", index=False)
        logging.info("Dataset split: %s (training) and %s (validation)", train_csv, val_csv)

    # Optional visualization.
    if args.plot:
        # Select a subset of triplets to plot.
        subset = df.sample(n=min(args.num_plots, len(df)), random_state=42)
        # For each triplet in the subset, create a figure with three subplots.
        for _, row in subset.iterrows():
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            # Plot the anchor structure on the first subplot.
            # We assume that plot_rna_structure can accept an Axes object via the 'ax' parameter.
            plot_rna_structure(row["anchor_seq"], row["anchor_structure"], ax=axs[0])
            axs[0].set_title("Anchor")
            # Plot the positive (modified) structure.
            plot_rna_structure(row["positive_seq"], row["positive_structure"], ax=axs[1])
            axs[1].set_title("Positive")
            # Plot the negative (shuffled and refolded) structure.
            plot_rna_structure(row["negative_seq"], row["negative_structure"], ax=axs[2])
            axs[2].set_title("Negative")
            # Tight layout to avoid overlap.
            fig.tight_layout()
            # Save the figure in the same plot directory.
            triplet_filename = os.path.join(plot_dir, f"triplet_{row['triplet_id']}.png")
            fig.savefig(triplet_filename)
            plt.close(fig)
            logging.debug("Saved triplet plot to %s", triplet_filename)

    logging.info("Data generation complete. Output saved in: %s", args.output_dir)


if __name__ == "__main__":
    main()
