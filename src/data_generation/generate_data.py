#!/usr/bin/env python
"""
generate_data.py

Main script for RNA triplet dataset generation.

This script parses command‐line arguments (including parameters for sequence generation, modifications,
and appending events), generates run metadata, calls the triplet‐generation pipeline in parallel
(with each worker generating a “thread” of triplets concurrently), saves the dataset (with sequential IDs),
optionally splits the dataset, and if requested, creates plots for a subset of triplets (each plotted
as one figure with three subplots for the anchor, positive, and negative structures).
"""

import argparse
import os
import json
import uuid
import datetime
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_generation_utils import (
    generate_triplet_thread,
    plot_rna_structure,
    split_dataset,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="RNA Structure Triplet Dataset Generator with forgi")
    # Sequence Generation Parameters
    parser.add_argument("--num_structures", type=int, default=100, help="Number of structures to generate")
    parser.add_argument("--seq_min_len", type=int, default=50, help="Minimum sequence length")
    parser.add_argument("--seq_max_len", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--seq_len_distribution", choices=["norm", "unif"], default="unif", help="Distribution of sequence lengths")
    parser.add_argument("--seq_len_mean", type=float, default=75, help="Mean sequence length (for normal distribution)")
    parser.add_argument("--seq_len_sd", type=float, default=10, help="Standard deviation of sequence length (for normal distribution)")
    parser.add_argument("--neg_len_variation", type=int, default=0, help="Maximum length variation for negative structures")
    
    # Stem Modifications
    parser.add_argument("--n_stem_indels", type=int, default=1, help="Number of stem modification cycles")
    parser.add_argument("--stem_min_size", type=int, default=2, help="Minimum stem size")
    parser.add_argument("--stem_max_size", type=int, default=10, help="Maximum stem size")
    parser.add_argument("--stem_max_n_modifications", type=int, default=1, help="Maximum modifications per stem")
    
    # Loop Modifications
    parser.add_argument("--n_hloop_indels", type=int, default=1, help="Number of hairpin loop modification cycles")
    parser.add_argument("--n_iloop_indels", type=int, default=1, help="Number of internal loop modification cycles")
    parser.add_argument("--n_bulge_indels", type=int, default=1, help="Number of bulge modification cycles")
    parser.add_argument("--n_mloop_indels", type=int, default=1, help="Number of multi loop modification cycles")
    
    # Loop Size Constraints
    parser.add_argument("--hloop_min_size", type=int, default=3, help="Minimum hairpin loop size")
    parser.add_argument("--hloop_max_size", type=int, default=10, help="Maximum hairpin loop size")
    parser.add_argument("--iloop_min_size", type=int, default=2, help="Minimum internal loop size")
    parser.add_argument("--iloop_max_size", type=int, default=10, help="Maximum internal loop size")
    parser.add_argument("--bulge_min_size", type=int, default=1, help="Minimum bulge loop size")
    parser.add_argument("--bulge_max_size", type=int, default=1, help="Maximum bulge loop size")
    parser.add_argument("--mloop_min_size", type=int, default=2, help="Minimum multi loop size")
    parser.add_argument("--mloop_max_size", type=int, default=15, help="Maximum multi loop size")
    
    # Loop Modification Limits
    parser.add_argument("--hloop_max_n_modifications", type=int, default=1, help="Maximum modifications per hairpin loop")
    parser.add_argument("--iloop_max_n_modifications", type=int, default=1, help="Maximum modifications per internal loop")
    parser.add_argument("--bulge_max_n_modifications", type=int, default=1, help="Maximum modifications per bulge loop")
    parser.add_argument("--mloop_max_n_modifications", type=int, default=1, help="Maximum modifications per multi loop")
    
    # Appending Parameters
    parser.add_argument("--appending_event_probability", type=float, default=0.3,
                        help="Probability that an appending event will occur for a triplet (default 0.3)")
    parser.add_argument("--both_sides_appending_probability", type=float, default=0.33,
                        help="Within appending events, probability to append on both sides (the remaining events are equally divided between left and right)")
    parser.add_argument("--linker_min", type=int, default=2,
                        help="Minimum linker length (in bases) for appending event")
    parser.add_argument("--linker_max", type=int, default=8,
                        help="Maximum linker length (in bases) for appending event")
    parser.add_argument("--appending_size_factor", type=float, default=1.0,
                        help="Factor to multiply the anchor length to obtain the mean for the normal distribution from which the appended RNA length is sampled. For example, if set to 0.5 and the anchor length is 100, the mean appended RNA length will be 50")
    
    # Performance
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output CSV, plots, and metadata")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of triplets to generate per thread (vectorized block)")
    
    # Visualization
    parser.add_argument("--plot", action="store_true", help="Generate structure plots")
    parser.add_argument("--num_plots", type=int, default=5, help="Number of structure triplets to plot")
    
    # Dataset Splitting
    parser.add_argument("--split", action="store_true", help="Enable dataset splitting")
    parser.add_argument("--train_fraction", type=float, help="Fraction of data for training")
    parser.add_argument("--val_fraction", type=float, help="Fraction of data for validation")
    
    # Debug option
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for detailed output")
    parser.add_argument("--timing-log", action="store_true", help="Enable detailed timing logs for structure modifications")
    
    return parser.parse_args()

def setup_logging(output_dir, debug_flag):
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    level = logging.DEBUG if debug_flag else logging.ERROR
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"debug_{timestamp}.log")
    fh = logging.FileHandler(log_filename)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.debug("Logging is configured. Debug output will be saved to %s", log_filename)

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot:
        plot_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
    setup_logging(args.output_dir, args.debug)
    logging.info("Starting RNA triplet generation with parameters: %s", vars(args))
    
    # Validate dataset splitting fractions:
    if args.split:
        if args.train_fraction is not None and args.val_fraction is not None:
            if abs(args.train_fraction + args.val_fraction - 1) > 1e-6:
                logging.error("train_fraction and val_fraction must sum to 1.")
                return
        elif args.train_fraction is not None:
            args.val_fraction = 1 - args.train_fraction
        elif args.val_fraction is not None:
            args.train_fraction = 1 - args.val_fraction
        else:
            args.train_fraction = 0.8
            args.val_fraction = 0.2

    metadata = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": vars(args)
    }
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info("Metadata saved to %s", metadata_file)
    
    total = args.num_structures
    batch_size = args.batch_size
    num_tasks = (total + batch_size - 1) // batch_size  # ceiling division
    task_sizes = [batch_size] * num_tasks
    if total % batch_size != 0:
        task_sizes[-1] = total % batch_size

    triplets = []
    from data_generation_utils import generate_triplet_thread  # Ensure the new function is imported
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                generate_triplet_thread,
                ts,
                args.seq_min_len, args.seq_max_len, args.seq_len_distribution, args.seq_len_mean, args.seq_len_sd,
                args.neg_len_variation,
                args.n_stem_indels, args.stem_min_size, args.stem_max_size, args.stem_max_n_modifications,
                args.n_hloop_indels, args.hloop_min_size, args.hloop_max_size, args.hloop_max_n_modifications,
                args.n_iloop_indels, args.iloop_min_size, args.iloop_max_size, args.iloop_max_n_modifications,
                args.n_bulge_indels, args.bulge_min_size, args.bulge_max_size, args.bulge_max_n_modifications,
                args.n_mloop_indels, args.mloop_min_size, args.mloop_max_size, args.mloop_max_n_modifications,
                args.appending_event_probability, args.both_sides_appending_probability,
                args.linker_min, args.linker_max, args.appending_size_factor
            )
            for ts in task_sizes
        ]
        pbar = tqdm(total=total, desc="Generating triplets")
        for future in as_completed(futures):
            try:
                thread_triplets = future.result()
                triplets.extend(thread_triplets)
                pbar.update(len(thread_triplets))
            except Exception:
                logging.exception("Error generating triplet thread:")
        pbar.close()

    triplets = triplets[:total]
    
    df = pd.DataFrame(triplets)
    df["triplet_id"] = list(range(len(df)))
    
    output_csv = os.path.join(args.output_dir, "rna_triplets.csv")
    with open(output_csv, "w") as f:
        f.write("# Metadata: " + json.dumps(metadata) + "\n")
    df.to_csv(output_csv, mode="a", index=False)
    logging.info("Full dataset saved to %s", output_csv)
    
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
    
    if args.plot:
        plot_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        dpi = 100  # adjust as needed
        for idx, row in df.sample(n=min(args.num_plots, len(df)), random_state=42).iterrows():
            L = len(row["anchor_seq"])
            subplot_size_px = 500 + 8 * max(0, L - 130)
            width_in = (3 * subplot_size_px) / dpi
            height_in = subplot_size_px / dpi
            logging.debug("Plotting Triplet %d: Anchor length=%d, subplot size=%d px, figure size=(%.1f in x %.1f in)",
                          idx, L, subplot_size_px, width_in, height_in)
            fig, axs = plt.subplots(1, 3, figsize=(width_in, height_in), dpi=dpi)
            plot_rna_structure(row["anchor_seq"], row["anchor_structure"], ax=axs[0])
            axs[0].set_title("Anchor")
            plot_rna_structure(row["positive_seq"], row["positive_structure"], ax=axs[1])
            axs[1].set_title("Positive")
            plot_rna_structure(row["negative_seq"], row["negative_structure"], ax=axs[2])
            axs[2].set_title("Negative")
            fig.suptitle(f"Triplet {idx}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_plot = os.path.join(plot_dir, f"triplet_{idx}.png")
            plt.savefig(output_plot)
            plt.close(fig)
            logging.debug("Saved triplet plot to %s", output_plot)
    
    logging.info("Data generation complete. Output saved in: %s", args.output_dir)

if __name__ == "__main__":
    main()
