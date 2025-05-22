#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
import pandas as pd
import optuna
import json
import os
import optuna.visualization as vis

def run_aggregated(input_tsv, percentile, alpha1, beta1, alpha2, beta2, gamma, num_workers):
    """
    Run the aggregated_score.py script with the given hyperparameters,
    return its output as a DataFrame.
    """
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as tmp:
        tmp_out = tmp.name
    cmd = [
        'python3', 'aggregated_score.py',
        '--input', input_tsv,
        '--percentile', str(percentile),
        '--alpha1', str(alpha1),
        '--beta1', str(beta1),
        '--alpha2', str(alpha2),
        '--beta2', str(beta2),
        '--gamma', str(gamma),
        '--num-workers', str(num_workers),
        '--output', tmp_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    df = pd.read_csv(tmp_out, sep='\t')
    os.remove(tmp_out)
    return df

def objective(trial, args):
    # Dynamic columns
    id1_col = f"{args.id_column}_1"
    id2_col = f"{args.id_column}_2"
    true1, true2 = args.true_pair

    # Fixed percentile & alpha/betas
    percentile = args.percentile
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    beta1  = args.beta1

    # Suggest hyperparameters
    beta2  = trial.suggest_float('beta2', 1.0, 2.0)
    gamma  = trial.suggest_float('gamma', 0.3, 0.7)

    # Compute scores
    df = run_aggregated(
        args.input, percentile,
        alpha1, beta1,
        alpha2, beta2,
        gamma, args.num_workers
    )

    # Find the true-pair score & its rank
    mask_true = (
        ((df[id1_col] == true1) & (df[id2_col] == true2)) |
        ((df[id1_col] == true2) & (df[id2_col] == true1))
    )
    if not mask_true.any():
        return -1e9
    true_score = df.loc[mask_true, 'score'].iloc[0]
    true_rank = int(df.index[mask_true][0])  # 0-based

    # Average of top-3 others
    others = df.loc[~mask_true, 'score']
    avg_top3 = others.sort_values(ascending=False).head(3).mean() if len(others) else true_score

    # Relative margin vs top-3
    margin = (true_score - avg_top3) / avg_top3

    # Mark whether this trial got rank1
    is_rank1 = (true_rank == 0)
    trial.set_user_attr('true_rank1', is_rank1)

    # If a previous rank1 exists, only allow new rank1 trials
    try:
        prev_rank1 = trial.study.best_trial.user_attrs.get('true_rank1', False)
    except Exception:
        prev_rank1 = False

    if prev_rank1 and not is_rank1:
        return -1e9

    return margin

def main():
    parser = argparse.ArgumentParser(
        description='Optimize hyperparameters for pairwise score'
    )
    parser.add_argument('--input',       required=True,
                        help='Path to sorted distances.tsv')
    parser.add_argument('--percentile',  type=float, required=True,
                        help='Fixed percentile to use (e.g. 0.1)')
    parser.add_argument('--alpha1',     type=float, default=1.0,
                        help='Numerator rank-decay exponent (default 1.0)')
    parser.add_argument('--beta1',      type=float, default=0.1,
                        help='Numerator off-diagonal decay (default 0.1)')
    parser.add_argument('--alpha2',     type=float, default=1.0,
                        help='Denominator rank-decay exponent (default 1.0)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Workers for aggregated_score')
    parser.add_argument('--trials',      type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--storage',     type=str, default='sqlite:///optuna_study.db',
                        help='Optuna storage URL')
    parser.add_argument('--study-name',  type=str, default='pair_opt',
                        help='Name of the Optuna study')
    parser.add_argument('--output',      default='best_params.json',
                        help='File to write best parameters JSON')
    parser.add_argument('--id-column',   type=str, default='exon_id',
                        help='Base name of the ID column (without _1/_2).')
    parser.add_argument('--true-pair',   nargs=2, metavar=('ID1','ID2'), required=True,
                        help='The two IDs (in either order) that should rank highest.')

    args = parser.parse_args()

    # Create or load study for resuming
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        load_if_exists=True
    )

    # Optimize
    try:
        study.optimize(lambda t: objective(t, args), n_trials=args.trials)
    except KeyboardInterrupt:
        print("Optimization interrupted; using partial results.")

    # Print and save best params
    print('Best parameters:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    with open(args.output, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f'Saved best parameters to {args.output}')

    # Generate diagnostic plots
    os.makedirs('optuna_plots', exist_ok=True)

    # Optimization history
    fig1 = vis.plot_optimization_history(study)
    trace = fig1.data[0]
    filtered = [(x, y) for x, y in zip(trace.x, trace.y) if x != 0]
    trace.x, trace.y = zip(*filtered)
    fig1.write_image('optuna_plots/optimization_history.png')
    fig1.write_html('optuna_plots/optimization_history.html')

    # Parameter importances
    fig2 = vis.plot_param_importances(study)
    fig2.write_image('optuna_plots/param_importances.png')
    fig2.write_html('optuna_plots/param_importances.html')

    # Slice plot
    fig3 = vis.plot_slice(study)
    fig3.write_image('optuna_plots/slice_plot.png')
    fig3.write_html('optuna_plots/slice_plot.html')

    print('Generated diagnostic plots in optuna_plots/')

if __name__ == '__main__':
    main()
