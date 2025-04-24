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
    Run the aggregated_metric.py script with the given hyperparameters,
    return its output as a DataFrame.
    """
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as tmp:
        tmp_out = tmp.name
    cmd = [
        'python3', 'aggregated_metric.py',
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
    # Fixed percentile
    percentile = args.percentile
    # Suggest hyperparameters
    alpha1 = trial.suggest_float('alpha1', 0.0, 0.5)
    alpha2 = trial.suggest_float('alpha2', 0.1, 1.0)
    beta1  = trial.suggest_float('beta1', 0.0001, 0.05, log=True)
    beta2  = trial.suggest_float('beta2', 1.0, 2.0)
    gamma  = trial.suggest_float('gamma', 0.3, 0.7)

    # Compute metrics
    df = run_aggregated(
        args.input, percentile,
        alpha1, beta1,
        alpha2, beta2,
        gamma, args.num_workers
    )

    # Extract true-positive score and its rank (0-based)
    mask_true = (
        ((df.exon_id_1 == 'ENSE00001655346.1') & (df.exon_id_2 == 'ENSE00004286647.1'))
        |
        ((df.exon_id_1 == 'ENSE00004286647.1') & (df.exon_id_2 == 'ENSE00001655346.1'))
    )
    if not mask_true.any():
        return -1e9
    true_score = df.loc[mask_true, 'metric'].iloc[0]
    true_rank = int(df.index[mask_true][0])  # 0-based

    # Avg of top-3 others
    others = df.loc[~mask_true, 'metric']
    if len(others) == 0:
        avg_top3 = true_score
    else:
        avg_top3 = others.sort_values(ascending=False).head(3).mean()

    # Relative margin vs top-3
    margin = (true_score - avg_top3) / avg_top3

    # Mark whether this trial got rank1
    is_rank1 = (true_rank == 0)
    trial.set_user_attr('true_rank1', is_rank1)

    # Now check existing best trial safely
    prev_rank1 = False
    try:
        best = trial.study.best_trial
        prev_rank1 = best.user_attrs.get('true_rank1', False)
    except ValueError:
        # no best_trial yet
        pass

    # If a previous rank1 exists, only allow new rank1 trials
    if prev_rank1 and not is_rank1:
        return -1e9

    return margin

def main():
    parser = argparse.ArgumentParser(
        description='Optimize hyperparameters for exon-pair metric')
    parser.add_argument('--input',       required=True,
                        help='Path to sorted distances.tsv')
    parser.add_argument('--percentile',  type=float, required=True,
                        help='Fixed percentile to use (e.g. 0.1)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Workers for aggregated_metric')
    parser.add_argument('--trials',      type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--storage',     type=str, default='sqlite:///optuna_study.db',
                        help='Optuna storage URL (e.g. sqlite:///db.sqlite)')
    parser.add_argument('--study-name',  type=str, default='exon_pair_opt',
                        help='Name of the Optuna study')
    parser.add_argument('--output',      default='best_params.json',
                        help='File to write best parameters JSON')
    args = parser.parse_args()

    # Create or load study for resuming
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        load_if_exists=True
    )

    # Run optimization (allow interruption)
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

    # Optimization history (remove first trial point)
    fig1 = vis.plot_optimization_history(study)
    trace = fig1.data[0]
    # drop the x==0 trial
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
