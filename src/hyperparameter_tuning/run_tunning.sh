cd '/home/nicolas/programs/GINFINITY/'
python -m src.hyperparameter_tuning.tune \
    --input_path '/home/nicolas/programs/GINFINITY/data/train/train.csv' \
    --batch_size 500 \
    --num_epochs 2 \
    --patience 4 \
    --num_workers 8 \
    --device cuda \
    --save_best_weights True \
    --pooling_type global_add_pool \
    --n_trials 2
    # --storage sqlite:////home/nicolas/programs/GINFINITY/src/hyperparameter_tuning/db/optuna_studies.db \
    # --output_dir '/home/nicolas/programs/GINFINITY/output'