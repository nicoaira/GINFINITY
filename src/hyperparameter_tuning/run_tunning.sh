cd '/home/nicolas/programs/GINFINITY/'
python -m src.hyperparameter_tuning.tune \
    --input_path '/home/nicolas/programs/GINFINITY/data/train/train_medium.csv' \
    --batch_size 500 \
    --num_epochs 25 \
    --patience 5 \
    --num_workers 8 \
    --device cuda \
    --save_best_weights True \
    --pooling_type global_add_pool \
    --n_trials 8 \
    --benchmark_datasets hard_rfam_benchmark_big \
    --quiet_benchmark \
    --retries 3 \
    # --study_id tuning_250128_2043 \
    # --finish_now