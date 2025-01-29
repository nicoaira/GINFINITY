cd '/home/nicolas/programs/GINFINITY/'
python -m src.hyperparameter_tuning.tune \
    --input_path '/home/nicolas/programs/GINFINITY/data/train/train_dummy.csv' \
    --batch_size 500 \
    --num_epochs 2 \
    --patience 5 \
    --num_workers 8 \
    --device cuda \
    --save_best_weights True \
    --pooling_type global_add_pool \
    --n_trials 8 \
    --benchmark_datasets easy_rfam_benchmark_big \
    --quiet_benchmark \
    --retries 3 \
    # --study_id tuning_250128_2339 \
    # --finish_now