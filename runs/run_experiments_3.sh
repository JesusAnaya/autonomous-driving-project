LEARNING_RATE=1e-4 WEIGHT_DECAY=0 OPTIMIZER=Adam BATCH_SIZE=64 SCHEDULER_TYPE=multistep python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_001

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-2 OPTIMIZER=Adam BATCH_SIZE=64 SCHEDULER_TYPE=multistep python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_002

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-3 OPTIMIZER=Adam BATCH_SIZE=64 SCHEDULER_TYPE=multistep python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_003

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=Adam BATCH_SIZE=64 SCHEDULER_TYPE=multistep python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_004
