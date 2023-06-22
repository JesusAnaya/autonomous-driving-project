#!/bin/bash

# Run experiment 1
#LEARNING_RATE=1e-3 WEIGHT_DECAY=1e-3 BATCH_SIZE=64 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_001

# Run experiment 2
#LEARNING_RATE=1e-3 WEIGHT_DECAY=1e-4 BATCH_SIZE=64 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_002

# Run experiment 3
#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 BATCH_SIZE=64 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_003

# Run experiment 4
#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_004

# Run experiment 5
LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-3 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_005

# Run experiment 6
LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=SGD python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_006

# Run experiment 7
LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-3 OPTIMIZER=SGD python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_007

# Add more commands as needed...