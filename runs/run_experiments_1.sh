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
#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-5 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_005

# Run experiment 6
#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=SGD python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_006

# Run experiment 7
# LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-5 OPTIMIZER=SGD python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_007

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=AdamW python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_008

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 SCHEDULER_STEP_SIZE=35 BATCH_SIZE=64 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_009

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 BATCH_SIZE=32 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_010

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 BATCH_SIZE=64 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_011

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 BATCH_SIZE=128 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_012

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-5 BATCH_SIZE=128 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_013

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-3 OPTIMIZER=AdamW BATCH_SIZE=64 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_014

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=AdamW BATCH_SIZE=64 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_015

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=AdamW BATCH_SIZE=128 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_016

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-5 OPTIMIZER=AdamW BATCH_SIZE=128 python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_017

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-2 OPTIMIZER=AdamW BATCH_SIZE=64 SCHEDULER_TYPE=nonscheduler python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_018

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-3 OPTIMIZER=AdamW BATCH_SIZE=64 SCHEDULER_TYPE=nonscheduler python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_019

#LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=AdamW BATCH_SIZE=64 SCHEDULER_TYPE=nonscheduler python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_020

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-3 OPTIMIZER=Adam BATCH_SIZE=64 SCHEDULER_TYPE=nonscheduler python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_021

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 OPTIMIZER=Adam BATCH_SIZE=64 SCHEDULER_TYPE=nonscheduler python cross_validation.py --epochs_count 60 --dataset_type udacity_sim_1 --tensorboard_run_name experiment_022

# Add more commands as needed...