
# LEARNING_RATE=1e-3 WEIGHT_DECAY=1e-3 python cross_validation.py --epochs_count 60 --tensorboard_run_name experiment_001

# LEARNING_RATE=1e-3 WEIGHT_DECAY=1e-4 python cross_validation.py --epochs_count 60 --tensorboard_run_name experiment_002

LEARNING_RATE=1e-3 WEIGHT_DECAY=1e-5 python cross_validation.py --epochs_count 60 --tensorboard_run_name experiment_003

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-3 python cross_validation.py --epochs_count 60 --tensorboard_run_name experiment_004

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-4 python cross_validation.py --epochs_count 60 --tensorboard_run_name experiment_005

LEARNING_RATE=1e-4 WEIGHT_DECAY=1e-5 python cross_validation.py --epochs_count 60 --tensorboard_run_name experiment_006
