
## Init

readonly dim="--N 2"
readonly convr_dyn="--dynamics_class ConveyorND $dim --tMax 2." # --deepreach_model vanilla
readonly canoe_dyn="--dynamics_class CanoeND $dim --tMax 2."

# readonly use_wandb="--use_wandb"
readonly use_wandb=""
readonly wandb_project="mulob_decomposed_models"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_test="--experiments_dir ./runs/mulob/test"
readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/Conveyor2D/"
readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/Canoe2D"

# readonly gt_args="--num_epochs 10000 --counter_end 20000" #TODO: gt supervision
readonly learn_params_fast="--num_epochs 5000 --counter_end 2000" # fast
readonly learn_params="--num_epochs 30000 --counter_end 2000" # mid
readonly learn_params_slow="--num_epochs 100000 --counter_end 20000" # mid
# readonly learn_params="--num_epochs 200000 --counter_end 40000" # long

## Conveyor

# python run_experiment.py $convr_dyn --reach_only True $learn_params_fast $exp_dir_convr --experiment_name reach_only_2D_fast --wandb_name reach_only $wandb_parms

# python run_experiment.py $convr_dyn --avoid_only True $learn_params_fast $exp_dir_convr --experiment_name avoid_only_2D_fast --wandb_name avoid_only $wandb_parms

# python run_experiment.py $convr_dyn --reach_only True $learn_params $exp_dir_convr --experiment_name reach_only_2D --wandb_name reach_only $wandb_parms

# python run_experiment.py $convr_dyn --avoid_only True $learn_params $exp_dir_convr --experiment_name avoid_only_2D --wandb_name avoid_only $wandb_parms

python run_experiment.py $convr_dyn --avoid_only True --avoid_type axes $learn_params $exp_dir_convr --experiment_name avoid_only_2D_axes_vanilla #--wandb_name avoid_only $wandb_parms

python run_experiment.py $convr_dyn --avoid_only True --avoid_type axes $learn_params_slow $exp_dir_convr --experiment_name avoid_only_2D_axes_slow_vanilla #--wandb_name avoid_only $wandb_parms

## Canoe

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_fast $exp_dir_canoe --experiment_name reach_1_only_2D_fast #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_2_only True $learn_params_fast $exp_dir_canoe --experiment_name reach_2_only_2D_fast #--wandb_name reach_2_only $wandb_parms


# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D_exact_1p5 --deepreach_model exact #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_slow $exp_dir_canoe --experiment_name reach_1_only_2D_slow_exact_1p5 --deepreach_model exact #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D_vanilla_1p5 --deepreach_model vanilla #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_slow $exp_dir_canoe --experiment_name reach_1_only_2D_slow_vanilla_1p5 --deepreach_model vanilla #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_2_only True $learn_params $exp_dir_canoe --experiment_name reach_2_only_2D #--wandb_name reach_2_only $wandb_parms