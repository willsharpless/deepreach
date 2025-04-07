
## Init

readonly base_params="--N 2 --tMax 2."
readonly convr_dyn="--dynamics_class ConveyorND $base_params"
readonly canoe_dyn="--dynamics_class CanoeND $base_params"
readonly convr_lam_dyn="--dynamics_class ConveyorNDlambda $base_params"
readonly canoe_lam_dyn="--dynamics_class CanoeNDlambda $base_params"

readonly use_wandb="--use_wandb"
# readonly use_wandb=""
readonly wandb_project="mulob_decomposed_models"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_test="--experiments_dir ./runs/mulob/test"
readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/Conveyor2D/"
readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/Canoe2D"

# readonly gt_args="--num_epochs 10000 --counter_end 20000" #TODO: gt supervision
readonly learn_params_fast="--num_epochs 5000 --counter_end 2000" # fast
readonly learn_params="--num_epochs 20000 --counter_end 2000" # mid
readonly learn_params_slow="--num_epochs 100000 --counter_end 20000" # mid
readonly learn_params_long="--num_epochs 300000 --counter_end 40000" # long

## Conveyor

python run_experiment.py $convr_dyn --reach_only True $learn_params $exp_dir_convr --experiment_name reach_only_2D --wandb_name reach_only $wandb_parms

# python run_experiment.py $convr_dyn --avoid_only True $learn_params_fast $exp_dir_convr --experiment_name avoid_only_2D_fast --wandb_name avoid_only $wandb_parms

# python run_experiment.py $convr_dyn --reach_only True $learn_params $exp_dir_convr --experiment_name reach_only_2D --wandb_name reach_only $wandb_parms

# python run_experiment.py $convr_dyn --avoid_only True $learn_params $exp_dir_convr --experiment_name avoid_only_2D --wandb_name avoid_only $wandb_parms

# python run_experiment.py $convr_dyn --deepreach_model vanilla --avoid_only True --avoid_type axes $learn_params $exp_dir_convr --experiment_name avoid_only_2D_axes_vanilla #--wandb_name avoid_only $wandb_parms

# python run_experiment.py $convr_dyn --deepreach_model vanilla --avoid_only True --avoid_type axes $learn_params_slow $exp_dir_convr --experiment_name avoid_only_2D_axes_slow_vanilla --wandb_name avoid_only_vanilla $wandb_parms

# python run_experiment.py $convr_dyn --deepreach_model vanilla --avoid_only True --avoid_type axes $learn_params_long $exp_dir_convr --experiment_name avoid_only_2D_axes_long_vanilla --wandb_name avoid_only_vanilla $wandb_parms

# python run_experiment.py $convr_dyn --deepreach_model exact --avoid_only True --avoid_type axes $learn_params_long $exp_dir_convr --experiment_name avoid_only_2D_axes_long_exact --wandb_name avoid_only_exact $wandb_parms

# python run_experiment.py $convr_dyn --old_brat True --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAT_slow_exact --wandb_name BRAT_slow_exact $wandb_parms

# python run_experiment.py $convr_dyn --mulob_type BRAT --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAT_mulob_slow_exact --wandb_name BRAT_mulob_slow_exact $wandb_parms

# python run_experiment.py $convr_dyn --mulob_type BRAAT --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAAT_slow_exact --wandb_name BRAAT_slow_exact $wandb_parms

# python run_experiment.py $convr_lam_dyn --mulob_type BRAAT --mulob_loss_type augment --lam_slice_supervision --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAAT_slow_exact_lam_lss --wandb_name BRAAT_slow_exact_lam_lss $wandb_parms

# python run_experiment.py $convr_lam_dyn --mulob_type BRAAT --mulob_loss_type augment --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAAT_slow_exact_lam --wandb_name BRAAT_slow_exact_lam $wandb_parms

## Canoe

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_fast $exp_dir_canoe --experiment_name reach_1_only_2D_fast #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_2_only True $learn_params_fast $exp_dir_canoe --experiment_name reach_2_only_2D_fast #--wandb_name reach_2_only $wandb_parms


# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D_exact_1p5 --deepreach_model exact #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_slow $exp_dir_canoe --experiment_name reach_1_only_2D_slow_exact_1p5 --deepreach_model exact #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D_vanilla_1p5 --deepreach_model vanilla #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_slow $exp_dir_canoe --experiment_name reach_1_only_2D_slow_vanilla_1p5 --deepreach_model vanilla #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_2_only True $learn_params $exp_dir_canoe --experiment_name reach_2_only_2D #--wandb_name reach_2_only $wandb_parms