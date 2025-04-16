#!/bin/bash

## Init

base_params="--tMax 2."
readonly convr_dyn="--dynamics_class ConveyorND $base_params"
readonly canoe_dyn="--dynamics_class CanoeND $base_params"
readonly convr_lam_dyn="--dynamics_class ConveyorNDlambda $base_params"
readonly canoe_lam_dyn="--dynamics_class CanoeNDlambda $base_params"

# readonly use_wandb=""
readonly use_wandb="--use_wandb"
readonly wandb_project="mulob_naive_combo_testing"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_test="--experiments_dir ./runs/mulob/test"
readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/decomposedND/"
readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/decomposedND"

readonly load_model_1="--load_decomposed_model_name_1 ./runs/mulob/ConveyorND/Conveyor2D/reach_only_2D"
readonly load_model_2="--load_decomposed_model_name_2 ./runs/mulob/ConveyorND/Conveyor2D/avoid_only_2D_fast2"
readonly load_params_2d="--load_decomposed_models $load_model_1 $load_model_2"

## Conveyor

i_max="10"
dims=()
for ((i=1; i<=${i_max}; i++)); do
    dims+=($((2 ** i)))
done

stds=(
    "0.25"
    "0.5"
    "1."
    "2."
)

# for j in $(seq 1 1); do
    # for i in $(seq 1 10); do

# for i in $(seq 2 7); do
# for i in $(seq 1 6); do
# for i in $(seq 4 4); do
for i in $(seq 1 1); do
    for j in $(seq 1 1); do

        # learn_params="--lr_std 2e-5 --pretrain_iters 500 --num_epochs 50000 --counter_end 500" 
        # learn_params="--lr_std 1e-6 --pretrain_iters 500 --num_epochs 50000 --counter_end 500" 
        learn_params="--lr_std 2e-5 --pretrain_iters 500 --num_epochs 5000 --counter_end 500" 

        sampling_params="--spatial_sampling_type truncated_normal --spatial_sampling_std ${stds[j-1]}"

        dim="--N ${dims[i-1]}"
        
        ## Conveyor
        python run_experiment.py $convr_dyn $dim --old_brat True --deepreach_model exact $learn_params $sampling_params $exp_dir_convr --experiment_name BRAT_${dims[i-1]}D_TN_std${stds[i-1]} --wandb_name BRAT_${dims[i-1]}D_TN_std${stds[j-1]} $wandb_parms
        python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --deepreach_model exact $learn_params $sampling_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_TN_std${stds[i-1]} --wandb_name BRAT_lam_${dims[i-1]}D_TN_std${stds[j-1]} $wandb_parms
        
        # python run_experiment.py $convr_dyn $dim --mulob_type BRAT --mulob_loss_type 'vanilla' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_${dims[i-1]}D_vanilla_uniform --wandb_name BRAT_${dims[i-1]}D_vanilla_uniform $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --mulob_loss_type 'vanilla' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_vanilla_uniform --wandb_name BRAT_lam_${dims[i-1]}D_vanilla_uniform $wandb_parms
        
        # nc_weight="1000"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight $wandb_parms
        # nc_weight="10"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight $wandb_parms
        # nc_weight="5"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight $wandb_parms
        # nc_weight="1"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_$nc_weight $wandb_parms
 
        # nc_weight="1"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --nc_decay --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncss_decay --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_decay_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_TN_uniform_ncss_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_lb_$nc_weight $wandb_parms

        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_grow --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_grow_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --nc_decay --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_decay --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_decay_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb_$nc_weight $wandb_parms

        # nc_weight="1"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_TN_uniform_ncss_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_lb_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb_$nc_weight $wandb_parms

        # nc_weight="0.5"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_TN_uniform_ncss_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_lb_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb_$nc_weight $wandb_parms

        # nc_weight="0.1"
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo self-supervision' --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_TN_uniform_ncss_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncss_lb_$nc_weight $wandb_parms
        # python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_weight --mulob_loss_type 'naive-combo supervision' $load_params_2d --nc_lower_bound --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb --wandb_name BRAT_lam_${dims[i-1]}D_uniform_ncs_lb_$nc_weight $wandb_parms

        # python run_experiment.py $convr_dyn $dim --old_brat True --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_${dims[i-1]}D --wandb_name BRAT_${dims[i-1]}D $wandb_parms

        # python run_experiment.py $convr_dyn $dim --reach_only True $learn_params $exp_dir_convr --experiment_name reach_only_${dims[i-1]}D_p$j --wandb_name reach_only_${dims[i-1]}D_p${j} $wandb_parms
        # python run_experiment.py $convr_dyn $dim --avoid_only True $learn_params $exp_dir_convr --experiment_name avoid_only_${dims[i-1]}D_p$j --wandb_name avoid_only_${dims[i-1]}D_p${j} $wandb_parms

        ## Canoe
        # python run_experiment.py $canoe_dyn $dim --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_${dims[i-1]}D_p$j --wandb_name reach_1_only_${dims[i-1]}D_p${j} $wandb_parms
        # python run_experiment.py $canoe_dyn $dim --reach_2_only True $learn_params $exp_dir_canoe --experiment_name reach_2_only_${dims[i-1]}D_p$j --wandb_name reach_2_only_${dims[i-1]}D_p${j} $wandb_parms

    done
done