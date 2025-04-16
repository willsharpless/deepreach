#!/bin/bash

## Init

base_params="--tMax 2."
readonly convr_dyn="--dynamics_class ConveyorND $base_params"
readonly canoe_dyn="--dynamics_class CanoeND $base_params"
readonly convr_lam_dyn="--dynamics_class ConveyorNDlambda $base_params"
readonly canoe_lam_dyn="--dynamics_class CanoeNDlambda $base_params"

# readonly use_wandb=""
readonly use_wandb="--use_wandb"
readonly wandb_project="mulob_BRAT_ND_comparison"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/BRAT_comparison_ND/"
# readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/comparison_ND"

## Conveyor

i_max="7"
dims=()
for ((i=1; i<=${i_max}; i++)); do
    dims+=($((2 ** i)))
done

for i in $(seq 1 7); do

    if [ $i == 1 ]; then
        j="1"
        learn_params="--lr_std 2e-5 --pretrain_iters 500  --num_epochs 5000   --counter_end 500"
    elif [ $i == 2 ]; then
        j="2"
        learn_params="--lr_std 5e-6 --pretrain_iters 1000 --num_epochs 50000  --counter_end 1000"
    elif [ $i == 3 ]; then
        j="3"
        learn_params="--lr_std 2e-6 --pretrain_iters 2000 --num_epochs 100000 --counter_end 5000"
    else
        j="4"
        learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 200000 --counter_end 10000" 
    fi
    # elif [ $i > 6 ]; then
    #     j="5"
    #     learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 300000 --counter_end 20000"  
    # fi

    dim="--N ${dims[i-1]}"
    
    ## Conveyor
    python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_${dims[i-1]}D_p$j --wandb_name BRAT_${dims[i-1]}D_p$j $wandb_parms

    nc_loss_divisor="0.5"

    # Using presolved reach fn for naive-combo supervisor
    load_model_1="--load_decomposed_model_name_1 ./value_fns/Conveyor/models/Conveyor2D/reach_only_${dims[i-1]}D"
    load_params="--load_decomposed_models $load_model_1 $load_model_1" # model 2 uneeded in BRAT

    python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo supervision' $load_params --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_ncs_$nc_loss_divisor --wandb_name BRAT_lam_${dims[i-1]}D_ncs_$nc_loss_divisor $wandb_parms
    python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --decay --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo supervision' $load_params --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_ncs_decay_$nc_loss_divisor --wandb_name BRAT_lam_${dims[i-1]}D_ncs_decay_$nc_loss_divisor $wandb_parms
    python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --nc_lower_bound --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo supervision' $load_params --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_ncs_lb_$nc_loss_divisor --wandb_name BRAT_lam_${dims[i-1]}D_ncs_lb_$nc_loss_divisor $wandb_parms

    # Using self lambda-slice for naive-combo supervisor
    python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_ncss_$nc_loss_divisor --wandb_name BRAT_lam_${dims[i-1]}D_ncss_$nc_loss_divisor $wandb_parms
    python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --decay --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_ncss_decay_$nc_loss_divisor --wandb_name BRAT_lam_${dims[i-1]}D_ncss_decay_$nc_loss_divisor $wandb_parms
    python run_experiment.py $convr_lam_dyn $dim --mulob_type BRAT --nc_lower_bound --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_lam_${dims[i-1]}D_ncss_lb_$nc_loss_divisor --wandb_name BRAT_lam_${dims[i-1]}D_ncss_lb_$nc_loss_divisor $wandb_parms

done