
###### Linear Supervision for Nonlinear, High-Dimensional Neural Control and Differential Games
###### Sharpless (wsharpless@ucsd.edu), Feng, Bansal, Herbert - 2024, L4DC 
###### Benchmark Script

## Init

readonly base_args="--dynamics_class LessLinearND --N 10 --goalR 0.25 --solve_grad"
readonly lin_sys_args="--gamma 0 --mu 0 --alpha 0"
readonly nlin_sys_args="--gamma 20 --mu 0 --alpha 0"

readonly use_wandb="--use_wandb"
# readonly use_wandb="" # if you dont want wandb
readonly wandb_project="test"

## Make Linear Model for Semi-Supervision (hopf or vanilla deepreach)

# via DeepReach
python run_experiment.py $base_args $lin_sys_args --num_epochs 30000 --baseline --experiment_name L10D_DR_linear --experiment_name L10D_DR $use_wandb --wandb_project $wandb_project --wandb_name L10D_DR --counter_end 20000

# from Hopf values (requires julia)
python run_experiment.py $base_args $lin_sys_args --num_epochs 30000 --hopf_loss lin_val_diff --solve_hopf --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name L10D_hopf_linear_dynamic $use_wandb --wandb_project $wandb_project --wandb_name L10D_hopf_dyanmic

# from bank of Hopf values
python run_experiment.py $base_args $lin_sys_args --just_make_hopf_bank --hopf_loss lin_val_diff --solve_hopf # make bank
python run_experiment.py $base_args $lin_sys_args --num_epochs 30000 --hopf_loss lin_val_diff --use_bank --bank_name Bank_Hopf_mp_10D_2Mpts_r25e-2_g0m0a0.npy --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name L10D_hopfb_linear_bank $use_wandb --wandb_project $wandb_project --wandb_name L10D_hopf_bank

## Make Nonlinear Model 

readonly supervisor="--load_hopf_model --load_hopf_model_name ./runs/L10D_hopfb_linear_bank" # can use any of the above for supervisor
readonly hopf_args="--hopf_loss lin_val_grad_diff $supervisor"

# baseline (vanilla DeepReach)
python run_experiment.py $base_args $nlin_sys_args --baseline --num_epochs 100000 --experiment_name LL10D_DR $use_wandb --wandb_project $wandb_project --wandb_name LL10D_DR

# Linear Supervision w/ Decay
python run_experiment.py $base_args $nlin_sys_args $hopf_args --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 0.75 --diff_con_loss_incr --num_epochs 10000 --experiment_name LL10D_LS_decay $use_wandb --wandb_project $wandb_project --wandb_name LL10D_LS_decay

# Linear Supervision w/ Augmentation
python run_experiment.py --dynamics_class LessLinearNDlambda --N 10 --goalR $goalR $nlin_sys_args $hopf_args --zerolambda_LS --LS_w_time_curr --hopf_pretrain_iters 1 --hopf_loss_divisor 50 --hopf_grad_loss_divisor 250 --num_epochs 100000 --experiment_name LL10D_LS_aug $use_wandb --wandb_project $wandb_project --wandb_name LL10D_LS_aug

# capacity test (supervision with ground truth, SLOW)
python run_experiment.py $base_args $nlin_sys_args --capacity_test --num_epochs 100000 --experiment_name LL10D_capacity $use_wandb --wandb_project $wandb_project --wandb_name LL10D_capacity