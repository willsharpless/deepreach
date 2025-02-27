

# LessLinearND 7D
# export CUDA_VISIBLE_DEVICES=1

readonly wandb_project="deepreach_hopf_retest"
# readonly experiment_name="bench_holder" ## must overwrite to save data, will save model_final?

readonly goalR="0.25"
readonly num_epochs="50000"
readonly base_args="--dynamics_class LessLinearND --N 10 --gamma 0 --mu 0 --alpha 0 --goalR $goalR --num_epochs $num_epochs"

## Make Linear Model for Semi-Supervision (hopf or vanilla deepreach)

### via DeepReach
python run_experiment.py $base_args --baseline --experiment_name RETEST_L10D_DR_linear --experiment_name RETEST_L10D_DR --use_wandb --wandb_project $wandb_project --wandb_name L10D_DR

### from Hopf values (requires julia)
python run_experiment.py $base_args --hopf_loss lin_val_diff --solve_hopf --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name RETEST_L10D_hopf_linear_dynamic --use_wandb --wandb_project $wandb_project --wandb_name L10D_hopf_dyanmic

### from bank of Hopf values
python run_experiment.py $base_args --hopf_loss lin_val_diff --solve_hopf --just_make_hopf_bank # make bank
python run_experiment.py $base_args --hopf_loss lin_val_diff --use_bank --bank_name Bank_Hopf_mp_10D_2Mpts_r25e-2_g0m0a0.npy --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name RETEST_L10D_hopfb_linear_bank --use_wandb --wandb_project $wandb_project --wandb_name L10D_hopf_bank

## Make Nonlinear Model (can use any of the above as supervisor)

### w/ decay
# python run_experiment.py --dynamics_class LessLinearND --N 7 --gamma -20 --mu 20 --alpha 1 --num_epochs 100000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 0.75 --load_hopf_model --load_hopf_model_name LL7D_hopfb_linear --diff_con_loss_incr --experiment_name LL7D_linsuper_lindecay

### w/ NL curriculum
# python run_experiment.py --dynamics_class LessLinearND --N 7 --gamma -20 --mu 20 --alpha 1 --num_epochs 200000 --nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000 --load_hopf_model --load_hopf_model_name LL7D_hopfb_linear --experiment_name LL7D_linsuper_nlcurr

### baseline
# python run_experiment.py --dynamics_class LessLinearND --N 7 --gamma -20 --mu 20 --alpha 1 --baseline --num_epochs 100000 --experiment_name LL7D_DR

### capacity test 
# python run_experiment.py --dynamics_class LessLinearND --N 7 --gamma -20 --mu 20 --alpha 1 --capacity_test --num_epochs 100000 --experiment_name LL7D_capacity