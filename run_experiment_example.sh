

# LessLinearND 7D


## Make Linear Model for Semi-Supervision (hopf or vanilla deepreach)

### from Hopf values (requires julia)
python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma 0 --mu 0 --alpha 0 --solve_hopf --num_epochs 50000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name LL7D_hopf_linear

### from bank of Hopf values
python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma 0 --mu 0 --alpha 0 --use_bank --bank_name Bank_Hopf_mp_refined_ts1e-2_nograd_7D_4Mpts_r15e-2_g0m0a0.npy --num_epochs 50000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name LL7D_hopfb_linear

### via DeepReach
python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma 0 --mu 0 --alpha 0 --numepochs 50000 --experiment_name LL7D_DR_linear


## Make Nonlinear Model (can use any of the above as supervisor)

### w/ decay
python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma -20 --mu 20 --alpha 1 --num_epochs 200000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 1.5 --load_hopf_model --load_hopf_model_name LL7D_hopfb_linear --diff_con_loss_incr --experiment_name LL7D_linsuper_lindecay

### w/ NL curriculum
python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma -20 --mu 20 --alpha 1 --num_epochs 200000 --nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000 --load_hopf_model --load_hopf_model_name LL7D_hopfb_linear --experiment_name LL7D_linsuper_nlcurr

### baseline
python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma -20 --mu 20 --alpha 1 --baseline --num_epochs 200000 --experiment_name LL7D_DR

### capacity test 
python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma -20 --mu 20 --alpha 1 --capacity_test --num_epochs 200000 --experiment_name LL7D_capacity


# Quadrotor

## Make Linear Model for Semi-Supervision (via DeepReach)

python run_experiment.py --dynamics_class QuadrotorLinear --pretrain --pretrain_iters 1000 --num_epochs 101000 --counter_end 100000 --num_nl 512  --collisionR 0.5 --collective_thrust_max 20  --set_mode avoid --experiment_name quadrotor_linear

## Make Nonlinear Model (w/ decay)

python run_experiment.py --dynamics_class Quadrotor --pretrain --pretrain_iters 1000 --num_epochs 111000 --counter_end 100000 --num_nl 512  --collisionR 0.5 --collective_thrust_max 20  --set_mode avoid --hopf_pretrain_iters 10000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 1.5 --load_hopf_model --load_hopf_model_name quadrotor_linsuper
