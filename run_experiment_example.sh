

# # LessLinearND 7D


# ## Make Linear Model for Semi-Supervision (hopf or vanilla deepreach)

# ### from Hopf values (requires julia)
# python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma 0 --mu 0 --alpha 0 --solve_hopf --num_epochs 50000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name LL7D_hopf_linear

# ### from bank of Hopf values
# python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma 0 --mu 0 --alpha 0 --use_bank --bank_name Bank_Hopf_mp_refined_ts1e-2_nograd_7D_4Mpts_r15e-2_g0m0a0.npy --num_epochs 50000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 20 --experiment_name LL7D_hopfb_linear

### via DeepReach
python run_experiment.py --dynamics_class LessLinearND --N 7 --gamma 0 --mu 0 --alpha 0 --num_epochs 50000 --experiment_name LL7D_DR_linear2 --baseline 


## Make Nonlinear Model (can use any of the above as supervisor)

### w/ decay
python run_experiment.py --dynamics_class LessLinearND --N 7 --gamma -20 --mu 20 --alpha 1 --num_epochs 100000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 0.75 --load_hopf_model --load_hopf_model_name LL7D_DR_linear2 --diff_con_loss_incr --experiment_name LL7D_linsuper_lindecay2 --gt_metrics --temporal_loss
# python run_experiment.py --dynamics_class LessLinearND --N 7 --gamma -20 --mu 20 --alpha 1 --num_epochs 200000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 1.5 --load_hopf_model --load_hopf_model_name LL7D_hopfb_linear --diff_con_loss_incr --experiment_name LL7D_linsuper_lindecay

# ### w/ NL curriculum
# python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma -20 --mu 20 --alpha 1 --num_epochs 200000 --nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000 --load_hopf_model --load_hopf_model_name LL7D_hopfb_linear --experiment_name LL7D_linsuper_nlcurr

# ### baseline
# python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma -20 --mu 20 --alpha 1 --baseline --num_epochs 200000 --experiment_name LL7D_DR

# ### capacity test 
# python run_experiment.py --dynamics_class LessLinearND -N 7 --gamma -20 --mu 20 --alpha 1 --capacity_test --num_epochs 200000 --experiment_name LL7D_capacity


# # Quadrotor

# ## Make Linear Model for Semi-Supervision (via DeepReach)

# python run_experiment.py --dynamics_class QuadrotorLinear --pretrain --pretrain_iters 1000 --num_epochs 101000 --counter_end 100000 --num_nl 512  --collisionR 0.5 --collective_thrust_max 20  --set_mode avoid --experiment_name quadrotor_linear
python run_experiment.py --dynamics_class QuadrotorLinear2 --pretrain --pretrain_iters 1000 --num_epochs 31000 --counter_end 30000 --num_nl 512  --collisionR 0.5 --collective_thrust_max 20  --set_mode avoid --experiment_name quadrotor_linear2 --baseline --N 13 --wandb_project hopf --wandb_group Quadrotor --wandb_name quadrotor_linear2 --wandb_entity zeyuanfe --use_wandb 
# ## Make Nonlinear Model (w/ decay)

python run_experiment.py --dynamics_class Quadrotor --pretrain --pretrain_iters 1000 --num_epochs 111000 --counter_end 100000 --num_nl 512  --collisionR 0.5 --collective_thrust_max 20  --set_mode avoid --experiment_name quadrotor_lindecay_newbc --hopf_pretrain_iters 10000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 1.5 --load_hopf_model --load_hopf_model_name quadrotor_linear2 --wandb_group Quadrotor --wandb_project hopf --wandb_name quadrotor_lindecay_newbc --wandb_entity zeyuanfe --use_wandb --N 13 --diff_con_loss_incr

python run_experiment.py --dynamics_class Quadrotor --pretrain --pretrain_iters 1000 --num_epochs 111000 --counter_end 100000 --num_nl 512  --collisionR 0.5 --collective_thrust_max 20  --set_mode avoid --experiment_name quadrotor_exponential_newbc2 --hopf_pretrain_iters 10000 --hopf_loss_decay --hopf_loss_decay_type exponential --hopf_loss_decay_w 1.5 --load_hopf_model --load_hopf_model_name quadrotor_linear --wandb_project hopf --wandb_group Quadrotor --wandb_name quadrotor_exponential_newbc2 --wandb_entity zeyuanfe --use_wandb --N 13 --diff_con_loss_incr

#baseline
python run_experiment.py --dynamics_class Quadrotor --pretrain --pretrain_iters 1000 --num_epochs 101000 --counter_end 100000 --num_nl 512  --collisionR 0.5 --collective_thrust_max 20  --set_mode avoid --experiment_name quadrotor_baseline_newbc --baseline --N 13 --wandb_project hopf --wandb_group Quadrotor --wandb_name quadrotor_baseline_newbc --wandb_entity zeyuanfe --use_wandb

# Quadrotor10D
python run_experiment.py --dynamics_class Quadrotor10D --pretrain --pretrain_iters 1000 --num_epochs 101000 --counter_end 100000 --num_nl 512  --collisionR 0.5  --set_mode avoid --experiment_name quadrotor10D_baseline2 --baseline --N 10 --wandb_project hopf --wandb_group Quadrotor10D --wandb_name quadrotor10D_baseline2 --wandb_entity zeyuanfe --use_wandb

# Quadrotor10D hopf lindecay
python run_experiment.py --dynamics_class Quadrotor10D --num_epochs 101000 --counter_end 100000 --num_nl 512  --collisionR 0.5  --set_mode avoid --experiment_name quadrotor10D_lindecay --hopf_pretrain_iters 10000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 1.5 --load_hopf_model --load_hopf_model_name quadrotor10D_linear --wandb_group Quadrotor10D --wandb_project hopf --wandb_name quadrotor10D_lindecay --wandb_entity zeyuanfe --use_wandb --N 10 --diff_con_loss_incr