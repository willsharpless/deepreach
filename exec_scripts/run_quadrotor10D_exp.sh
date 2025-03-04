# # Quadrotor
# baseline
python run_experiment.py --dynamics_class Quadrotor10D --pretrain --pretrain_iters 1000 --num_epochs 101000 --counter_end 100000 --num_nl 512  --collisionR 0.5  --set_mode avoid --experiment_name quadrotor10D_baseline_nrange --baseline --N 10 --wandb_project hopf --wandb_group Quadrotor10D --wandb_name quadrotor10D_baseline_nrange --wandb_entity zeyuanfe --use_wandb --epochs_til_ckpt 100000 --tMax 1.5 --numpoints 10000

# Quadrotor10DLinear
python run_experiment.py --dynamics_class Quadrotor10D --pretrain --pretrain_iters 1000 --num_epochs 61000 --counter_end 60000 --num_nl 512  --collisionR 0.5  --set_mode avoid --experiment_name quadrotor10D_linear --baseline --N 10 --wandb_project hopf --wandb_group Quadrotor10D --wandb_name quadrotor10D_linear --wandb_entity zeyuanfe --use_wandb --epochs_til_ckpt 1000 --tMax 1.5 --numpoints 10000 --bank_name quad10lin --experiment_class DeepReach --linear True


# lindecay rel grads
python run_experiment.py --dynamics_class Quadrotor10D --num_epochs 112000 --counter_end 100000 --num_nl 512 --collisionR 0.5 --set_mode avoid --experiment_name quadrotor10D_lindecay --hopf_pretrain_iters 10000 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 1.0 --load_hopf_model --load_hopf_model_name runs/quadrotor10D_linear --wandb_group Quadrotor10D --wandb_project hopf --wandb_name quadrotor10D_lindecay --wandb_entity zeyuanfe --use_wandb --N 10 --hopf_loss lin_val_grad_diff --numpoints 10000 --hopf_loss_divisor 2.5 --hopf_grad_loss_divisor 2.5 --adj_rel_grads True --tMax 1.5 --linear False --bank_name test --experiment_class DeepReach --diff_con_loss_incr

# Lambda
python run_experiment.py --wandb_group Quadrotor10Dlambda --wandb_project hopf --wandb_name quadrotor10D_lambda --wandb_entity zeyuanfe --use_wandb --dynamics_class Quadrotor10Dlambda --experiment_name quadrotor10D_lambda --N 10 --set_mode avoid --hopf_loss lin_val_grad_diff --load_hopf_model --load_hopf_model_name runs/quadrotor10D_lindecay --zerolambda_LS --LS_w_time_curr --collisionR 0.5 --numpoints 10000 --num_epochs 66000 --counter_end 66000 --hopf_pretrain_iters 10000 --hopf_loss_divisor 2.5 --hopf_grad_loss_divisor 2.5 --adj_rel_grads True --tMax 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 1.0 --linear 1 --bank_name test --experiment_class DeepReach

python run_experiment.py --mode test --checkpoint_toload -1 --data_step run_basic_recovery --experiment_name quadrotor10D_lindecay