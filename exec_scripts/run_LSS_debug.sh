
## Init

readonly wandb_project="deepreach_hopf_tunes"
readonly output_dir_name="./runs"
readonly group_run_name="debug"
# mkdir -p $output_dir_name

readonly goalR="0.25"
readonly num_epochs="100000"
readonly bs="10000"

readonly base_args="run_experiment.py --experiments_dir $output_dir_name/$group_run_name --use_wandb --wandb_project $wandb_project --dynamics_class LessLinearND --goalR $goalR"

readonly bench_mag="20"
readonly bench1="--gamma $bench_mag --mu 0 --alpha 0"
readonly bench2="--gamma -$bench_mag --mu 0 --alpha 0"
readonly bench3="--gamma -$bench_mag --mu -$bench_mag --alpha 1"
readonly bench4="--gamma $bench_mag --mu -$bench_mag --alpha 1"

## Method

readonly baseline_args="--N 50 --baseline" ## Baseline
readonly base_method_args="--hopf_loss lin_val_grad_diff --hopf_pretrain_iters 3000" ## Decayed LSS
# readonly method_args="--hopf_loss lin_val_diff --nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000 --load_hopf_model --load_hopf_model_name $supervisor" ## NL Curriculum Scale LSS
# readonly method_args="--capacity" ## Capacity

## LSS Testing

# method_args_1="--N 7 --load_hopf_model --load_hopf_model_name ./runs/LL7D_DR_linear_fast"
# method_args_2="--N 15 --load_hopf_model --load_hopf_model_name ./runs/LL15D_DR_linear_fast"
# method_args_1="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr"
# method_args_2="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_decay_w 5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr"
# method_args_3="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 100000 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr"
# method_args_4="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 100000 --hopf_loss_decay_w 5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr"
# method_args_3="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_decay_w 0.2"

## Divisor Variation?
# method_args_1="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 2 --hopf_grad_loss_divisor 25 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b3_more_hopf --wandb_name LSS_b3_more_hopf"
# method_args_2="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 10 --hopf_grad_loss_divisor 25 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b3_less_hopf --wandb_name LSS_b3_less_hopf"
# method_args_3="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 5 --hopf_grad_loss_divisor 12 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b3_more_grad --wandb_name LSS_b3_more_grad"
# method_args_4="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 5 --hopf_grad_loss_divisor 50 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b3_less_grad --wandb_name LSS_b3_less_grad"
# method_args_5="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 2 --hopf_grad_loss_divisor 12 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b3_more_both --wandb_name LSS_b3_more_both"
# method_args_6="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 10 --hopf_grad_loss_divisor 50 --hopf_loss_decay_w 1.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b3_less_both --wandb_name LSS_b3_less_both"

method_args_1="--lr_std 2e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 50000 --hopf_loss_divisor 2 --hopf_grad_loss_divisor 25 --hopf_loss_decay_w 0.25 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_more_hopf_lr6_dwp25 --wandb_name LSS_b4_more_hopf_lr6_dwp25"
method_args_2="--lr_std 2e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 50000 --hopf_loss_divisor 10 --hopf_grad_loss_divisor 25 --hopf_loss_decay_w 0.25 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_less_hopf_lr6_dwp25 --wandb_name LSS_b4_less_hopf_lr6_dwp25"
method_args_3="--lr_std 2e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 50000 --hopf_loss_divisor 5 --hopf_grad_loss_divisor 12 --hopf_loss_decay_w 0.25 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_more_grad_lr6_dwp25 --wandb_name LSS_b4_more_grad_lr6_dwp25"
method_args_4="--lr_std 2e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 50000 --hopf_loss_divisor 5 --hopf_grad_loss_divisor 50 --hopf_loss_decay_w 0.25 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_less_grad_lr6_dwp25 --wandb_name LSS_b4_less_grad_lr6_dwp25"
method_args_5="--lr_std 2e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 50000 --hopf_loss_divisor 2 --hopf_grad_loss_divisor 12 --hopf_loss_decay_w 0.25 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_more_both_lr6_dwp25 --wandb_name LSS_b4_more_both_lr6_dwp25"
method_args_6="--lr_std 2e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 50000 --hopf_loss_divisor 10 --hopf_grad_loss_divisor 50 --hopf_loss_decay_w 0.25 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_less_both_lr6_dwp25 --wandb_name LSS_b4_less_both_lr6_dwp25"

method_args_7="--lr_std 5e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 0.5 --hopf_grad_loss_divisor 25 --hopf_loss_decay_w 0.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_more_hopf_10x --wandb_name LSS_b4_more_hopf_10x"
method_args_8="--lr_std 5e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 50 --hopf_grad_loss_divisor 25 --hopf_loss_decay_w 0.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_less_hopf_10x --wandb_name LSS_b4_less_hopf_10x"
method_args_9="--lr_std 5e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 5 --hopf_grad_loss_divisor 2.5 --hopf_loss_decay_w 0.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_more_grad_10x --wandb_name LSS_b4_more_grad_10x"
method_args_10="--lr_std 5e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 5 --hopf_grad_loss_divisor 250 --hopf_loss_decay_w 0.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_less_grad_10x --wandb_name LSS_b4_less_grad_10x"
method_args_11="--lr_std 5e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 0.5 --hopf_grad_loss_divisor 2.5 --hopf_loss_decay_w 0.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_more_both_10x --wandb_name LSS_b4_more_both_10x"
method_args_12="--lr_std 5e-6 --N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000 --hopf_loss_divisor 50 --hopf_grad_loss_divisor 250 --hopf_loss_decay_w 0.5 --hopf_loss_decay --hopf_loss_decay_type linear --diff_con_loss_incr --experiment_name LSS_b4_less_both_10x --wandb_name LSS_b4_less_both_10x"

## No gradual pde?
# method_args_1="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000  --hopf_loss_decay --hopf_loss_decay_w 1.5 --hopf_loss_decay_type linear"
# method_args_2="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 20000  --hopf_loss_decay --hopf_loss_decay_w 5 --hopf_loss_decay_type linear"
# method_args_3="--N 50 --load_hopf_model --load_hopf_model_name ./runs/LL50D_DR_linear_r25 --solve_grad --numpoints 10000 --num_epochs 100000  --hopf_loss_decay --hopf_loss_decay_w 1.5 --hopf_loss_decay_type linear"

## LSS Testing

## Execute
methods=(
    "$method_args_1" 
    "$method_args_2" 
    "$method_args_3" 
    "$method_args_4" 
    "$method_args_5" 
    "$method_args_6" 
    "$method_args_7" 
    "$method_args_8" 
    "$method_args_9" 
    "$method_args_10" 
    "$method_args_11" 
    "$method_args_12"
)

for i in {1..12}; do
    # echo ${methods[i-1]}
    # name="--experiment_name LSS_b3_tune2_$i --wandb_name vari_$i"
    python $base_args $base_method_args $bench4 ${methods[i-1]} $name
done

# python $base_args $bench $baseline_args

# python $base_args $bench $base_method_args $method_args_1 

# python $base_args $bench2 $base_method_args $method_args_2

# arg1="--experiment_name LSS_b3_1_2 $method_args_1"
# python $base_args $base_method_args $bench3 $arg1

# arg2="--experiment_name LSS_b4_1 $method_args_1"
# python $base_args $base_method_args $bench4 $arg2

# arg3="--experiment_name LSS_b3_2_2 $method_args_2"
# python $base_args  $base_method_args $bench3 $arg3

# arg4="--experiment_name LSS_b4_2 $method_args_2"
# python $base_args $base_method_args $bench4 $arg4

# arg5="--experiment_name LSS_b3_3_2 $method_args_3"
# python $base_args $base_method_args $bench3 $arg5

# arg6="--experiment_name LSS_b4_3 $method_args_3"
# python $base_args $base_method_args $bench4 $arg6



