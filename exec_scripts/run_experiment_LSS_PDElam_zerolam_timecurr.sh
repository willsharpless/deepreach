

## Init

readonly wandb_project="DRH_LLl7D_ZLLS_TC"

readonly output_dir_name="./runs/DRH_LLl50D_ZLLS_TC_tuning"

readonly dim="50"
readonly goalR="0.25"
# num_epochs="100000"

readonly supervisor="./runs/LL50D_DR_linear_r25"
readonly base_args="run_experiment.py --use_wandb --wandb_project $wandb_project --dynamics_class LessLinearNDlambda --N $dim --goalR $goalR --solve_grad"
# readonly base_args="run_experiment.py --dynamics_class LessLinearNDlambda --N $dim --goalR $goalR --solve_grad" # no wandb test

readonly bench_mag="20"
readonly bench1="--gamma $bench_mag --mu 0 --alpha 0"
readonly bench2="--gamma -$bench_mag --mu 0 --alpha 0"
readonly bench3="--gamma -$bench_mag --mu -$bench_mag --alpha 1"
readonly bench4="--gamma $bench_mag --mu -$bench_mag --alpha 1"

## Method

readonly baseline_args="--baseline --numpoints 60000 --lr_std 5e-6 --num_epochs 300000 --experiment_name $output_dir_name/LLl7D_baseline_300k --wandb_name LLl50D_baseline_300k" # ALSO NEED TO TUNE BASELINE LAMBDA MODEL!

readonly hopf_pretrain_iters="1"
readonly base_ZLLS_TC_args="--hopf_loss lin_val_grad_diff --load_hopf_model --load_hopf_model_name $supervisor --zerolambda_LS --LS_w_time_curr --hopf_pretrain_iters $hopf_pretrain_iters"

## Variations

# more less hopf w/ bigger batch
bs="10000"
# method_args_1="--numpoints $bs --hopf_loss_divisor 25 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld25_hgld250_bs10k_e200k --wandb_name hld25_hgld250_bs10k_e200k --num_epochs 200000"
# method_args_2="--numpoints $bs --hopf_loss_divisor 50 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld50_hgld250_bs10k_e200k --wandb_name hld50_hgld250_bs10k_e200k --num_epochs 200000"
# method_args_3="--numpoints $bs --hopf_loss_divisor 100 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld100_hgld250_bs10k_e200k --wandb_name hld100_hgld250_bs10k_e200k --num_epochs 200000"
# method_args_4="--numpoints $bs --hopf_loss_divisor 200 --hopf_grad_loss_divisor 500 --experiment_name $output_dir_name/hld200_hgld500_bs10k_e200k --wandb_name hld200_hgld500_bs10k_e200k --num_epochs 200000"

# more less hopf w/ bigger batch
bs="60000"
# method_args_5="--numpoints $bs --hopf_loss_divisor 25 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld25_hgld250_bs60k_e100k --wandb_name hld25_hgld250_bs60k_e100k --num_epochs 100000"
# method_args_6="--numpoints $bs --hopf_loss_divisor 50 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld50_hgld250_bs60k_e100k --wandb_name hld50_hgld250_bs60k_e100k --num_epochs 100000"
# method_args_7="--numpoints $bs --hopf_loss_divisor 100 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld100_hgld250_bs60k_e100k --wandb_name hld100_hgld250_bs60k_e100k --num_epochs 100000"
# method_args_8="--numpoints $bs --hopf_loss_divisor 200 --hopf_grad_loss_divisor 500 --experiment_name $output_dir_name/hld200_hgld500_bs60k_e100k --wandb_name hld200_hgld500_bs60k_e100k --num_epochs 100000"

# method_args_5="--numpoints $bs --hopf_loss_divisor 25 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld25_hgld250_bs60k_e200k --wandb_name hld25_hgld250_bs60k_e200k --num_epochs 200000"
# method_args_6="--numpoints $bs --hopf_loss_divisor 50 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld50_hgld250_bs60k_e200k --wandb_name hld50_hgld250_bs60k_e200k --num_epochs 200000"
# method_args_7="--numpoints $bs --hopf_loss_divisor 100 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld100_hgld250_bs60k_e200k --wandb_name hld100_hgld250_bs60k_e200k --num_epochs 200000"
# method_args_8="--numpoints $bs --hopf_loss_divisor 200 --hopf_grad_loss_divisor 500 --experiment_name $output_dir_name/hld200_hgld500_bs60k_e200k --wandb_name hld200_hgld500_bs60k_e200k --num_epochs 200000"

# more less hopf w/ bigger batch and long time
method_args_9=" --lr_std 5e-6 --numpoints $bs --hopf_loss_divisor 200 --hopf_grad_loss_divisor 1000 --experiment_name $output_dir_name/hld200_hgld1000_bs60k_e300k --wandb_name hld200_hgld1000_bs60k_e300k --num_epochs 300000"
# method_args_10=" --lr_std 5e-6 --numpoints $bs --hopf_loss_divisor 100 --hopf_grad_loss_divisor 1000 --experiment_name $output_dir_name/hld100_hgld1000_bs60k_e300k --wandb_name hld100_hgld1000_bs60k_e300k --num_epochs 300000"
# method_args_11=" --lr_std 5e-6 --numpoints $bs --hopf_loss_divisor 50 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld50_hgld250_bs60k_e300k --wandb_name hld50_hgld250_bs60k_e300k --num_epochs 300000"
# method_args_12=" --lr_std 5e-6 --numpoints $bs --hopf_loss_divisor 25 --hopf_grad_loss_divisor 250 --experiment_name $output_dir_name/hld25_hgld250_bs60k_e300k --wandb_name hld25_hgld250_bs60k_e300k --num_epochs 300000"

# more less hopf w/ slower lr w/ bigger batch
# bs="60000"
# method_args_9="--numpoints $bs --lr_std 5e-6 --hopf_loss_divisor 50 --hopf_grad_loss_divisor 25 --experiment_name bs60k_LR5e6_less_hopf_10x --wandb_name bs60k_LR5e6_less_hopf_10x"
# method_args_10="--numpoints $bs --lr_std 5e-6 --hopf_loss_divisor 10 --hopf_grad_loss_divisor 2.5 --experiment_name bs60k_LR5e6_more_grad_10x --wandb_name bs60k_LR5e6_more_grad_10x"
# method_args_11="--numpoints $bs --lr_std 5e-6 --hopf_loss_divisor 10 --hopf_grad_loss_divisor 250 --experiment_name bs60k_LR5e6_less_grad_10x --wandb_name bs60k_LR5e6_less_grad_10x"
# method_args_12="--numpoints $bs --lr_std 5e-6 --hopf_loss_divisor 50 --hopf_grad_loss_divisor 250 --experiment_name bs60k_LR5e6_less_both_10x --wandb_name bs60k_LR5e6_less_both_10x"

## Execute

mkdir -p $output_dir_name

# python $base_args $bench1 $method_args_1
# python $base_args $bench1 $method_args_2
# python $base_args $bench1 $method_args_3

## Execute
methods=(
    # "$method_args_1" 
    # "$method_args_2" 
    # "$method_args_3" 
    # "$method_args_4" 
    # "$method_args_5" 
    # "$method_args_6" 
    # "$method_args_7" 
    # "$method_args_8" 
    # "$method_args_9" 
    # "$method_args_10" 
    # "$method_args_11" 
    # "$method_args_12"
)

len=$(( ${#methods[@]} + 1 ))
for i in $(seq 1 $len); do
    echo "Run" $i
    if [ "$i" -lt "$len" ]; then
        # python $base_args $bench1 $base_ZLLS_TC_args ${methods[i-1]}
        echo $base_args $bench1 $base_ZLLS_TC_args ${methods[i-1]}
    else
        python $base_args $bench1 $baseline_args
        python $base_args $bench2 $baseline_args
        python $base_args $bench3 $baseline_args
        python $base_args $bench4 $baseline_args
        # python $base_args $bench1 $baseline_args
        # python $base_args $bench1 $baseline_args
        # echo $base_args $bench1 $baseline_args
    fi
done