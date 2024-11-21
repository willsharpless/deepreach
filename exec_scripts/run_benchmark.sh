

## Init

readonly wandb_project="deepreach_hopf_benchmarks"
# readonly experiment_name="bench_holder" ## must overwrite to save data, will save model_final?

readonly summary_dir_name="./summaries/ZLLS_hld100_hgld250_50D_300k"
readonly experiments_dir="./runs"
readonly dim="50"
readonly goalR="0.25"
readonly num_epochs="300000"
readonly bs="65000"
readonly lr="1e-6"

readonly base_args="run_experiment.py --use_wandb --experiments_dir $experiments_dir --wandb_project $wandb_project --N $dim --goalR $goalR --num_epochs $num_epochs --numpoints $bs --lr_std $lr"

readonly bench_mag="20"
readonly bench1="--gamma $bench_mag --mu 0 --alpha 0"
readonly bench2="--gamma -$bench_mag --mu 0 --alpha 0"
readonly bench3="--gamma -$bench_mag --mu -$bench_mag --alpha 1"
readonly bench4="--gamma $bench_mag --mu -$bench_mag --alpha 1"

## Method

readonly supervisor="./runs/LL50D_DR_linear_r25"

# readonly method_args="--dynamics_class LessLinearND --baseline" ## Baseline

# readonly method_args="--dynamics_class LessLinearND --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w 0.8 --diff_con_loss_incr --load_hopf_model --load_hopf_model_name $supervisor" ## Decayed LSS

# readonly method_args="--dynamics_class LessLinearND --nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000  --load_hopf_model --load_hopf_model_name $supervisor" ## NL Curriculum Scale LSS

readonly method_args="--dynamics_class LessLinearNDlambda --hopf_loss lin_val_grad_diff --load_hopf_model --load_hopf_model_name $supervisor --zerolambda_LS --LS_w_time_curr --hopf_pretrain_iters 1 --hopf_loss_divisor 100 --hopf_grad_loss_divisor 250"
readonly method_args_2="--dynamics_class LessLinearNDlambda --hopf_loss lin_val_grad_diff --load_hopf_model --load_hopf_model_name $supervisor --zerolambda_LS --LS_w_time_curr --hopf_pretrain_iters 1 --hopf_loss_divisor 1000 --hopf_grad_loss_divisor 4000"
readonly method_args_3="--dynamics_class LessLinearNDlambda --hopf_loss lin_val_grad_diff --load_hopf_model --load_hopf_model_name $supervisor --zerolambda_LS --LS_w_time_curr --hopf_pretrain_iters 1 --hopf_loss_divisor 500 --hopf_grad_loss_divisor 4000"

# readonly method_args="--capacity" ## Capacity

## Execute

# mkdir -p $summary_dir_name

# name="ZLLS_hld100_hgld250_50D_300k_b1"
# experiment_name="--experiment_name $name"
# wandb_name="--wandb_name ZLLS_hld100_hgld250_b1"
# python $base_args $bench1 $method_args $experiment_name $wandb_name
# # echo $base_args $bench1 $method_args $experiment_name $wandb_name

# cp wandb/latest-run/files/wandb-summary.json $summary_dir_name/wandb-summary_b1.json
# cp $experiments_dir/$name/training/checkpoints/model_final.pth $summary_dir_name/model_final_b1.pth
# cp $experiments_dir/$name/training/checkpoints/BRS_validation_plot.png $summary_dir_name/BRS_validation_plot_b1.png

# name="ZLLS_hld1k_hgld4k_50D_300k_b2"
# experiment_name="--experiment_name $name"
# wandb_name="--wandb_name ZLLS_hld1k_hgld4k_b2"
# python $base_args $bench2 $method_args_2 $experiment_name $wandb_name
# # echo $base_args $bench2 $method_args $experiment_name $wandb_name

# cp wandb/latest-run/files/wandb-summary.json $summary_dir_name/wandb-summary_b2.json
# cp $experiments_dir/$name/training/checkpoints/model_final.pth $summary_dir_name/model_final_b2.pth
# cp $experiments_dir/$name/training/checkpoints/BRS_validation_plot.png $summary_dir_name/BRS_validation_plot_b2.png

# name="ZLLS_hld100_hgld250_50D_300k_b3"
# experiment_name="--experiment_name $name"
# wandb_name="--wandb_name ZLLS_hld100_hgld250_b3"
# python $base_args $bench3 $method_args $experiment_name $wandb_name
# # echo $base_args $bench3 $method_args $experiment_name $wandb_name

# cp wandb/latest-run/files/wandb-summary.json $summary_dir_name/wandb-summary_b3.json
# cp $experiments_dir/$name/training/checkpoints/model_final.pth $summary_dir_name/model_final_b3.pth
# cp $experiments_dir/$name/training/checkpoints/BRS_validation_plot.png $summary_dir_name/BRS_validation_plot_b3.png

name="ZLLS_hld500_hgld4k_50D_300k_b4"
experiment_name="--experiment_name $name"
wandb_name="--wandb_name ZLLS_hld500_hgld4k_b4"
python $base_args $bench4 $method_args_3 $experiment_name $wandb_name
# echo $base_args $bench4 $method_args $experiment_name $wandb_name

cp wandb/latest-run/files/wandb-summary.json $summary_dir_name/wandb-summary_b4.json
cp $experiments_dir/$name/training/checkpoints/model_final.pth $summary_dir_name/model_final_b4.pth
cp $experiments_dir/$name/training/checkpoints/BRS_validation_plot.png $summary_dir_name/BRS_validation_plot_b4.png

cd $summary_dir_name
jq -s '.' wandb-summary_b1.json wandb-summary_b2.json wandb-summary_b3.json wandb-summary_b4.json > benchmark_tally.json