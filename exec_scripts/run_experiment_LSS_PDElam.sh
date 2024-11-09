

## Init

readonly wandb_project="deepreach_hopf_lambda_var"

# readonly output_dir_name="./runs/baseline_50D_200k"

readonly dim="7"
readonly goalR="0.25"
# readonly num_epochs="100000"
readonly bs="10000"

readonly base_args="run_experiment.py --use_wandb --wandb_project $wandb_project --dynamics_class LessLinearNDlambda --N $dim --goalR $goalR --numpoints $bs --solve_grad"

readonly bench_mag="20"
readonly bench1="--gamma $bench_mag --mu 0 --alpha 0"
readonly bench2="--gamma -$bench_mag --mu 0 --alpha 0"
readonly bench3="--gamma -$bench_mag --mu -$bench_mag --alpha 1"
readonly bench4="--gamma $bench_mag --mu -$bench_mag --alpha 1"

## Method

readonly supervisor="./runs/LL7D_DR_linear_fast"
readonly hopf_loss_decay_w="0.8"
readonly hopf_pretrain_iters="5000"

# readonly method_args="--baseline" ## Baseline
# readonly method_args="--hopf_loss lin_val_grad_diff --hopf_pretrain_iters $hopf_pretrain_iters --nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000  --load_hopf_model --load_hopf_model_name $supervisor" ## NL Curriculum Scale LSS
# readonly method_args="--capacity" ## Capacity

# readonly method_args="--hopf_loss lin_val_diff --solve_grad --hopf_pretrain_iters $hopf_pretrain_iters --load_hopf_model --load_hopf_model_name $supervisor" ## --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w $hopf_loss_decay_w --diff_con_loss_incr
# readonly method_args="--hopf_loss lin_val_diff --solve_grad --hopf_pretrain_iters $hopf_pretrain_iters --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w $hopf_loss_decay_w --load_hopf_model --load_hopf_model_name $supervisor --experiment_name LLlambda7D_LSS_decay_fast --wandb_name LLlambda7D_LSS_fast" ## --diff_con_loss_incr


# Basic High Performers

# ## Linear Supervision Decayed (Lambda Model)
# readonly method_args_1="--hopf_loss lin_val_diff --solve_grad --hopf_pretrain_iters $hopf_pretrain_iters --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w $hopf_loss_decay_w --diff_con_loss_incr --load_hopf_model --load_hopf_model_name $supervisor --experiment_name LLlambda7D_LSS_decay_dwp8_long --wandb_name LLlambda7D_LSS_decay_dwp8_long --num_epochs 100000" ## --diff_con_loss_incr

# ## Linear Supervision on Zero Lambda (Lambda Model)
# readonly method_args_2="--hopf_loss lin_val_diff --zerolambda_LS --solve_grad --hopf_pretrain_iters $hopf_pretrain_iters --load_hopf_model --load_hopf_model_name $supervisor --experiment_name LLlambda7D_ZLLS_long --wandb_name LLlambda7D_ZLLS_long --num_epochs 100000" ## --diff_con_loss_incr

# ## Linear Supervision on Zero Lambda with time-curriculum sampling (Lambda Model)
# readonly method_args_3="--hopf_loss lin_val_diff --zerolambda_LS --LS_w_time_curr --solve_grad --hopf_pretrain_iters 1 --load_hopf_model --load_hopf_model_name $supervisor --experiment_name LLlambda7D_ZLLS_timecurr --wandb_name LLlambda7D_ZLLS_timecurr --num_epochs 100000" ## --diff_con_loss_incr

## With Grad
readonly method_args_1="--hopf_loss lin_val_grad_diff --hopf_pretrain_iters $hopf_pretrain_iters --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w $hopf_loss_decay_w --diff_con_loss_incr --load_hopf_model --load_hopf_model_name $supervisor --experiment_name LLlambda7D_LSS_decay_dwp8_grad --wandb_name LLlambda7D_LSS_decay_dwp8_grad --num_epochs 100000" ## --diff_con_loss_incr
readonly method_args_2="--hopf_loss lin_val_grad_diff --zerolambda_LS --hopf_pretrain_iters $hopf_pretrain_iters --load_hopf_model --load_hopf_model_name $supervisor --experiment_name LLlambda7D_ZLLS_grad --wandb_name LLlambda7D_ZLLS_grad --num_epochs 100000" ## --diff_con_loss_incr
readonly method_args_3="--hopf_loss lin_val_grad_diff --zerolambda_LS --LS_w_time_curr --hopf_pretrain_iters 1 --load_hopf_model --load_hopf_model_name $supervisor --experiment_name LLlambda7D_ZLLS_timecurr_grad --wandb_name LLlambda7D_ZLLS_timecurr_grad --num_epochs 100000" ## --diff_con_loss_incr

# ALSO NEED TO TUNE BASELINE LAMBDA MODEL!


## Execute

# mkdir -p $output_dir_name

python $base_args $bench1 $method_args_1
python $base_args $bench1 $method_args_2
python $base_args $bench1 $method_args_3

# cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_b1.json
# cp runs/$experiment_name/training/checkpoints/model_final.pth $output_dir_name/model_final_b1.pth
# cp runs/$experiment_name/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_b1.png

# python $base_args $bench2 $method_args

# cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_b2.json
# cp runs/$experiment_name/training/checkpoints/model_final.pth $output_dir_name/model_final_b2.pth
# cp runs/$experiment_name/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_b2.png

# python $base_args $bench3 $method_args

# cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_b3.json
# cp runs/$experiment_name/training/checkpoints/model_final.pth $output_dir_name/model_final_b3.pth
# cp runs/$experiment_name/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_b3.png

# python $base_args $bench4 $method_args

# cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_b4.json
# cp runs/$experiment_name/training/checkpoints/model_final.pth $output_dir_name/model_final_b4.pth
# cp runs/$experiment_name/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_b4.png

# cd $output_dir_name
# jq -s '.' wandb-summary_b1.json wandb-summary_b2.json wandb-summary_b3.json wandb-summary_b4.json > benchmark_tally.json
# cd ..
