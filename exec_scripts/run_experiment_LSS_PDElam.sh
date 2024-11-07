

## Init

readonly wandb_project="deepreach_hopf_lambda_var"
readonly experiment_name="LLlambda7D_DR" ## must overwrite to save data, will save model_final?

# readonly output_dir_name="./runs/baseline_50D_200k"

readonly dim="7"
readonly goalR="0.25"
readonly num_epochs="100000"
readonly bs="65000"

readonly base_args="run_experiment.py --experiment_name $experiment_name --use_wandb --wandb_project $wandb_project --wandb_name LLlambda7D_NL_LSS --dynamics_class LessLinearNDlambda --N $dim --goalR $goalR --num_epochs $num_epochs --numpoints $bs"

readonly bench_mag="20"
readonly bench1="--gamma $bench_mag --mu 0 --alpha 0"
readonly bench2="--gamma -$bench_mag --mu 0 --alpha 0"
readonly bench3="--gamma -$bench_mag --mu -$bench_mag --alpha 1"
readonly bench4="--gamma $bench_mag --mu -$bench_mag --alpha 1"

## Method

readonly supervisor="LL7D_DR_linear"
readonly hopf_loss_decay_w="0.8"
readonly hopf_pretrain_iters="10000"

# readonly method_args="--baseline" ## Baseline
readonly method_args="--hopf_pretrain_iters $hopf_pretrain_iters --hopf_loss_decay --hopf_loss_decay_type linear --hopf_loss_decay_w $hopf_loss_decay_w --diff_con_loss_incr --load_hopf_model --load_hopf_model_name $supervisor" ## Decayed LSS
# readonly method_args="--hopf_pretrain_iters $hopf_pretrain_iters --nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000  --load_hopf_model --load_hopf_model_name $supervisor" ## NL Curriculum Scale LSS
# readonly method_args="--capacity" ## Capacity

## Execute

# mkdir -p $output_dir_name

python $base_args $bench1 $method_args

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
