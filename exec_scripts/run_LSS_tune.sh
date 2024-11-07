
## Init

readonly wandb_project="deepreach_hopf_tunes"
readonly output_dir_name="./runs_tune/LSS_decay"
readonly group_run_name="run_1"
mkdir -p $output_dir_name

readonly dim="50"
readonly goalR="0.25"
readonly num_epochs="100000"
# readonly bs="65000"

readonly base_args="run_experiment.py --experiments_dir $output_dir_name/$group_run_name --use_wandb --wandb_project $wandb_project --wandb_name tune_run1 --dynamics_class LessLinearND --N $dim --goalR $goalR --num_epochs $num_epochs"

readonly bench_mag="20"
readonly bench="--gamma $bench_mag --mu 0 --alpha 0"
# readonly bench2="--gamma -$bench_mag --mu 0 --alpha 0"
# readonly bench3="--gamma -$bench_mag --mu -$bench_mag --alpha 1"
# readonly bench4="--gamma $bench_mag --mu -$bench_mag --alpha 1"

## Method

readonly supervisor="LL50D_DR_linear_r25"
readonly hopf_pretrain_iters="4000"

# readonly method_args="--baseline" ## Baseline
readonly method_args="--hopf_pretrain_iters $hopf_pretrain_iters --hopf_loss_decay --diff_con_loss_incr --load_hopf_model --load_hopf_model_name $supervisor" ## Decayed LSS
# readonly method_args="--nl_scale --nl_scale_epoch_step 2000 --nl_scale_epoch_post 50000 --load_hopf_model --load_hopf_model_name $supervisor" ## NL Curriculum Scale LSS
# readonly method_args="--capacity" ## Capacity

method_args_1="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t1 --hopf_loss_decay_type linear --hopf_loss_decay_w 0.4 --numpoints 10000"
method_args_2="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t2 --hopf_loss_decay_type linear --hopf_loss_decay_w 0.8 --numpoints 10000"
method_args_3="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t3 --hopf_loss_decay_type exponential --hopf_loss_decay_w 0.99997 --numpoints 10000"
method_args_4="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t4 --hopf_loss_decay_type exponential --hopf_loss_decay_w 0.99999 --numpoints 10000"

method_args_5="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t5 --hopf_loss_decay_type linear --hopf_loss_decay_w 0.4 --numpoints 65000"
method_args_6="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t6 --hopf_loss_decay_type linear --hopf_loss_decay_w 0.8 --numpoints 65000"
method_args_7="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t7 --hopf_loss_decay_type exponential --hopf_loss_decay_w 0.99997 --numpoints 65000"
method_args_8="--hopf_loss lin_val_diff --experiment_name LL50D_LSS_decay_t8 --hopf_loss_decay_type exponential --hopf_loss_decay_w 0.99999 --numpoints 65000"

# other things to tr?
# solve_gradient
# hopf_loss_divisor
# if I am going to bring down the decay_w, can I also reduce the number of epochs?

## Execute

python $base_args $bench $method_args_1 

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t1.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t1/training/checkpoints/model_final.pth $output_dir_name/model_final_t1.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t1/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t1.png

python $base_args $bench $method_args_2

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t2.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t2/training/checkpoints/model_final.pth $output_dir_name/model_final_t2.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t2/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t2.png

python $base_args $bench $method_args_3

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t3.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t3/training/checkpoints/model_final.pth $output_dir_name/model_final_t3.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t3/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t3.png

python $base_args $bench $method_args_4

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t4.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t4/training/checkpoints/model_final.pth $output_dir_name/model_final_t4.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t4/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t4.png

python $base_args $bench $method_args_5 

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t5.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t5/training/checkpoints/model_final.pth $output_dir_name/model_final_t5.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t5/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t5.png

python $base_args $bench $method_args_6

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t6.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t6/training/checkpoints/model_final.pth $output_dir_name/model_final_t6.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t6/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t6.png

python $base_args $bench $method_args_7

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t7.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t7/training/checkpoints/model_final.pth $output_dir_name/model_final_t7.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t7/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t7.png

python $base_args $bench $method_args_8

cp wandb/latest-run/files/wandb-summary.json $output_dir_name/wandb-summary_t8.json
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t8/training/checkpoints/model_final.pth $output_dir_name/model_final_t8.pth
cp $output_dir_name/$group_run_name/LL50D_LSS_decay_t8/training/checkpoints/BRS_validation_plot.png $output_dir_name/BRS_validation_plot_t8.png

cd $output_dir_name
jq -s '.' wandb-summary_t1.json wandb-summary_t2.json wandb-summary_t3.json wandb-summary_t4.json wandb-summary_t5.json wandb-summary_t6.json wandb-summary_t7.json wandb-summary_t8.json > tune_tally.json
cd ..
