
## Init

readonly base_params="--N 2 --tMax 2."
readonly convr_dyn="--dynamics_class ConveyorND $base_params"
readonly canoe_dyn="--dynamics_class CanoeND $base_params"
readonly convr_lam_dyn="--dynamics_class ConveyorNDlambda $base_params"
readonly canoe_lam_dyn="--dynamics_class CanoeNDlambda $base_params"

readonly use_wandb="--use_wandb"
# readonly use_wandb=""
readonly wandb_project="mulob_BRAAT_comp"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_test="--experiments_dir ./runs/mulob/test"
readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/Conveyor2D/"
readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/Canoe2D"

# readonly gt_args="--num_epochs 10000 --counter_end 20000" #TODO: gt supervision
readonly learn_params_fast="--pretrain_iters 500 --num_epochs 5000 --counter_end 500" # fast
readonly learn_params="--pretrain_iters 500 --num_epochs 22000 --counter_end 1000" # mid
readonly learn_params_slow="--pretrain_iters 2000 --num_epochs 50000 --counter_end 5000" # mid
readonly learn_params_long="--pretrain_iters 2000 --num_epochs 300000 --counter_end 40000" # long

readonly load_params="--load_decomposed_models --load_decomposed_model_name_1 ./runs/mulob/ConveyorND/Conveyor2D/reach_only_2D --load_decomposed_model_name_2 ./runs/mulob/ConveyorND/Conveyor2D/avoid_only_2D_fast2"
readonly super_pt_params="--super_pretrain --super_pretrain_iters 5000"
# readonly super_params="--super_pretrain --super_pretrain_iters 10000 --solve_grad --grad_super"

### Conveyor

## Decomposed

# python run_experiment.py $convr_dyn --reach_only True $learn_params_fast $exp_dir_convr --experiment_name reach_only_2D_fast2 --wandb_name reach_only $wandb_parms

# python run_experiment.py $convr_dyn --reach_only True $learn_params $exp_dir_convr --experiment_name reach_only_2D --wandb_name reach_only $wandb_parms

# python run_experiment.py $convr_dyn --avoid_only True $learn_params_fast $exp_dir_convr --experiment_name avoid_only_2D_fast2 --wandb_name avoid_only $wandb_parms

# python run_experiment.py $convr_dyn --avoid_only True $learn_params $exp_dir_convr --experiment_name avoid_only_2D --wandb_name avoid_only $wandb_parms

## Avoid only - axes

# python run_experiment.py $convr_dyn --deepreach_model vanilla --avoid_only True --avoid_type axes $learn_params $exp_dir_convr --experiment_name avoid_only_2D_axes_vanilla #--wandb_name avoid_only $wandb_parms

# python run_experiment.py $convr_dyn --deepreach_model vanilla --avoid_only True --avoid_type axes $learn_params_slow $exp_dir_convr --experiment_name avoid_only_2D_axes_slow_vanilla --wandb_name avoid_only_vanilla $wandb_parms

# python run_experiment.py $convr_dyn --deepreach_model vanilla --avoid_only True --avoid_type axes $learn_params_long $exp_dir_convr --experiment_name avoid_only_2D_axes_long_vanilla --wandb_name avoid_only_vanilla $wandb_parms

# python run_experiment.py $convr_dyn --deepreach_model exact --avoid_only True --avoid_type axes $learn_params_long $exp_dir_convr --experiment_name avoid_only_2D_axes_long_exact --wandb_name avoid_only_exact $wandb_parms

## BRAAT

# python run_experiment.py $convr_dyn --old_brat True --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAT_slow_exact --wandb_name BRAT_slow_exact $wandb_parms

# python run_experiment.py $convr_dyn --mulob_type BRAT --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAT_mulob_slow_exact --wandb_name BRAT_mulob_slow_exact $wandb_parms

# python run_experiment.py $convr_dyn $load_params --mulob_type BRAAT $learn_params_slow $exp_dir_convr --experiment_name BRAAT_std --wandb_name BRAAT_std $wandb_parms

# python run_experiment.py $convr_dyn --mulob_type BRAAT --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAAT_gtdecomp --wandb_name BRAAT_gtdecomp $wandb_parms

# python run_experiment.py $convr_lam_dyn --mulob_type BRAAT --mulob_loss_type augment --lam_slice_super --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAAT_slow_exact_lam_lss --wandb_name BRAAT_slow_exact_lam_lss $wandb_parms

# python run_experiment.py $convr_lam_dyn --mulob_type BRAAT --mulob_loss_type augment --deepreach_model exact $learn_params_slow $exp_dir_convr --experiment_name BRAAT_slow_exact_lam --wandb_name BRAAT_slow_exact_lam $wandb_parms

# super tests
# python run_experiment.py $convr_lam_dyn $super_params --mulob_type BRAAT --mulob_loss_type augment --lam_slice_super --deepreach_model exact $learn_params $exp_dir_convr --experiment_name SUPER_TEST_BRAAT_lam_slice --wandb_name SUPER_TEST_BRAAT_lam_slice $wandb_parms

# python run_experiment.py $convr_lam_dyn $super_params --mulob_type BRAAT --mulob_loss_type augment --deepreach_model exact $learn_params $exp_dir_convr --experiment_name SUPER_TEST_BRAAT_lam --wandb_name SUPER_TEST_BRAAT_lam $wandb_parms


# # standard BRAT
# python run_experiment.py $convr_dyn --old_brat True $learn_params_slow $exp_dir_convr --experiment_name BRAT --wandb_name BRAT $wandb_parms

# # standard BRAAT
# python run_experiment.py $convr_dyn $load_params --mulob_type BRAAT $learn_params_slow $exp_dir_convr --experiment_name BRAAT --wandb_name BRAAT $wandb_parms

# # lambda-var BRAAT
# python run_experiment.py $convr_lam_dyn $load_params --mulob_type BRAAT $learn_params_slow $exp_dir_convr --experiment_name BRAAT_lam --wandb_name BRAAT_lam $wandb_parms

# # lambda-var BRAAT + lambda super vision on slices (w/wo time-curriculum, gradual pinn intro)
# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super --lam_slice_super --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_slice_noTcurr_grad --wandb_name BRAAT_lam_super_slice_noTcurr_grad $wandb_parms

# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 500 --solve_grad --grad_super --lam_slice_super --LS_w_time_curr --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_slice_wTcurr_grad --wandb_name BRAAT_lam_super_slice_wTcurr_grad $wandb_parms

# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super --lam_slice_super --gradual_pinn_loss --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_slice_noTcurr_easepinn_grad --wandb_name BRAAT_lam_super_slice_noTcurr_easepinn_grad $wandb_parms

# # lambda-var BRAAT + lambda super vision across lam (w/wo time-curriculum, gradual pinn intro)
# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_free_noTcurr_grad --wandb_name BRAAT_lam_super_free_noTcurr_grad $wandb_parms

# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 500 --solve_grad --grad_super --LS_w_time_curr --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_free_wTcurr_grad --wandb_name BRAAT_lam_super_free_wTcurr_grad $wandb_parms

# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super --gradual_pinn_loss --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_slice_wTcurr_easepinn_grad --wandb_name BRAAT_lam_super_slice_wTcurr_easepinn_grad $wandb_parms

# speed check
python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --lam_slice_super --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_slice_noTcurr_grad_faster_nohopf --wandb_name BRAAT_lam_super_slice_noTcurr_faster_nohopf $wandb_parms
# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super --lam_slice_super --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_slice_noTcurr_grad_faster --wandb_name BRAAT_lam_super_slice_noTcurr_grad_faster $wandb_parms
python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super_time --lam_slice_super --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_slice_noTcurr_gradtime_faster_nohopf --wandb_name BRAAT_lam_super_slice_noTcurr_gradtime_faster_nohopf $wandb_parms

python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000  --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_free_noTcurr_grad_faster_nohopf --wandb_name BRAAT_lam_super_free_noTcurr_faster_nohopf $wandb_parms
# python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_free_noTcurr_grad_faster --wandb_name BRAAT_lam_super_free_noTcurr_grad_faster $wandb_parms
python run_experiment.py $convr_lam_dyn $load_params --super_pretrain --super_pretrain_iters 5000 --solve_grad --grad_super_time --mulob_type BRAAT --mulob_loss_type augment $learn_params $exp_dir_convr --experiment_name BRAAT_lam_super_free_noTcurr_gradtime_faster_nohopf --wandb_name BRAAT_lam_super_free_noTcurr_gradtime_faster_nohopf $wandb_parms


### Canoe

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_fast $exp_dir_canoe --experiment_name reach_1_only_2D_fast #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_2_only True $learn_params_fast $exp_dir_canoe --experiment_name reach_2_only_2D_fast #--wandb_name reach_2_only $wandb_parms


# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D_exact_1p5 --deepreach_model exact #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_slow $exp_dir_canoe --experiment_name reach_1_only_2D_slow_exact_1p5 --deepreach_model exact #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_2D_vanilla_1p5 --deepreach_model vanilla #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_1_only True $learn_params_slow $exp_dir_canoe --experiment_name reach_1_only_2D_slow_vanilla_1p5 --deepreach_model vanilla #--wandb_name reach_1_only $wandb_parms

# python run_experiment.py $canoe_dyn --reach_2_only True $learn_params $exp_dir_canoe --experiment_name reach_2_only_2D #--wandb_name reach_2_only $wandb_parms