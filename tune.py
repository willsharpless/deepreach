import subprocess
import os
import argparse
from datetime import datetime

## Parse
parser = argparse.ArgumentParser()
parser.add_argument("-trf",   "--trainfile",         default="run_experiment_hopf_tune.py")
parser.add_argument("-fn",    "--runname",           default="test") #DR_Hopf_WAS
parser.add_argument("-nt",    "--nametag",           default="test") #sw1
parser.add_argument("-poi",   "--paramsofinterest",  default="lr lr_hopf hopf_pretrain_iters")
parser.add_argument("-sc",    "--scale",             default="1e-5, 1e-5, 1000 2000 5000") #sep p sets by comma + space, p vals by space eg. "NC C HC, 2 4 6 8 10, 1e-3 1e-4"
# parser.add_argument("-ld",    "--loadpath",          default="") # if not loading: "", else give local name i.e. "AEwarm"
parser.add_argument("-ns",    "--numberofseeds",     default="1")
parser.add_argument("-is",    "--initialseed",       default="1")
# parser.add_argument("-owt",   "--overwrite",         default="True") # (not the loaded file)
# parser.add_argument("-plt",   "--plotting",          default="True")
args = vars(parser.parse_args())

## Make Dir
pois = args["paramsofinterest"].split(" ")
fn = args["runname"] if args["runname"] else "Scan_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
scan_path = os.path.join(os.getcwd(), "Tuning", fn)
# os.makedirs(scan_path, exist_ok=True)

## Make Scale
# scale = args["scale"].split(" ")
scales = [sc.split(" ") for sc in args["scale"].split(", ")]

## Print for Log
print("\nDeepReach + Hopf Formula HJR Learning")
print("Level 2 Nonlinearity Sweep (Gamma 20, Mu -20, Alpha 1)")
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("Run by WAS")

print("\nHyper Parameter scan of " + str(pois))
print("scales:" + str(scales))

## Iteratively Train for Different Parameter Values and Examine

## What I was hoping to solve:
# sweep_configuration = {
#     "method": "random",
#     "name": "bayes",
#     "metric": {"goal": "maximize", "name": "Max Smooth Jaccard Index over Time"}, 
#     "parameters": {
#         "seed":                {"values": [0, 1, 2]},
#         "lr":                  {"max": 1e-3,   "min": 1e-5},
#         "lr_hopf":             {"max": 1e-3,   "min": 1e-5},
#         "lr_decay_w":          {"max": 0.99,   "min": 0.96},
#         "lr_hopf_decay_w":     {"max": 0.99,   "min": 0.96},
#         "hopf_loss_divisor":   {"max": 10.,    "min": 0.1},
#         "hopf_loss_decay_w":   {"max": 0.9999, "min": 0.9996},
#         "hopf_pretrain_iters": {"max": 20000,  "min": 2000},
#     },
# }

def train_helper(pois, scales, pval_path_init, poi_args_init, depth, num_seeds):

    poi = pois[depth]
    scale = scales[depth]

    for px, pval in enumerate(scale):

        pval_path = os.path.join(pval_path_init, poi + '_' + pval)
        poi_args = poi_args_init + ['--' + poi, pval]
        # os.makedirs(pval_path, exist_ok=True)

        ## At Final Depth, iterate Seeds and Train
        if depth == len(scales) - 1:
            for sdx in range(int(args["initialseed"]), int(args["initialseed"]) + num_seeds):
                # printB = "True" if (px == 0 and sdx == 1) else "False"

                name = (' ').join([args["nametag"]] + poi_args + ["--seed", str(sdx)])
                # load_name = args["loadpath"] + "_" + str(sdx) if args["loadpath"] else ""

                info_args = ["--gamma", "20", "--mu", "-20", "--alpha", "1",
                             "--experiment_name", "test", 
                            #  "--use_wandb", #FIXME
                            #  "--wandb_project", "deepreach_hopf_sweep",
                            #  "--wandb_entity", "sas-lab",
                            #  "--wandb_group", "LessLinear2D",
                            #  "--wandb_name", name
                             ]
                
                # load_args = ["-ld", "True", "-lp", pval_path + '/' + load_name] if args["loadpath"] else []
                
                # if args["overwrite"] == "False":
                #     if os.path.isfile(pval_path + '/' + name): 
                #         continue
                
                ## example call:
                # python run_experiment_hopf.py
                #    --lr 0.00002 --pretrain --hopf_pretrain --hopf_pretrain_iters 10000 --hopf_loss_divisor 5 
                #    --hopf_loss_decay --hopf_loss_decay_w 0.9998

                command = ["python", args["trainfile"]] + info_args + poi_args + ["--seed", str(sdx)]
                # print(command)
                subprocess.run(command)

            # if args['plotting'] == "True":
            #     subprocess.run(["python3", "examine.py", "-pm", pval_path + '/', "-nt", args["nametag"], "-sf", "True", "-pf", scan_path + '/'])

        else:
            train_helper(pois, scales, pval_path, poi_args, depth + 1, num_seeds)

num_seeds = int(args["numberofseeds"])
train_helper(pois, scales, scan_path, [], 0, num_seeds)

print("\n He terminado.\n")
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))