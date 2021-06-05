import json
import sys

# check that base file was given and subsequently load it
try:
    basefile = sys.argv[1]
except:
    print("base model should be given as an argument")
    exit()

with open(basefile, "r") as f:
    basemodel = json.load(f)

# receiver configurations
rec_depths = [[0.05], [1.95], [0.05, 1.95]]
num_receivers = [11, 21, 31, 51, 76, 101]

# set parameters
for depth in rec_depths:
    for num in num_receivers:
        print(f"depth = {depth}, num = {num}")

        # check depth of receivers
        if depth == [0.05]:
            depths = "l"
        elif depth == [1.95]:
            depths = "r"
        elif depth == [0.05, 1.95]:
            depths = "l_and_r"

        # set number of receivers and depth
        basemodel["acquisition"]["num_receivers"] = num
        basemodel["acquisition"]["rec_depths"] = depth

        # basename
        name = "depth_" + depths + "_num_" + str(num)
        # define output directory
        basemodel["output"] = {"outdir": "results/ls_ii_" + name}
        # set output files
        basemodel["data"] = {"shots": "shots/ls_ii_" + name,
                             "initfile": None,
                             "pic": "pv_ls_ii_" + name + ".png",
                             "resultfile": "vp_ls_ii_" + name + ".hdf5",
                             "fobj": "fobj_" + name + ".npy"}

        # save json file
        with open("ls_ii_" + name + ".json", "w") as f:
            json.dump(basemodel, f, indent=4) 
