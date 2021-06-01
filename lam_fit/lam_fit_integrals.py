import sys
sys.path.append("./")
sys.path.append("..")
from lib import *
import os 
import statistics
import torch
import torch.nn as nn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default = "GaussNet")
    parser.add_argument('--exp', type=str, default = None)
    parser.add_argument("--width", type=int, default = 100)
    parser.add_argument("--n-inputs", type = int, default = 6)
    parser.add_argument("--n-models", type = int, default = 10**3)
    parser.add_argument("--d-in", type = int, default = 1)
    parser.add_argument("--d-out", type = int, default = 1)
    parser.add_argument("--sb", type = float, default = 1.0)
    parser.add_argument("--sw", type = float, default = 1.0)
    parser.add_argument("--mb", type = float, default = 0.0)
    parser.add_argument("--mw", type = float, default = 0.0)
    parser.add_argument("--cuda", action = 'store_true', default = False)
    parser.add_argument("--n-pt", type = int, default = 4)
    parser.add_argument("--parallel", type = bool, default = False) # optional arument for parallel computation
    parser.add_argument("--cutoff", type = float, default = 100.0) # optional arument for parallel computation
    parser.add_argument("--inputset", type = int, default = 1) # optional arument for parallel computation
    parser.add_argument("--order", type = str, default = "0") # optional arument for parallel computation

    args = parser.parse_args()

    for order in ["0", "2", "220"]: # modify or remove this loop for computations of only some integrals
        for inputset in [1,2]: # for the test/train inputs
            args.inputset = inputset

            if args.activation == "Erf":
                if args.inputset == 1:
                    xs = (np.sqrt(2)/2)*0.01*torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
                    xset = "xsetEt"
                
                if args.inputset == 2:
                    xs = 0.01*torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
                    xset = "xsetE"

                widths = [5] # defines NGP
                cutoffs = [100]


            if args.activation == "GaussNet":
                if args.inputset == 1:
                    xs = (np.sqrt(2)/2)*0.01*torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
                    xset = "xsetGt"
                
                if args.inputset == 2:
                    xs = 0.01*torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
                    xset = "xsetG"

                widths = [1000] # defines NGP
                cutoffs = [np.inf]

            if args.activation == "ReLU":
                if args.inputset == 1:
                    xs = (np.sqrt(2)/2)*torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
                    xset = "xsetRt"
                
                if args.inputset == 2:
                    xs = torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
                    xset = "xsetR"

                widths = [20] # defines NGP
                cutoffs = [100]

                args.sb = 10**-100


            # optional arguments for parallel:
            # order = args.order
            # order = "0"
            # order = "2"
            # order = "220"
            args.n_inputs = len(xs)

            params = {"0": 1, "2": 2, "220": 3}
            lam_array = np.ndarray((params[order]))
            
            for cutoff in cutoffs:
                log_cutoff = np.round(np.log10(cutoff), 2)
                print("Computing at cutoff ", cutoff)

                # this just computes all the integrals beforehand for all (unique) tensor elements
                local0_integral = np.array([[[[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)])
                local2_integral = np.array([[[[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)]) # set to be the same shape array
                nonlocal_integral = np.array([[[[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)]) # set to be the same shape array

                path_nonloc = "nonlocal_integral_"+args.activation+"_"+xset+"_"+str(cutoff)+"_"+str(args.sb)+".pickle"
                path_loc0 = "loc0_integral_"+args.activation+"_"+xset+"_"+str(cutoff)+"_"+str(args.sb)+".pickle"
                path_loc2 = "loc2_integral_"+args.activation+"_"+xset+"_"+str(cutoff)+"_"+str(args.sb)+".pickle"
                computed = False
                computed = os.path.isfile(path_nonloc) and os.path.isfile(path_loc0) and os.path.isfile(path_loc2) # boolean - is the file here?

                if computed:
                    print("importing integrals")
                    local0_integral = pickle.load(open(path_loc0, "rb"))
                    local2_integral = pickle.load(open(path_loc2, "rb"))
                    nonlocal_integral = pickle.load(open(path_nonloc, "rb"))

                if not computed:
                    for i in range(args.n_inputs):
                        print("computing integrals for first tensor index ", i)
                        for j in range(args.n_inputs):
                            for k in range(args.n_inputs):
                                for l in range(args.n_inputs):
                                    if order == "0":
                                        local0_integral[i,j,k,l] = local0(xs[i], xs[j], xs[k], xs[l], cutoff, args)
                                        print(local0_integral[i,j,k,l])

                                    if order == "2":
                                        local2_integral[i,j,k,l] = local2(xs[i], xs[j], xs[k], xs[l], cutoff, args)
                                        print(local2_integral[i,j,k,l])
                                    if order == "220":
                                    
                                        print("computing nonlocal integral",i,j,k,l)
                                        nonlocal_integral[i,j,k,l] = nonlocal22(xs[i], xs[j], xs[k], xs[l], cutoff, args)
                                        print(nonlocal_integral[i,j,k,l])

                    # save nonlocal integral at this cutoff and activation and bias
                    if order == "0":
                        pickle.dump(local0_integral, open(path_loc0, "wb"))
                    if order == "2":
                        pickle.dump(local2_integral, open(path_loc2, "wb"))
                    if order == "220":
                        pickle.dump(nonlocal_integral, open(path_nonloc, "wb"))
                