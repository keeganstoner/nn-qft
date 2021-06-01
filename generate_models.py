import sys
sys.path.append("./")
sys.path.append("..")
from lib import *
import itertools
import numpy as np

def Print(*s):
    print(s)
    sys.stdout.flush()

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

    args = parser.parse_args()

    widths = [2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] # ten
    
    runs = 1 # runs per width, usually set to 10 or 1

    # # parallelize via MPI #
    # from mpi4py import MPI
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    # widths = [widths[rank]]

    if args.d_in == 1:
        if args.activation == "Erf":
            xs = torch.tensor([[-1],[-0.6],[-0.2],[0.2],[0.6], [1.0]])
            xset = "xset1"
        if args.activation == "GaussNet":
            xs = 0.01*torch.tensor([[-1],[-0.6],[-0.2],[0.2],[0.6], [1.0]])
            xset = "xset2"
        if args.activation == "ReLU":
            xs = torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
            xset = "xset1A"

    if args.d_in == 2:
        xs = torch.tensor([-1.0, 1.0])
        xs = torch.cartesian_prod(xs, xs)
        xset = "xset3"
        if args.activation == "GaussNet":
                xs = 0.01*xs
                xset = "xset4"
        if args.activation == "ReLU":
            xs = torch.tensor([0.5, 1.0])
            xs = torch.cartesian_prod(xs, xs)
            xset = "xset3A"

    if args.d_in == 3:
        xs = torch.tensor([[-1., -1., -1.],[ 1.,  1., -1.],[-1.,  1.,  1.],[ 1., -1.,  1.]])
        xset = "xset5"
        if args.activation == "GaussNet":
            xs = 0.01*xs
            xset = "xset6"
        if args.activation == "ReLU":
            xs = torch.tensor([[0.2, 0.2, 0.2],[ 1.,  1., 0.2],[0.2,  1.,  1.],[ 1., 0.2,  1.]])
            xset = "xset5A"
    
    args.n_inputs = len(xs)


    for args.width in widths:
        for run in range(runs):
            print("Generating networks for "+args.activation+" at width "+str(args.width), " - run ", run+1, " of ", runs)
            fss = create_networks(xs, args)
            #print("Pickling: "+args.activation+" at width "+str(args.width))
            pickle.dump(fss, open("run"+str(run+1)+"_din="+str(args.d_in)+"_"+args.activation+"_1e"+str(int(np.log10(args.n_models)))+"models_"+str(args.width)+"width_"+xset+".pickle",'wb'))