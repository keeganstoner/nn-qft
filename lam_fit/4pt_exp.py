import sys
sys.path.append("./")
sys.path.append("..")
from lib import *

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
    parser.add_argument("--inputset", type = int, default = 1)

    args = parser.parse_args()
    n = args.n_pt

    runs = 10

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

        args.n_inputs = len(xs)

        for args.width in widths:
            for run in range(runs):
                print("Generating networks for "+args.activation+" with xset ", xset, " at width "+str(args.width), " - run ", run+1, " of ", runs)
                fss = create_networks(xs, args)
                #print("Pickling: "+args.activation+" at width "+str(args.width))
                pickle.dump(fss, open("run"+str(run+1)+"_din="+str(args.d_in)+"_"+args.activation+"_1e"+str(int(np.log10(args.n_models)))+"models_"+str(args.width)+"width_"+xset+".pickle",'wb'))


        fss = {}    # dictionary for storing outputs after importing
                    # keys are widths

        print("Computing npt functions with activation ", args.activation, " and xset ", xset)
        for width in widths:
            print("Unpickling width "+str(width))
            args.width = width
            for run in range(runs):
                with open("run"+str(run+1)+"_din="+str(args.d_in)+"_"+args.activation+"_1e"+str(int(np.log10(args.n_models)))+"models_"+str(args.width)+"width_"+xset+".pickle",'rb') as handle:
                    if run == 0:
                        fss[width] = pickle.load(handle)
                    else:
                        fss[width] = torch.cat((fss[width], pickle.load(handle)))
        

        print("Computing "+str(n)+"-pt function for activation "+args.activation)
        if n == 4:
            n_thy = four_pt_tensor(xs, args)

        fss_chunk = {}
        # split data into k chunks so background level can be plotted
        k = 10
        chunk = len(fss[widths[0]])//k
        print("Models in each chunk: ", chunk)

        widths_list, n_diff_full, backgrounds, n_exp = [], [], [], [0. for _ in range(10)]
        for width in widths:
            for chunk_num in range(10):
                # this is a dictionary (with keys = widths) for a single chunk
                fss_chunk[width] = fss[width].narrow_copy(0,chunk_num*chunk,chunk)

                # computes the experimental n-pt function and averages over models elementwise
                n_tensor = torch.mean(n_point(fss_chunk[width], n), dim=0)
                assert(args.d_out == 1) # this code is written for d_out = 1
                n_tensor = n_tensor.view(n_tensor.shape[0:n])
                n_exp[chunk_num] = n_tensor.tolist()

            n_diff = np.nanmean(n_exp, axis = 0) - np.array(n_thy)
            pickle.dump(n_exp, open("four_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle",'wb'))


