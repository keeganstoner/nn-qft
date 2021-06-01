import sys
sys.path.append("./")
sys.path.append("..")
from lib import *
import statistics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default = "ReLU")
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
    parser.add_argument("--n-pt", type = int, default = 6)

    args = parser.parse_args()
    n = args.n_pt

    runs = 1

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
            args.sb == 10**-100
        
    args.n_inputs = len(xs)

    if args.activation == "ReLU":
        width = 20
    if args.activation == "GaussNet":
        width = 100
    if args.activation == "Erf":
        width = 5
    args.width = width


    fss, fss_chunk = {}, {}
    print("Unpickling width "+str(width))
    # args.width = width
    for run in range(runs):
        with open("run"+str(run+1)+"_din="+str(args.d_in)+"_"+args.activation+"_1e"+str(int(np.log10(args.n_models)))+"models_"+str(args.width)+"width_"+xset+".pickle", 'rb') as handle:
            if run == 0:
                fss[width] = pickle.load(handle)
            else:
                fss[width] = torch.cat((fss[width], pickle.load(handle)))
    
    # chunk
    k = 10
    chunk = args.n_models*runs//k
 
    # GP 6-pt function
    if path.exists("six_pt_tensor_"+args.activation+"_din"+str(args.d_in)+"_"+xset+".pickle"):
        n_thy = pickle.load(open("six_pt_tensor_"+args.activation+"_din"+str(args.d_in)+"_"+xset+".pickle",'rb'))   
        GP_array = np.array(n_thy)
    else:
        GP_array = np.array(six_pt_tensor(xs, args))

    # experimental 6-pt function
    n_exp, n_exp4 = [0. for _ in range(k)], [0. for _ in range(k)]

    # GP 6-pt function
    if path.exists("six_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle"):
        exp_array = pickle.load(open("six_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle",'rb'))   
        exp_array = np.array(exp_array)
    else:
        for chunk_num in range(k):
            fss_chunk[width] = fss[width].narrow_copy(0,chunk_num*chunk,chunk)

            n_tensor = torch.mean(n_point(fss_chunk[width], n), dim=0)
            n_tensor4 = torch.mean(n_point(fss_chunk[width], 4), dim=0)
            # only true when d_out = 1:
            assert(args.d_out == 1)
            n_tensor = n_tensor.view(n_tensor.shape[0:n])
            n_exp[chunk_num] = n_tensor.tolist()
            n_exp4[chunk_num] = n_tensor4.tolist()

            if run == 4:
                print("done with run ", run+1)

        exp_array_full = np.nanmean(n_exp, axis = 0)
        exp_array = trim_sym_tensor(exp_array_full, args)
        expt4 = trim_sym_tensor4(np.nanmean(n_exp4, axis = 0), args)

    if args.activation == "ReLU":
        cutoffs = [7, 10, 15, 20, 30, 40, 50, 70, 100, 200, 500, 1000, 2000, 5000, 7000, 10000, 20000, 40000, 60000, 80000, 100000] #21
    
    if args.activation == "GaussNet" or args.activation == "Erf":
        cutoffs = [np.inf]

    logcut, GP_list, GP_lam_list = [], [], []

    GP = GP_array/exp_array

    fourptcontribution = [[[[[[0. for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))]
    xslist = xs.tolist()


    def calc_lam(xs):  
        denom = [[[[np.nan for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))]   
        denom = np.array(denom)
        numerator = np.array(four_pt_tensor(xs, args)) - expt4
        for i in range(len(xs)):
            for j in range(i, len(xs)):
                for k in range(j, len(xs)):
                    for l in range(k, len(xs)):
                        denom[i][j][k][l] = four_pt_int(xs[i], xs[j], xs[k], xs[l], cutoff, args)
        denom = np.array(denom)
        lam = (numerator/denom)
        print("cutoff: ", cutoff, "stdev in lam = ", np.nanstd(lam))
        return lam


    # # parallelize the cutoff integrals
    # from mpi4py import MPI
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    # cutoffs = [cutoffs[rank]]


    for cutoff in cutoffs:
        print("Running cutoff ", cutoff)
        for x1 in xs:
            for x2 in xs[1:]:
                for x3 in xs[2:]:
                    for x4 in xs[3:]:
                        for x5 in xs[4:]:
                            for x6 in xs[5:]:
                                num = 0.
                                # all possible 6pt diagrams containing a 4pt vertex (lambda)
                                num += intkappa(x5, x6, x3, x4, x1, x2, cutoff, args)
                                num += intkappa(x6, x2, x5, x5, x1, x3, cutoff, args)
                                num += intkappa(x6, x2, x5, x3, x1, x4, cutoff, args)
                                num += intkappa(x6, x2, x3, x4, x1, x5, cutoff, args)
                                num += intkappa(x5, x2, x3, x4, x1, x6, cutoff, args)
                                num += intkappa(x1, x4, x6, x5, x2, x3, cutoff, args)
                                num += intkappa(x1, x6, x5, x3, x2, x4, cutoff, args)
                                num += intkappa(x1, x3, x6, x4, x2, x5, cutoff, args)
                                num += intkappa(x1, x5, x3, x4, x2, x6, cutoff, args)
                                num += intkappa(x1, x2, x5, x6, x3, x4, cutoff, args)
                                num += intkappa(x1, x2, x4, x6, x3, x5, cutoff, args)
                                num += intkappa(x4, x2, x5, x1, x3, x6, cutoff, args)
                                num += intkappa(x2, x1, x3, x6, x4, x5, cutoff, args)
                                num += intkappa(x5, x2, x1, x3, x4, x6, cutoff, args)
                                num += intkappa(x4, x2, x1, x3, x5, x6, cutoff, args)
                                # total lambda contribution to the 6pt function
                                fourptcontribution[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])][xslist.index([x5])][xslist.index([x6])] = num
        if path.exists("lam_"+str(cutoff)+"_"+args.activation+"_width_"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle"):
            lambda_tensor = pickle.load(open("lam_"+str(cutoff)+"_"+args.activation+"_width_"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle",'rb'))
            print("Loaded lambda tensor from rg analysis")
        else: 
            lambda_tensor = calc_lam(xs)

        lambda_average = np.nanmean(lambda_tensor)
        # computation of lambda term
        lambda_term = 24.*lambda_average*np.array(fourptcontribution)
        # normalized GP + lambda theory prediction
        GP_lam = (GP_array + lambda_term)/exp_array

        print(cutoff, "kernel = ", np.nanmean(GP_array))
        print("lam term ", np.nanmean(lambda_term))
        print("exp array = ", np.nanmean(exp_array))
        print("delta = ", np.nanmean(GP_lam))

        # flatten everything for pandas
        GPflat = [i for i in GP.flatten().tolist() if (~np.isnan(i))]
        GP_lamflat = [i for i in GP_lam.flatten().tolist() if (~np.isnan(i))]
        GP_list.extend(GPflat)
        GP_lam_list.extend(GP_lamflat)
        for i in range(len(GP_lamflat)):
            logcut.append(np.round(np.log10(cutoff), 2))

    import pandas as pd
    df = pd.DataFrame({"log_cutoff": logcut})
    df["GP"] = GP_list
    df["GP_lam"] = GP_lam_list


    import seaborn as sns
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib import rc  

    #pickle.dump(df, open("delta_add_"+args.activation+".pickle", "wb"))
    #df = pickle.load(open("delta_add_"+args.activation+".pickle", "rb"))

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    fsize = 22
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('figure', titlesize=fsize)  # fontsize of the figure title

    sns.set_style("ticks", {"xtick.major.size":18,
    "ytick.major.size":18})
    def lt(s):
        return (r'$\mathrm{' + s + r'}$').replace(" ", "\,\,")

    def lm(s):
        return r'$' + s + r'$'

    if args.activation == "GaussNet":
        act = "Gauss\\text{-}net"
    if args.activation == "Erf":
        act = "Erf\\text{-}net"
    if args.activation == "ReLU":
        act = "ReLU\\text{-}net"
    title_size, label_size, tick_size = fsize, fsize, fsize
    sns.set_style(style="darkgrid")
    sns.lineplot(x='log_cutoff',y='GP', data=df, label = lm("G^{(6)}_{GP}/G^{(6)}"))
    sns.lineplot(x='log_cutoff',y='GP_lam', data=df, label = lm("(G^{(6)}_{GP} + \\bar\\lambda")+lt(" contribution)")+lm("/G^{(6)}"))
    plt.tick_params(labelsize=tick_size)
    plt.title(lt(act+" ")+lm("G^{(6)}")+lt(" prediction, ")+lm("N="+str(width)),fontsize=title_size)
    plt.ylabel(lt("Normalized contributions to ")+lm("G^{6}"),fontsize=0.8*label_size)
    plt.ylim(0, 1.2)
    plt.xlabel(lm("\\log_{10}\\Lambda"),fontsize=label_size)
    plt.tight_layout()
    plt.savefig("six_pt_pred_"+args.activation+".pdf",bbox_inches='tight')
    plt.legend()
    plt.figure()
    # plt.show()