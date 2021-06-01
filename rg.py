import sys
sys.path.append("./")
sys.path.append("..")
from lib import *

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

    args = parser.parse_args()
    n = 4

    runs = 1
    if args.d_in == 1:
        if args.activation == "ReLU":
            xs = torch.tensor([[0.2],[0.4],[0.6],[0.8],[1.0],[1.2]])
            xset = "xset1A"

    if args.d_in == 2:
        if args.activation == "ReLU":
            xs = torch.tensor([0.5, 1.0])
            xs = torch.cartesian_prod(xs, xs)
            xset = "xset3A"

    if args.d_in == 3:
        if args.activation == "ReLU":
            xs = torch.tensor([[0.2, 0.2, 0.2],[ 1.,  1., 0.2],[0.2,  1.,  1.],[ 1., 0.2,  1.]])
            xset = "xset5A"

    args.sb == 10**-100
    args.n_inputs = len(xs)
    width = 20 # defines NGP 
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

    # # #
    n_thy = four_pt_tensor(xs, args)
    # # #

    n_exp = [0. for _ in range(10)]
    
    for chunk_num in range(10):
        fss_chunk[width] = fss[width].narrow_copy(0,chunk_num*chunk,chunk)

        n_tensor = torch.mean(n_point(fss_chunk[width], n), dim=0)

        # only true when d_out = 1:
        assert(args.d_out == 1)
        n_tensor = n_tensor.view(n_tensor.shape[0:n])
        n_exp[chunk_num] = n_tensor.tolist()

        if run == 4:
            print("done with run ", run+1)

    numerator = np.nanmean(n_exp, axis = 0) - np.array(n_thy)

    print(np.nanmean(np.nanmean(n_exp, axis = 0)))
    print(np.nanmean(np.array(n_thy)))


    xslist = xs.tolist()
    logcut, loglamlist = [], []
    denom = [[[[np.nan for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))] for i in range(len(xs))]
    lambda_averages = {}

    
    if args.activation == "ReLU":
        cutoffs = [7, 10, 15, 20, 30, 40, 50, 70, 100, 200, 500, 1000, 2000, 5000, 7000, 10000, 20000, 40000, 60000, 80000, 100000] #21
    
    if args.activation == "GaussNet" or args.activation == "Erf":
        cutoffs = [np.inf]

    for cutoff in cutoffs:
        log_cutoff = np.inf
        if args.activation == "ReLU":
            log_cutoff = np.round(np.log10(cutoff), 2)
        print("Computing with cutoff ", cutoff)
        for i in range(len(xs)):
            for j in range(i, len(xs)):
                for k in range(j, len(xs)):
                    for l in range(k, len(xs)):
                        denom[i][j][k][l] = four_pt_int(xs[i], xs[j], xs[k], xs[l], cutoff, args)
        denom = np.array(denom)
        lam = (numerator/denom)
        print("cutoff: ", cutoff, "stdev in lam = ", np.nanstd(lam))
        pickle.dump(lam, open("lam_"+str(cutoff)+"_"+args.activation+"_width_"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle",'wb'))
        lambda_average = np.nanmean(lam)
        log_lam = [np.log10(np.abs(i)) for i in lam.flatten().tolist() if (~np.isnan(i))]
        loglamlist.extend(log_lam)
        for _ in range(len(log_lam)):
            logcut.append(log_cutoff)


    print(str(datetime.datetime.now()))

    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib import rc  
    from scipy import stats

    df_lam = pd.DataFrame({"log_lambda": loglamlist, "log_cutoff": logcut})
    
    if args.activation == "ReLU":
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_lam['log_cutoff'],df_lam['log_lambda'])


    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    fsize = 22
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
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
    # # #
    # plot!
    # # #
    title_size, label_size, tick_size = fsize, fsize, fsize
    sns.set_style(style="darkgrid")
    sns.regplot(x='log_cutoff',y='log_lambda', data=df_lam, line_kws={'label':lt("slope="+str(np.round(slope, 3))+", ")+lm("R^{2}="+str(np.round(r_value**2, 5)))})
    plt.tick_params(labelsize=tick_size)
    plt.title(lt(act+" \\lambda_{m}, ")+lm("N="+str(width))+lt(" with ")+lm("d_{in}="+str(args.d_in)),fontsize=title_size)
    plt.ylabel(lm("\\log_{10}\\lambda_{m}"),fontsize=label_size)
    plt.xlabel(lm("\\log_{10}\\Lambda"),fontsize=label_size)
    plt.tight_layout()
    plt.legend()
    import datetime
    plt.savefig("rg_"+args.activation+"_din"+str(args.d_in)+".pdf",bbox_inches='tight')
    plt.figure()
    # plt.show()