import sys
sys.path.append("./")
sys.path.append("..")
from lib import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default = "Erf")
    parser.add_argument('--exp', type=str, default = None)
    parser.add_argument("--width", type=int, default = 100)
    parser.add_argument("--n-inputs", type = int, default = 6)
    parser.add_argument("--n-models", type = int, default = 10**4)
    parser.add_argument("--d-in", type = int, default = 1)
    parser.add_argument("--d-out", type = int, default = 1)
    parser.add_argument("--sb", type = float, default = 1.0)
    parser.add_argument("--sw", type = float, default = 1.0)
    parser.add_argument("--mb", type = float, default = 0.0)
    parser.add_argument("--mw", type = float, default = 0.0)
    parser.add_argument("--cuda", action = 'store_true', default = False)
    parser.add_argument("--n-pt", type = int, default = 6)

    args = parser.parse_args()

    # this n is the npt function being computed in this script
    # it can be changes directly here, or above in the defaults for --n-pt
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

    if args.activation == "ReLU":
        args.sb == 10**-100

    args.n_inputs = len(xs)
    widths = [2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] #ten

    fss = {}    # dictionary for storing outputs after importing
                # keys are widths

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
    if n == 2:
        n_thy = kernel_tensor(xs, args)
    if n == 4:
        n_thy = four_pt_tensor(xs, args)
    if n == 6:
        n_thy = six_pt_tensor(xs, args)
        #store the GP 6pt function for later
        pickle.dump(n_thy, open("six_pt_tensor_"+args.activation+"_din"+str(args.d_in)+"_"+xset+".pickle",'wb')) 
    if n not in [2, 4, 6]:
        print("Not a 2, 4, or 6pt function")
        exit()

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

        n_diff = np.abs(np.nanmean(n_exp, axis = 0) - np.array(n_thy))/np.array(n_thy)

        if n == 6:
            pickle.dump(n_exp, open("six_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle",'wb')) 

        # computes elementwise standard deviation among chunks
        n_diff_std = np.nanstd(n_exp, axis = 0)/np.array(n_thy)
        n_diff = [i for i in n_diff.flatten().tolist() if (~np.isnan(i))]
        n_diff_std = [i for i in n_diff_std.flatten().tolist() if (~np.isnan(i))]

        n_diff_full.extend(n_diff)

        mean1 = np.mean(n_diff)
        background1 = np.mean(n_diff_std)
        backgrounds.append(background1)

        for i in range(len(n_diff)):
            widths_list.append(width)

    signal = sum(backgrounds)/len(backgrounds)
    backgrounds = []
    for i in range(len(n_diff_full)):
        backgrounds.append(signal)

    df = pd.DataFrame({"width": widths_list, "n_point": n_diff_full, "background": backgrounds})
    df['log10width'] = np.log10(df['width'])
    df['log10n_point'] = np.log10(df['n_point'])
    df['log10background'] = np.log10(df['background'])

    import seaborn as sns
    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import rc  

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    fsize = 24
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

    ###
    # plot!
    if args.activation == "GaussNet":
        act = "Gauss\\text{-}net"
    if args.activation == "Erf":
        act = "Erf\\text{-}net"
    if args.activation == "ReLU":
        act = "ReLU\\text{-}net"
    title_size, label_size, tick_size = fsize, fsize, fsize
    sns.set_style(style="darkgrid")
    sns.lineplot(data=df,x='log10width',y='log10n_point', label = lt(str(n)+"\\text{-}pt signal"))
    sns.lineplot(data=df,x='log10width',y='log10background', label = lt("background"))
    plt.tick_params(labelsize=tick_size)
    plt.title(lt(act+" "+str(n)+"\\text{-}pt Deviation, ")+lm("d_{in}="+str(args.d_in)),fontsize=title_size)
    plt.ylabel(lm("\\log_{10} m_{"+str(n)+"}"),fontsize=label_size)
    plt.xlabel(lm("\\log_{10} N"),fontsize=label_size)
    plt.tight_layout()
    b, t = plt.ylim() # discover the values for bottom and top
    b -= 0.01 # aesthetics
    t += 0.01
    plt.ylim(b, t) 
    plt.margins(0,0) # aesthetics
    #plt.savefig("gp_"+args.activation+str(n)+".pdf",bbox_inches='tight')
    plt.legend()
    plt.figure()
    plt.show()