import sys
sys.path.append("./")
sys.path.append("..")
from lib import *
import itertools
import numpy as np

import seaborn as sns
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default = "ReLU")    # changes. should be provided
    parser.add_argument('--exp', type=str, default = None)                 # does not change
    parser.add_argument("--width", type=int, default = 100)                # changes. should be provided
    parser.add_argument("--n-inputs", type = int, default = 6)             # does not change
    parser.add_argument("--n-models", type = int, default = 10**3)         # each run has 10^6 models
    parser.add_argument("--d-in", type = int, default = 1)                 # does not change
    parser.add_argument("--d-out", type = int, default = 1)                # does not change
    parser.add_argument("--sb", type = float, default = 1.0)               # changes. 1 for Gauss, Erf, 0 for ReLU
    parser.add_argument("--sw", type = float, default = 1.0)               # does not change
    parser.add_argument("--mb", type = float, default = 0.0)               # does not change
    parser.add_argument("--mw", type = float, default = 0.0)               # does not change
    parser.add_argument("--cuda", action = 'store_true', default = False)  # does not change
    parser.add_argument("--n-pt", type = int, default = 6)

    args = parser.parse_args()
    
    runs = 1
    n = args.n_pt
    
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
            args.sb = 10**-100

    args.n_inputs = len(xs)
    widths = [2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] #ten
    xslist = xs.tolist()


    if path.exists("six_pt_tensor_"+args.activation+"_din"+str(args.d_in)+"_"+xset+".pickle"):
        six_pt_thy = np.array(pickle.load(open("six_pt_tensor_"+args.activation+"_din"+str(args.d_in)+"_"+xset+".pickle",'rb')))
    else:
        six_pt_thy = np.array(six_pt_tensor(xs, args))
    
    if path.exists("four_pt_tensor_"+args.activation+"_din"+str(args.d_in)+"_"+xset+".pickle"):
        four_pt_thy = np.array(pickle.load(open("four_pt_tensor_"+args.activation+"_din"+str(args.d_in)+"_"+xset+".pickle",'rb')))
    else:
        four_pt_thy = np.array(four_pt_tensor(xs, args))


    six_diff_full = []                             # list for plotting 6pt connected piece to see 1/N^2 dependence
    backgrounds = []
    widths_list = []
    background_per_width = []                      # normalized width-dependent background
    six_diff_full_unnormalized = []                # list for plotting unnormalized 6pt connected to see 1/N^2 dependence
    background_per_width_unnormalized = []         # unnormalized width-dependent background


    for i in range(len(widths)) :
        args.width = widths[i]
        width = widths[i]

        six_pt_expt_list = pickle.load(open("six_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle",'rb')) 
        four_pt_expt_list = pickle.load(open("four_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle",'rb')) 


        # 6pt connected term, according to QFT definition : G^6 + 15 combination(2*G2*G2*G2 - G4*G2) = G^6 + 2*G^6_GP - 15 combo G4*G2
        sixptdev_O1 = np.nanmean(six_pt_expt_list, axis = 0) + 2*np.array(six_pt_thy)

        fourptexpt = np.nanmean(four_pt_expt_list, axis = 0)              # with modified definition of 6-pt connected from this is what we need
        
        six_pt_fluctuations = np.nanstd(six_pt_expt_list, axis = 0)       # an array with elementwise 6pt fluctuations at tree level
        four_pt_fluctuations = np.nanstd(four_pt_expt_list, axis = 0)     # an array with elementwise 4pt fluctuations at tree level
        
        six_pt_fluctuations_O1 = [[[[[[0. for q in range(len(xs))] for j in range(len(xs))] for k in range(len(xs))] for l in range(len(xs))] for m in range(len(xs))] for p in range(len(xs))]                      # an array to contain elementwise fluctuation to 6pt at O(1/N)

        sixptdev_O2 = []
        sixptdev_O2_unnormalized = []
        print("beginning 6pt calculation")
        for x1 in xs:
            for x2 in xs:
                for x3 in xs:
                    for x4 in xs:
                        for x5 in xs:
                            for x6 in xs:
                                G4G2_contribution = (fourptexpt[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])])*K_int(x5, x6, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x5])])*K_int(x4, x6, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x2])][xslist.index([x5])][xslist.index([x4])])*K_int(x3, x6, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x5])][xslist.index([x3])][xslist.index([x4])])*K_int(x2, x6, args)+ (fourptexpt[xslist.index([x5])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])])*K_int(x1, x6, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x6])])*K_int(x5, x4, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x2])][xslist.index([x6])][xslist.index([x4])])*K_int(x5, x3, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x6])][xslist.index([x3])][xslist.index([x4])])*K_int(x5, x2, args)+ (fourptexpt[xslist.index([x6])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])])*K_int(x5, x1, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x2])][xslist.index([x5])][xslist.index([x6])])*K_int(x4, x3, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x5])][xslist.index([x3])][xslist.index([x6])])*K_int(x4, x2, args)+ (fourptexpt[xslist.index([x5])][xslist.index([x2])][xslist.index([x3])][xslist.index([x6])])*K_int(x4, x1, args)+ (fourptexpt[xslist.index([x1])][xslist.index([x5])][xslist.index([x6])][xslist.index([x4])])*K_int(x3, x2, args)+ (fourptexpt[xslist.index([x5])][xslist.index([x2])][xslist.index([x6])][xslist.index([x4])])*K_int(x3, x1, args)+ (fourptexpt[xslist.index([x5])][xslist.index([x6])][xslist.index([x3])][xslist.index([x4])])*K_int(x2, x1, args)

                                a = sixptdev_O1[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])][xslist.index([x5])][xslist.index([x6])] - G4G2_contribution  # this causes G^6 + 2*G2*G2*G2 - G4*G2
    
                                b = six_pt_thy[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])][xslist.index([x5])][xslist.index([x6])]
                                if (~np.isnan(a)) and (~np.isnan(b)):
                                    sixptdev_O2.append(np.abs(a/b))
                                    sixptdev_O2_unnormalized.append(np.abs(a))

                                six_pt_fluctuations_O1[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])][xslist.index([x5])][xslist.index([x6])] = (four_pt_fluctuations[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])])*K_int(x5, x6, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x5])])*K_int(x4, x6, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x2])][xslist.index([x5])][xslist.index([x4])])*K_int(x3, x6, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x5])][xslist.index([x3])][xslist.index([x4])])*K_int(x2, x6, args)+ (four_pt_fluctuations[xslist.index([x5])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])])*K_int(x1, x6, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x2])][xslist.index([x3])][xslist.index([x6])])*K_int(x5, x4, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x2])][xslist.index([x6])][xslist.index([x4])])*K_int(x5, x3, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x6])][xslist.index([x3])][xslist.index([x4])])*K_int(x5, x2, args)+ (four_pt_fluctuations[xslist.index([x6])][xslist.index([x2])][xslist.index([x3])][xslist.index([x4])])*K_int(x5, x1, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x2])][xslist.index([x5])][xslist.index([x6])])*K_int(x4, x3, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x5])][xslist.index([x3])][xslist.index([x6])])*K_int(x4, x2, args)+ (four_pt_fluctuations[xslist.index([x5])][xslist.index([x2])][xslist.index([x3])][xslist.index([x6])])*K_int(x4, x1, args)+ (four_pt_fluctuations[xslist.index([x1])][xslist.index([x5])][xslist.index([x6])][xslist.index([x4])])*K_int(x3, x2, args)+ (four_pt_fluctuations[xslist.index([x5])][xslist.index([x2])][xslist.index([x6])][xslist.index([x4])])*K_int(x3, x1, args)+ (four_pt_fluctuations[xslist.index([x5])][xslist.index([x6])][xslist.index([x3])][xslist.index([x4])])*K_int(x2, x1, args)


    
        six_diff_full.extend(sixptdev_O2)
        six_diff_full_unnormalized.extend(sixptdev_O2_unnormalized)
        
        # this is statistical fluctuations in 6-pt connected at O(1/N^2) : STD( G^6 + 2G^6_GP + 15* 24*lambda * [XI] diagram ) = STD( G^6 + 2*G^6_GP - 15*(G^4*K) = STD(G^6) + 2*STD(G^6_GP) + + STD(15*G^4*K) = STD(G^6) + STD(15*G^4*K). So after normalization, six_diff_STD = (STD(G^6) + STD(G^4)*15*K ) / G^6_GP

        six_diff_std = (six_pt_fluctuations + six_pt_fluctuations_O1)/np.array(six_pt_thy)   # statistical fluctuations in O(1/N^2) 6-pt connected
        six_diff_std = [i for i in six_diff_std.flatten().tolist() if (~np.isnan(i))]        # remove zeros and turn into a list to back background data
        backgrounds.append(np.mean(six_diff_std))
        
        six_diff_std_unnormalized = (six_pt_fluctuations + six_pt_fluctuations_O1)           # statistical fluctuations in unnormalized O(1/N^2) 6-pt connected
        six_diff_std_unnormalized = [i for i in six_diff_std_unnormalized.flatten().tolist() if (~np.isnan(i))]        # remove zeros and turn into a list to back background data
        
        for i in range(len(sixptdev_O2)):
            widths_list.append(width)
            background_per_width.append(np.mean(six_diff_std))
            background_per_width_unnormalized.append(np.mean(six_diff_std_unnormalized))
    
    
    signal = sum(backgrounds)/len(backgrounds)
    backgrounds = []
    for i in range(len(six_diff_full)):
        backgrounds.append(signal)

    print("save dataframe")
    assert(len(widths_list) == len(six_diff_full))
    assert(len(widths_list) ==len(backgrounds))
    assert(len(widths_list) ==len(background_per_width))

    dataframe_tosave = np.zeros((6,len(widths_list)))                   # save this dataframe for plotting locally on my computer
    dataframe_tosave[0,:] = widths_list                                 # first row is the list of widths as in panda dataframe
    dataframe_tosave[1,:] = six_diff_full                               # second row is the six pt connected at O(1/N^2)
    dataframe_tosave[2,:] = backgrounds                                 # third row is the background level
    dataframe_tosave[3,:] = background_per_width                        # fourth row is background which is not averaged over width
    dataframe_tosave[4,:] = six_diff_full_unnormalized                  # fifth row is the six pt connected at O(1/N^2)
    dataframe_tosave[5,:] = background_per_width_unnormalized           # sixth row is unnormalized background which is not averaged over width

    pickle.dump(dataframe_tosave, open("6pt_connected.pickle",'wb'))

    print("onto plots")

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


    df = pd.DataFrame({"width": widths_list, "6_point_dev": six_diff_full, "background": backgrounds},dtype=float)
    df['log10width'] = np.log10(df['width'])
    df['log106_point_dev'] = np.log10(df['6_point_dev'])    # log of 6pt deviation at O(1/N^2)
    df['log10background'] = np.log10(df['background'])      # log of widths
    
    z = np.polyfit(df['log10width'], df['log106_point_dev'], 1)
    p = np.poly1d(z)
    trendline_eq = str(p)
    print(trendline_eq, activation)
    
    if activation == "GaussNet":
        act = "Gauss\\text{-}net"
    if activation == "Erf":
        act = "Erf\\text{-}net"
    if activation == "ReLU":
        act = "ReLU\\text{-}net"
        
    title_size, label_size, tick_size = fsize, fsize, fsize
    sns.set_style(style="darkgrid")
    plt.figure()
    sns.lineplot(data=df,x='log10width',y='log106_point_dev', label = lt(str(6)+"\\text{-}pt signal"))
    sns.lineplot(data=df,x='log10width',y='log10background', label = lt("background"))
    plt.plot(df['log10width'],p(df['log10width']),linestyle=':')
    plt.legend()
    
    plt.tick_params(labelsize=tick_size)
    plt.title(lt(act+" "+str(6)+"\\text{-}pt Deviation, ")+lm("d_{in}=1"),fontsize=title_size)
    plt.ylabel(lm("\\log_{10} m_{"+str(6)+"}"),fontsize=label_size)
    plt.xlabel(lm("\\log_{10} N"),fontsize=label_size)
    plt.tight_layout()
    b, t = plt.ylim() # discover the values for bottom and top
    b -= 0.01 # aesthetics
    t += 0.01
    plt.ylim(b, t)
    plt.margins(0,0) # aesthetics
    plt.savefig(path + "sixpt_"+activation+"_6ptdev_v2.pdf",bbox_inches='tight')
    plt.legend()
    plt.show()



