import sys
sys.path.append("./")
sys.path.append("..")
from lib import *
import os 
import statistics
import torch
import torch.nn as nn

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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
    parser.add_argument("--parallel", type = bool, default = False)

    args = parser.parse_args()
    n = args.n_pt


    if args.activation == "Erf":
        width = 5
        cutoffs = [100]
        xset = "xsetE"


    if args.activation == "GaussNet":
        xset = "xsetG"
        width = 1000
        cutoffs = [np.inf]


    if args.activation == "ReLU":
        xset = "xsetR"
        width = 20
        cutoffs = [100]
        args.sb = 10**-100

    path_LHS_test = "four_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+".pickle"
    path_LHS_train = "four_pt_exp_"+args.activation+"_width"+str(width)+"_din"+str(args.d_in)+"_"+xset+"t.pickle"

    deltaG4test = np.nanmean(pickle.load(open(path_LHS_test, "rb")), axis = 0)
    deltaG4train = np.nanmean(pickle.load(open(path_LHS_train, "rb")), axis = 0)

    deltaG4train_flat = np.array([i for i in deltaG4train.flatten().tolist() if (~np.isnan(i))])
    deltaG4test_flat = np.array([i for i in deltaG4test.flatten().tolist() if (~np.isnan(i))])

    dG4 = torch.DoubleTensor(deltaG4train_flat)
    dG4_test = torch.DoubleTensor(deltaG4test_flat)

    
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

        path_nonloct = "nonlocal_integral_"+args.activation+"_"+xset+"t_"+str(cutoff)+"_"+str(args.sb)+".pickle"
        path_loc0t = "loc0_integral_"+args.activation+"_"+xset+"t_"+str(cutoff)+"_"+str(args.sb)+".pickle"
        path_loc2t = "loc2_integral_"+args.activation+"_"+xset+"t_"+str(cutoff)+"_"+str(args.sb)+".pickle"
        computed = os.path.isfile(path_nonloc) and os.path.isfile(path_loc0) and os.path.isfile(path_loc2)

        if computed:
            print("importing integrals")

            local0_integral = pickle.load(open(path_loc0, "rb"))
            local2_integral = pickle.load(open(path_loc2, "rb"))
            nonlocal_integral = pickle.load(open(path_nonloc, "rb"))

            local0_integralt = pickle.load(open(path_loc0t, "rb"))
            local2_integralt = pickle.load(open(path_loc2t, "rb"))
            nonlocal_integralt = pickle.load(open(path_nonloct, "rb"))

        if not computed:
            print("Please run lam_fit_integrals.py first.")
            quit()
            

        local0_integral_train = local0_integral
        local2_integral_train = local2_integral
        nonlocal_integral_train = nonlocal_integral
        
        # change this part because integrals are indexed differently
        local0_integral_test = local0_integralt
        local2_integral_test = local2_integralt
        nonlocal_integral_test = nonlocal_integralt


        local0_integral_flat = np.array([i for i in local0_integral_train.flatten().tolist() if (~np.isnan(i))])
        local2_integral_flat = np.array([i for i in local2_integral_train.flatten().tolist() if (~np.isnan(i))])
        nonlocal_integral_flat = np.array([i for i in nonlocal_integral_train.flatten().tolist() if (~np.isnan(i))])

        local0_integral_flat_test = np.array([i for i in local0_integral_test.flatten().tolist() if (~np.isnan(i))])
        local2_integral_flat_test = np.array([i for i in local2_integral_test.flatten().tolist() if (~np.isnan(i))])
        nonlocal_integral_flat_test = np.array([i for i in nonlocal_integral_test.flatten().tolist() if (~np.isnan(i))])


        T0, T2, TNL = torch.DoubleTensor(local0_integral_flat), torch.DoubleTensor(local2_integral_flat), torch.DoubleTensor(nonlocal_integral_flat)
        T0_test, T2_test, TNL_test = torch.DoubleTensor(local0_integral_flat_test), torch.DoubleTensor(local2_integral_flat_test), torch.DoubleTensor(nonlocal_integral_flat_test)

        l0, l2, lNL = torch.tensor(0.0,requires_grad=True), torch.tensor(0.0,requires_grad=True), torch.tensor(0.0,requires_grad=True)

        def MAPE_out(A,P):
            return torch.mean(100.0*torch.abs((A-P)/A)), torch.max(100.0*torch.abs((A-P)/A))
        def MAPE(A,P):
            return torch.mean(100.0*torch.abs((A-P)/A))

        # LEARNING RATE - needs to be played with
        lr = 1e-6


        optimizer = torch.optim.SGD([l0,l2,lNL],lr=lr)

        # can decide which criterion to optimize with
        criterion = nn.MSELoss()
        # criterion = MAPE

        def converged(recent_losses,thresh=1): # default within 1%
            m1, m2, m = max(recent_losses), min(recent_losses), sum(recent_losses)/len(recent_losses)
            if 100.0*(m1-m2)/m <= thresh:
                return True
            return False


        last_losses = []
        print("BEGIN MSE", criterion(dG4, l0*T0+ l2*T2 + lNL*TNL).detach().cpu().numpy())
        print("\n")
        for epoch in range(500000):
            #print(l0,l2)
            # loss = criterion(dG4, l0*T0) # this is the only time we have to change the lambda expressions
            # loss = criterion(dG4, l0*T0 + l2*T2)
            loss = criterion(dG4, l0*T0 + l2*T2 + lNL*TNL)
            if epoch%1000 == 0: print("epoch, l0, l2, lNL, loss",epoch,float(l0.detach().cpu().numpy()),float(l2.detach().cpu().numpy()),float(lNL.detach().cpu().numpy()),float(loss.detach().cpu().numpy()))
            # if epoch%20 == 19: quit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if len(last_losses) <= 50:
                last_losses.append(float(loss.detach().cpu().numpy()))
            else:
                last_losses = last_losses[1:]
                last_losses.append(float(loss.detach().cpu().numpy()))
                if converged(last_losses,10**-9):
                   print("CONVERGED EARLY: ", epoch, float(l0.detach().cpu().numpy()),float(l2.detach().cpu().numpy()),float(lNL.detach().cpu().numpy()),float(loss.detach().cpu().numpy()))
                   break

            
        print("epoch",epoch, "\nl0 = ", float(l0.detach().cpu().numpy()),"  \nl2 = ", float(l2.detach().cpu().numpy()), "  \nlNL = ", float(lNL.detach().cpu().numpy()))
        # print(max(last_losses)-min(last_losses))
        print("\nMAPE: ", MAPE_out(dG4, l0*T0+ l2*T2 + lNL*TNL)[0].item(), "%  \nMAX APE: ", MAPE_out(dG4, l0*T0+ l2*T2 + lNL*TNL)[1].item(), "% \nMSE: ", criterion(dG4, l0*T0+ l2*T2 + lNL*TNL).item())

        print("\ntestset MAPE: ", MAPE_out(dG4_test,l0*T0_test + l2*T2_test + lNL*TNL_test )[0].item(), "%  \ntestset MAX APE: ", MAPE_out(dG4_test,l0*T0_test + l2*T2_test + lNL*TNL_test )[1].item(), "% \ntestset MSE: ", criterion(dG4_test,l0*T0_test + l2*T2_test + lNL*TNL_test ).item())
        print("\nlr = ", lr)
