import torch
from torch import nn
import numpy as np
import math
from scipy import special
from scipy import integrate
import pickle
import datetime
import argparse
import os
import pandas as pd
import datetime
import os.path
from os import path
os.environ['KMP_DUPLICATE_LIB_OK']='True' # solves strange Mac bug

import sys
def Print(*s):
  print(s)
  sys.stdout.flush()

import datetime

# # # # # # # #
# experiments #
# # # # # # # # 
        
def activation(args):
    if args.activation == "Erf":
        return Erf_nobackprop()
    elif args.activation == "ReLU":
        return nn.ReLU()

class Erf_nobackprop(nn.Module):
    def forward(self,x):
        n = x.detach().numpy()
        return torch.tensor(special.erf(n))

class expln(nn.Module):
    def forward(self,x):
        x = torch.exp(x)
        mean, std = torch.mean(x,dim=1), torch.std(x,dim=1)
        mean, std = mean.view(mean.shape[0],1), std.view(std.shape[0],1)
        x = (x-mean)/std
        return x

class GaussNet(nn.Module):
    def __init__(self,args):
        super(GaussNet, self).__init__()
        self.args = args
        
        # create and initialize input layer
        self.input = nn.Linear(args.d_in,args.width)
        torch.nn.init.normal_(self.input.weight,mean=args.mw,std=args.sw/math.sqrt(self.input.in_features))
        torch.nn.init.normal_(self.input.bias,mean=args.mb,std=args.sb)
        
        # create and initialize output layer
        self.output = nn.Linear(args.width,args.d_out)
        torch.nn.init.normal_(self.output.weight,mean=args.mw,std=args.sw/math.sqrt(self.output.in_features))
        torch.nn.init.normal_(self.output.bias,mean=args.mb,std=args.sb)
        
    def forward(self,x):
        z = self.input(x)
        ez = torch.exp(z)
        # norm = torch.exp((4*args.sb**2+4*args.sw**2*torch.norm(x,dim=1)**2)/2.0)
        # fix args issue above
        norm = torch.exp((4+4*torch.norm(x,dim=1)**2)/(2.0*self.args.d_in))
        norm = norm.view(norm.shape[0],1)
        norm = torch.sqrt(norm)
        ezonorm = ez / norm
        return self.output(ezonorm)


def init_weights(m, args):
    torch.nn.init.normal_(m.weight,mean=0,std=args.sw/math.sqrt(m.in_features)) # Schoenholz et al conventions
    torch.nn.init.normal_(m.bias,mean=0,std=args.sb)

def create_networks(xs, args):
    #Print(args.width)
    fss = None
    for i in range(args.n_models):
        #if i % (5*10**5) == 0:
        #    Print("Finished ", i, "models", datetime.datetime.now())
        if args.activation == "GaussNet":
            model = GaussNet(args)
        if args.activation == "Erf":
            with torch.no_grad():
                model = nn.Sequential(
                    nn.Linear(args.d_in,args.width),
                    activation(args),
                    nn.Linear(args.width,args.d_out)
                )
                model.apply(lambda m: init_weights(m, args) if type(m) == nn.Linear else None)
        if args.activation == "ReLU":
            model = nn.Sequential(
                nn.Linear(args.d_in,args.width),
                activation(args),
                nn.Linear(args.width,args.d_out)
        )
            model.apply(lambda m: init_weights(m, args) if type(m) == nn.Linear else None)
            #print(list(model.parameters()))
        fs = model(xs)
        fs = fs.view(1,fs.shape[0],fs.shape[1])

        if fss is None:
            fss = fs 
        else:
            fss = torch.cat((fss,fs))
    return fss

# n-point functions from network outputs

def n_point(fss,n):
    num_nets, n_inputs, d_out = list(fss.shape)
    shape = [num_nets,n_inputs]
    while(len(shape)) < n+1:
        shape.append(1)
    shape.append(d_out)
    while(len(shape)) < 2*n+1:
        shape.append(1)
    fss1 = fss.view(shape)
    out = fss1
    for k in range(2,n+1):
        cur = torch.transpose(fss1,1,k)
        cur = torch.transpose(cur,1+n,k+n)
        out = out * cur
    return out


# # # # # #
# theory  #
# # # # # #

def K(x,xp,args):
    if args.activation == "Erf":
        return K_erf(x,xp,args)
    if args.activation == "ReLU":
        return K_relu(x, xp, args)
    if args.activation == "GaussNet":
        return K_GaussNet(x, xp, args)

def K_erf(x,xp,args):
    mfxxp, mfxx, mfxpxp = 2*(args.sb**2+ ((torch.dot(x,xp)*(args.sw**2))/args.d_in) ), 2*(args.sb**2+((torch.dot(x,x)*(args.sw**2))/args.d_in) ), 2*(args.sb**2+((torch.dot(xp,xp)*(args.sw**2))/args.d_in) )
    den = torch.sqrt((1+mfxx)*(1+mfxpxp))
    return args.sb**2 + 2*(args.sw**2/np.pi)*np.arcsin(mfxxp/den)

def K_GaussNet(x,xp, args):
    return args.sb**2 + args.sw**2*torch.exp(-args.sw**2*(torch.norm(x-xp)**2)/(2.0*args.d_in))

def K_relu(x,xp,args):
    mxx = torch.sqrt(args.sb**2 + ((args.sw**2)*(torch.dot(x, x))/args.d_in) )
    mxpxp = torch.sqrt(args.sb**2 + ((args.sw**2)*(torch.dot(xp, xp))/args.d_in) )
    mxxp = args.sb**2 + ((args.sw**2)*(torch.dot(x, xp))/args.d_in)
    costheta = mxxp/(mxx*mxpxp)
    theta = 0.0
    kern = 0.0
    if costheta>1. :
        theta = np.arccos(1)
        kern = (mxx*mxpxp/(2.0*np.pi)) * (np.pi - theta)*costheta
    else :
        theta = np.arccos(float(costheta))
        kern = (mxx*mxpxp/(2.0*np.pi)) * (torch.sqrt( 1 - costheta**2 ) + (np.pi - theta)*costheta )
    return args.sb**2 + args.sw**2*kern


def kernel_tensor(xs,args):
    k_th = [[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)]
    for i in range(args.n_inputs):
        for j in range(args.n_inputs):
            k_th[i][j] = G_theory([i,j], xs, args)
    return torch.Tensor(k_th)

def four_pt_tensor(xs, args):
    k_four = [[[[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)]
    for i in range(args.n_inputs):
        for j in range(i, args.n_inputs):
            for k in range(j, args.n_inputs):
                for l in range(k, args.n_inputs):
                    k_four[i][j][k][l] = G_theory([i,j,k,l], xs, args)
    return torch.Tensor(k_four)

def six_pt_tensor(xs, args):
    k_six = [[[[[[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)]
    for i in range(args.n_inputs):
        for j in range(i, args.n_inputs):
            for k in range(j, args.n_inputs):
                for l in range(k, args.n_inputs):
                    for m in range(l, args.n_inputs):
                        for n in range(m, args.n_inputs):
                            k_six[i][j][k][l][m][n] = G_theory([i,j,k,l,m,n], xs, args)
    return torch.Tensor(k_six)

def G_theory(idxs, xs, args):
    # point_pairs = connect all point in idx in pairs
    point_pairs = []

    if len(idxs)%2 == 0 :
        wick_pairs = int(len(idxs)/2)
        anew = list(partition(idxs))
        for i in range(len(anew)):
            index = [0]*wick_pairs
            
            if len(anew[i]) == wick_pairs :
                element = anew[i]
                for j in range(len(element)):
                    if len(element[j]) == 2 :
                        index[j] = 1
            
                if sum(index) == wick_pairs :
                    point_pairs.append(element)

    #print("point pairs", point_pairs)

    tot = 0
    for pairs in point_pairs:
        tot += K_product(xs, pairs, args)
    return tot

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    #     print('f', first)
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            #print(n, subset, 'p')
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
            # put `first` in its own subset
        yield [ [ first ] ] + smaller

def K_product(xs,pairs,args):
    prod = 1
    for pair in pairs:
        prod = prod * K(xs[pair[0]],xs[pair[1]], args)
    return prod



# scipy integration requires numpy, not torch

def K_int(x,xp,args):
    if args.activation == "Erf":
        return K_int_erf(x,xp,args)
    if args.activation == "ReLU":
        return K_int_relu(x, xp, args)
    if args.activation == "GaussNet":
        return K_int_GaussNet(x, xp, args)

def K_int_GaussNet(x,xp, args):
    return args.sb**2 + args.sw**2*np.exp(-args.sw**2*(np.dot(x, x)-2.*np.dot(x,xp)+np.dot(xp,xp))/2.0)

def K_int_erf(x,xp,args):
    mfxxp, mfxx, mfxpxp = 2*(args.sb**2+ ((np.dot(x,xp)*(args.sw**2))/args.d_in) ), 2*(args.sb**2+((np.dot(x,x)*(args.sw**2))/args.d_in) ), 2*(args.sb**2+((np.dot(xp,xp)*(args.sw**2))/args.d_in) )
    den = np.sqrt((1+mfxx)*(1+mfxpxp))
    return args.sb**2 + 2*(args.sw**2/np.pi)*np.arcsin(mfxxp/den)


def K_int_relu(x,xp,args):
    mxx = np.sqrt(args.sb**2 + ((args.sw**2)*(np.dot(x, x))/args.d_in) )
    mxpxp = np.sqrt(args.sb**2 + ((args.sw**2)*(np.dot(xp, xp))/args.d_in) )
    mxxp = args.sb**2 + ((args.sw**2)*(np.dot(x, xp))/args.d_in)
    costheta = mxxp/(mxx*mxpxp)
    theta = 0.0
    kern = 0.0
    if costheta>1. :
        theta = np.arccos(1)
        kern = (mxx*mxpxp/(2.0*np.pi)) * (np.pi - theta)*costheta
    else :
        theta = np.arccos(float(costheta))
        kern = (mxx*mxpxp/(2.0*np.pi)) * (np.sqrt( 1 - costheta**2 ) + (np.pi - theta)*costheta )
    return args.sb**2 + args.sw**2*kern


# for trimming tensors that have duplicate components

def trim_sym_tensor(a, args):
    k_six = [[[[[[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)]
    for i in range(args.n_inputs):
        for j in range(i, args.n_inputs):
            for k in range(j, args.n_inputs):
                for l in range(k, args.n_inputs):
                    for m in range(l, args.n_inputs):
                        for n in range(m, args.n_inputs):
                            k_six[i][j][k][l][m][n] = a[i][j][k][l][m][n]
    return np.array(k_six)

def trim_sym_tensor4(a, args):
    k_four = [[[[np.nan for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)] for _ in range(args.n_inputs)]
    for i in range(args.n_inputs):
        for j in range(i, args.n_inputs):
            for k in range(j, args.n_inputs):
                for l in range(k, args.n_inputs):
                    k_four[i][j][k][l] = a[i][j][k][l]
    return np.array(k_four)


# the next two functions need to be integrated to not include the origin, since the ReLU integration will give an error otherwise

def four_pt_int(x1, x2, x3, x4, cutoff, args):
    if args.d_in == 1:
        neg = integrate.quad(lambda x: (24.)*K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), -cutoff, 0)[0]
        pos = integrate.quad(lambda x: (24.)*K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), 0, cutoff)[0]
        return neg + pos
    if args.d_in == 2:
        q1 = integrate.dblquad(lambda y, x: (24.)*K_EFT((x, y), tuple(x1.tolist()), args)*K_EFT((x, y), tuple(x2.tolist()), args)*K_EFT((x, y), tuple(x3.tolist()), args)*K_EFT((x, y), tuple(x4.tolist()), args), -cutoff, 0, -cutoff, 0)[0]
        q2 = integrate.dblquad(lambda y, x: (24.)*K_EFT((x, y), tuple(x1.tolist()), args)*K_EFT((x, y), tuple(x2.tolist()), args)*K_EFT((x, y), tuple(x3.tolist()), args)*K_EFT((x, y), tuple(x4.tolist()), args), -cutoff, 0, 0, cutoff)[0]
        q3 = integrate.dblquad(lambda y, x: (24.)*K_EFT((x, y), tuple(x1.tolist()), args)*K_EFT((x, y), tuple(x2.tolist()), args)*K_EFT((x, y), tuple(x3.tolist()), args)*K_EFT((x, y), tuple(x4.tolist()), args), 0, cutoff, -cutoff, 0)[0]
        q4 = integrate.dblquad(lambda y, x: (24.)*K_EFT((x, y), tuple(x1.tolist()), args)*K_EFT((x, y), tuple(x2.tolist()), args)*K_EFT((x, y), tuple(x3.tolist()), args)*K_EFT((x, y), tuple(x4.tolist()), args), 0, cutoff, 0, cutoff)[0]
        return q1 + q2 + q3 + q4
    if args.d_in == 3:
        q1 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), -cutoff, 0, -cutoff, 0,-cutoff, 0)[0]
        q2 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), 0, cutoff, -cutoff, 0,-cutoff, 0)[0]
        q3 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), -cutoff, 0, -cutoff, 0,0, cutoff)[0]
        q4 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), 0, cutoff, -cutoff, 0,0, cutoff)[0]
        q5 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), -cutoff, 0, 0, cutoff,-cutoff, 0)[0]
        q6 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), 0, cutoff, 0, cutoff,-cutoff, 0)[0]
        q7 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), -cutoff, 0, 0, cutoff ,0, cutoff)[0]
        q8 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), 0, cutoff, 0, cutoff,0, cutoff)[0]
        return q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8


def intkappa(x1, x2, x3, x4, x5, x6, cutoff, args):
    if args.activation == "GaussNet" or args.activation == "Erf":
        return intkappa_infinite(x1, x2, x3, x4, x5, x6, cutoff, args)
    xx1 = x1.item()
    xx2 = x2.item()
    xx3 = x3.item()
    xx4 = x4.item()
    xx5 = x5.item()
    xx6 = x6.item()
    left = integrate.quad(lambda x: K_EFT(x, xx1, args)*K_EFT(x, xx2, args)*K_EFT(x, xx3, args)*K_EFT(x, xx4, args)*K_EFT(xx5, xx6, args), -cutoff, 0)[0]
    right = integrate.quad(lambda x: K_EFT(x, xx1, args)*K_EFT(x, xx2, args)*K_EFT(x, xx3, args)*K_EFT(x, xx4, args)*K_EFT(xx5, xx6, args), 0, cutoff)[0]
    return left + right


def K_EFT(x,xp,args):
    if args.activation == "Erf":
        a = K_int_erf(x,xp,args) - args.sb**2
        return a
    if args.activation == "ReLU":
        a = K_int_relu(x, xp, args) - args.sb**2
        return a
    if args.activation == "GaussNet":
        a = K_int_GaussNet(x, xp, args) - args.sb**2
        return a

def intkappa_infinite(x1, x2, x3, x4, x5, x6, cutoff, args):
    xx1 = x1.item()
    xx2 = x2.item()
    xx3 = x3.item()
    xx4 = x4.item()
    xx5 = x5.item()
    xx6 = x6.item()
    full = integrate.quad(lambda x: K_EFT(x, xx1, args)*K_EFT(x, xx2, args)*K_EFT(x, xx3, args)*K_EFT(x, xx4, args)*K_EFT(xx5, xx6, args), -cutoff, cutoff)[0]
    return full

def four_pt_int_infinite(x1, x2, x3, x4, cutoff, args):
    if args.d_in == 1:
        q1 = integrate.quad(lambda x: (24.)*K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), -cutoff, cutoff)[0]
        return q1
    if args.d_in == 2:
        q1 = integrate.dblquad(lambda y, x: (24.)*K_EFT((x, y), tuple(x1.tolist()), args)*K_EFT((x, y), tuple(x2.tolist()), args)*K_EFT((x, y), tuple(x3.tolist()), args)*K_EFT((x, y), tuple(x4.tolist()), args), -cutoff, cutoff, -cutoff, cutoff)[0]
        return q1
    if args.d_in == 3:
        q1 = integrate.tplquad(lambda z, y, x: (24.)*K_EFT((x, y, z), tuple(x1.tolist()), args)*K_EFT((x, y, z), tuple(x2.tolist()), args)*K_EFT((x, y, z), tuple(x3.tolist()), args)*K_EFT((x, y, z), tuple(x4.tolist()), args), -cutoff, cutoff, -cutoff, cutoff,-cutoff, cutoff)[0]
        return q1





def local0(x1, x2, x3, x4, cutoff, args):
    int_feyn = 0
    if args.activation == "ReLU" or args.activation == "Erf":
        int_feyn += integrate.quad(lambda x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), -cutoff, 0)[0]
        int_feyn += integrate.quad(lambda x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), 0, cutoff)[0]

    if args.activation == "GaussNet":
        int_feyn += integrate.quad(lambda x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), -cutoff, cutoff)[0]

    return 24*int_feyn                                # combinatorial factor 24

def local2(x1, x2, x3, x4, cutoff, args):
    int_feyn = 0
    if args.activation == "ReLU" or args.activation == "Erf":
        int_feyn += integrate.quad(lambda x: x**2*K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), -cutoff, 0)[0]
        int_feyn += integrate.quad(lambda x: x**2*K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), 0, cutoff)[0]
    
    if args.activation == "GaussNet":
        int_feyn += integrate.quad(lambda x: x**2*K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(x, x4.item(), args), -cutoff, cutoff)[0]
    
    return 24*int_feyn                               # combinatorial factor 24

def nonlocal22(x1, x2, x3, x4, cutoff, args):
    int_feyn = 0
    if args.activation == "ReLU" or args.activation == "Erf":
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(y, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, 0, -cutoff, 0)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(y, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, 0, 0, cutoff)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(y, x3.item(), args)*K_EFT(y, x4.item(), args), 0, cutoff, -cutoff, 0)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(y, x3.item(), args)*K_EFT(y, x4.item(), args), 0, cutoff, 0, cutoff)[0]

        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(y, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, 0, -cutoff, 0)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(y, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, 0, 0, cutoff)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(y, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), 0, cutoff, -cutoff, 0)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(y, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), 0, cutoff, 0, cutoff)[0]

        int_feyn += integrate.dblquad(lambda y, x: K_EFT(y, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, 0, -cutoff, 0)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(y, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, 0, 0, cutoff)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(y, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), 0, cutoff, -cutoff, 0)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(y, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), 0, cutoff, 0, cutoff)[0]
            
    if args.activation == "GaussNet":
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(y, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, cutoff, -cutoff, cutoff)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(x, x1.item(), args)*K_EFT(y, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, cutoff, -cutoff, cutoff)[0]
        int_feyn += integrate.dblquad(lambda y, x: K_EFT(y, x1.item(), args)*K_EFT(x, x2.item(), args)*K_EFT(x, x3.item(), args)*K_EFT(y, x4.item(), args), -cutoff, cutoff, -cutoff, cutoff)[0]
    
#        print("doing integral nonlocal")
    return 8*int_feyn                              # to make up total 24 diagrams