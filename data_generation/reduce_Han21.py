import sys
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from multiprocessing import Pool
import os
from scipy.optimize import curve_fit
import argparse

from ..bin import pysm
from pysm import pysm3
import pysm3.units as pyu
from pysm3 import utils as pyutils
from utils import gnu_Kcmb, Tcmb

def parse_args():
    parser = argparse.ArgumentParser(description="Reduce Han21 data")
    parser.add_argument('--data_root', type=str, required=False, help='Directory of inputs', default='./Han21_raw')
    parser.add_argument('--save_root', type=str, required=False, help='Directory to save outputs', default='./Han21_reduce')
    parser.add_argument('--ID', type=str, required=False, help='Full-sky ID', default='00000')
    parser.add_argument('--cores', type=int, required=False, help='Cores for processing', default=8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
        
    save_dir = os.path.join(args.save_root,args.ID)
    os.makedirs(save_dir,exist_ok=True)

    data_dir = os.path.join(args.data_root,args.ID)
    os.makedirs(data_dir,exist_ok=True)

    # SZ
    sz_148 = hp.read_map(os.path.join(data_dir, 'tsz_148ghz_%s.fits'%(args.ID)))
    sz = sz_148*1e-6/(gnu_Kcmb(148)*Tcmb) #dimensionless
    sz[sz<0] = 0
    hp.write_map(os.path.join(save_dir,'sz_y.fits'),sz,overwrite=True)
    
    # CMB
    os.system("cp -r %s %s"%(os.path.join(data_dir,'lensed_cmb_T_%s.fits'%(args.ID)), os.path.join(save_dir,'lensed_cmb_T.fits')))




    ### Radio

    freqs_radio = np.array(['030','090','148'])
    freqs_num_radio = np.array([30,90,148])
    freq_maps = []
    for i, freq in enumerate(freqs_radio):
        print(freq)
        fn = os.path.join(data_dir , 'rad_pts_%sghz_%s.fits'%(freq,args.ID))
        f = hp.read_map(fn)
        f_uK_CMB = f*pyu.uK_CMB
        f_uK_RJ = (f_uK_CMB).to_value(pyu.uK_RJ, equivalencies=pyu.cmb_equivalencies(freqs_num_radio[i]*pyu.GHz))*pyu.uK_RJ

        freq_maps.append(f_uK_RJ)
        
        
    freq_maps = np.array(freq_maps)

    inds = np.arange(2048**2*12)

    x = freqs_num_radio
    x = x/90

    
    

    def func_radio(i):
        y = freq_maps[:,i]
        
        f = lambda X,m,b: b*X**m
        
        try:
            popt,pcov = curve_fit(f,x,y,maxfev=5000)
            return popt
        except:
            return np.array([0,0])
        
    p = Pool(int(args.cores))
    data = p.map(func_radio,inds)

    np.savez(os.path.join(save_dir,'radio_params.npz'),np.array(data))


    del x
    del data



    ### IR

    freqs_ir = np.array(['030','090','148','219','277','350'])
    freqs_num_ir = np.array([30,90,148,219,277,350])
    nside = 2048


    freq_maps = []
    for i, freq in enumerate(freqs_ir):
        print(freq)
        fn = os.path.join(data_dir , 'ir_pts_%sghz_%s.fits'%(freq,args.ID))
        f = hp.read_map(fn)
        f_uK_CMB = f*pyu.uK_CMB
        f_uK_RJ = (f_uK_CMB).to_value(pyu.uK_RJ, equivalencies=pyu.cmb_equivalencies(freqs_num_ir[i]*pyu.GHz))*pyu.uK_RJ

        freq_maps.append(f_uK_RJ)

        
    freq_maps = np.array(freq_maps)

    inds = np.arange(nside**2*12)

    x = freqs_num_ir
    x = x/148

    def func_ir(i):
        
        y = freq_maps[:,i]
        
        
        
        f = lambda X,m1,b1,m2,b2: b1*X**m1 + b2*X**m2
        
        try:
            popt,pcov = curve_fit(f,x,y)

            return popt
        except:
            return np.zeros(4)
        
    p = Pool(int(args.cores))
    data = p.map(func_ir,inds)



    np.savez(os.path.join(save_dir,'ir_params.npz'),data)

