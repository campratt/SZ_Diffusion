import numpy as np
import healpy as hp
import scipy

import pysm3
import pysm3.units as pyu
from pysm3 import utils as pyutils
import healpy as hp
import numpy as np

import astropy
import astropy.io.fits as fits

import astropy.units as uu
from astropy.constants import c
sol = c.to('km/s').value

hplanck_cgs = 6.6261e-27 #cm2 g s-1
sol_cgs = c.to('cm/s').value
kb_cgs = 1.3807e-16 #cm2 g s-2 K-1
Tcmb = 2.7255 # K



freqs_planck = np.array([30,44,70,100,143,217,353,545,857])


nside_freq = {30 :1024,
            44 :1024,
            70 :1024,
            100:2048,
            143:2048,
            217: 2048,
            353: 2048,
            545:2048,
            857:2048}

gnu_freq = {23 :-5.371,
            30 :-5.336,
            33 :-5.291,
            41 :-5.212,
            44 :-5.178,
            61 :-4.933,
            70 :-4.766,
            94 :-4.261,
            100:-4.031,
            143:-2.785,
            217: 0.187,
            353: 6.205,
            545:14.455,
            857:26.335}

res_freq = {23 :52.8,
            30 :33.16,
            33 :39.6,
            41 :30.6,
            44 :28.09,
            61 :21,
            70 :13.08,
            94 :13.2,
            100:9.59,
            143:7.18,
            217: 4.87,
            353: 4.7,
            545:4.73,
            857:4.51}

# Helpful notebook: https://www.zonca.dev/posts/2020-06-19-white-noise-hitmap-fullsky.html
# C_ell in uKcmb^2
# Values from Table 1 in https://arxiv.org/pdf/2307.01043.pdf   McCarthy+23
noise_freq_planck = {
            30 :0.00190,
            
            
            44 :0.00222,
            
            70 :0.00373,
            
            100:0.000507,
            143:0.0000921,
            217: 0.000185,
            353: 0.00200,
            545:0.0551,
            857:30.9}





def bb_deriv(nu_GHz,Tcmb=2.7255):
    h_over_k = hplanck_cgs/kb_cgs
    nu = nu_GHz*1e9
    x = h_over_k*nu/Tcmb
    return 2*hplanck_cgs*nu**3/(sol_cgs**2*np.expm1(x)) * np.exp(x)/(np.exp(x)-1) * x/Tcmb


def wprime(nu_GHz,Tcmb=2.7255):
    h_over_k = hplanck_cgs/kb_cgs
    nu = nu_GHz*1e9
    x = h_over_k*nu/Tcmb
    return x**2*np.exp(x)/(np.exp(x)-1)**2


def gnu_RJ(nu_GHz,Tcmb=2.7255):
    h_over_k = hplanck_cgs/kb_cgs
    nu = nu_GHz*1e9
    x = h_over_k*nu/Tcmb
    return x**2*np.exp(x)/(np.exp(x)-1)**2 * ((x*(np.exp(x)+1)/(np.exp(x)-1)) - 4)

def gnu_Jy(nu_GHz,Tcmb=2.7255):
    h_over_k = hplanck_cgs/kb_cgs
    nu = nu_GHz*1e9
    x = h_over_k*nu/Tcmb
    return x**4*np.exp(x)/(np.exp(x)-1)**2 * ((x*(np.exp(x)+1)/(np.exp(x)-1)) - 4)


def gnu_Kcmb(nu_GHz,Tcmb=2.7255):
    h_over_k = hplanck_cgs/kb_cgs
    nu=nu_GHz*1e9
    x = h_over_k*nu/Tcmb
    return (x*(np.exp(x)+1)/(np.exp(x)-1)) - 4


hdu_HFI = fits.open('../bin/RIMO_HFI_NPIPE.fits')
hdu_LFI = fits.open('../bin/RIMO_LFI_NPIPE.fits')


# Use bandpass transmission to calculate theoretical gnu


def get_trans_planck(nu_GHz,Tcmb=2.7255):
    freq_HFI = [100,143,217,353,545,857]
    freq_LFI = [30,44,70]
    nu = nu_GHz*1e9
    
    if nu_GHz in freq_HFI:
        hdu = hdu_HFI
        b = hdu["BANDPASS_F%i"%nu_GHz].data
        freq = b['WAVENUMBER']*1e-7*sol*1e3*1e9
        
        freq_GHz = freq/1e9
        trans = b['TRANSMISSION']
        
        where_freq_min = np.where((trans<1e-3)&(freq_GHz<nu_GHz))[0][-1]
        where_freq_max = np.where((trans<1e-3)&(freq_GHz>nu_GHz))[0][0]
        trans = trans[where_freq_min:where_freq_max]
        freq_GHz = freq_GHz[where_freq_min:where_freq_max]
        
    elif nu_GHz in freq_LFI:
        hdu = hdu_LFI
        b = hdu["BANDPASS_0%i"%nu_GHz].data
        freq = b['WAVENUMBER']*1e9
        
        freq_GHz = freq/1e9
        trans = b['TRANSMISSION']

    return freq_GHz[1:], trans[1:]



def get_gnu_planck(nu_GHz,Tcmb=2.7255):
    freq_HFI = [100,143,217,353,545,857]
    freq_LFI = [30,44,70]
    nu = nu_GHz*1e9
    
    if nu_GHz in freq_HFI:
        hdu = hdu_HFI
        b = hdu["BANDPASS_F%i"%nu_GHz].data
        freq = b['WAVENUMBER']*1e-7*sol*1e3*1e9
        
        freq_GHz = freq/1e9
        trans = b['TRANSMISSION']
        
        where_freq_min = np.where((trans<1e-3)&(freq_GHz<nu_GHz))[0][-1]
        where_freq_max = np.where((trans<1e-3)&(freq_GHz>nu_GHz))[0][0]
        trans = trans[where_freq_min:where_freq_max]
        freq_GHz = freq_GHz[where_freq_min:where_freq_max]
        
    elif nu_GHz in freq_LFI:
        hdu = hdu_LFI
        b = hdu["BANDPASS_0%i"%nu_GHz].data
        freq = b['WAVENUMBER']*1e9
        
        freq_GHz = freq/1e9
        trans = b['TRANSMISSION']
    
    
    f = scipy.interpolate.interp1d(freq_GHz,trans,fill_value=0)
    
    freqs = np.logspace(np.log10(freq_GHz[1]),np.log10(freq_GHz[-2]),int(1e6))
    #freqs = np.linspace(freq_GHz[1],freq_GHz[-1],int(1e6))
    return np.trapz(f(freqs)*gnu_Kcmb(freqs)*bb_deriv(freqs)*Tcmb,freqs)/np.trapz(f(freqs)*bb_deriv(freqs),freqs)



def get_trans(nu_GHz):
    if nu_GHz in freqs_planck:
        return get_trans_planck(nu_GHz)
    else:
        return None
    
def get_gnu(nu_GHz):
    if nu_GHz in freqs_planck:
        return get_gnu_planck(nu_GHz)
    else:
        return None


