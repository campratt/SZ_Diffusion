import numpy as np
import healpy as hp
import time
import os

from ..bin import pysm
from pysm import pysm3
import pysm3
import pysm3.units as pyu
from pysm3 import utils as pyutils
from numba import njit
import argparse

import astropy.units as uu
from astropy.constants import c
sol = c.to('km/s').value

from utils import gnu_Kcmb, Tcmb, freqs_planck, noise_freq_planck, nside_freq, get_trans




def get_args_parser():
    parser = argparse.ArgumentParser('Create components', add_help=False)
    parser.add_argument('--ID', default='00000', type=str, help="""Simulation ID""")
    parser.add_argument('--data_root', default='./Han21_reduce', type=str, help="""Path to reduced data""")
    parser.add_argument('--freq', default=217, type=int, help="""Frequency""")
    parser.add_argument('--nside', default=2048, type=int, help="""NSIDE parameter""")
    parser.add_argument('--output_root', default='SkyModels_no_beam', type=str, help="""Output directory""")
    
    return parser.parse_args()  




if __name__ == "__main__":
    args = get_args_parser()


    save_dir = os.path.join(args.output_root,f'{args.ID}')
    os.makedirs(save_dir,exist_ok=True)

    data_dir = os.path.join(args.data_root,f'{args.ID}')

    seed = int(args.ID)

    # CMB
    class CMB(pysm3.Model):
        def __init__(self):

            #fn_cmb = f'/nfs/turbo/lsa-jbregman/campratt/UNET/data/reduce_Han21/{args.ID}/lensed_cmb_T_{args.ID}.fits'
            fn_cmb = os.path.join(data_dir,f'lensed_cmb_T.fits')
            cmb_map = hp.read_map(fn_cmb)
            self.cmb_map = cmb_map.copy()
            self.cmb_map = hp.ud_grade(self.cmb_map, args.nside)
            self.nside = args.nside
            
        @pyu.quantity_input
        def get_emission(self, freqs: pyu.GHz, weights=None) -> pyu.uK_RJ:
            
            #https://pysm3.readthedocs.io/en/3.2.0/_modules/pysm/models/cmb.html
            
            freqs_num = pyutils.check_freq_input(freqs)
            weights_norm = pyutils.normalize_weights(freqs_num, weights)
            convert_to_uK_RJ = (1*pyu.uK_CMB).to_value(pyu.uK_RJ, equivalencies=pyu.cmb_equivalencies(freqs))
            if freqs_num.size == 1:
                scaling_factor = convert_to_uK_RJ
            else:
                scaling_factor = np.trapz(convert_to_uK_RJ * weights_norm, x=freqs_num)
            
            val = self.cmb_map * scaling_factor
            val = np.array([val,val,val])
        
            
            return pyu.Quantity(val, unit=pyu.uK_RJ, copy=False)
        

    # SZ
    class SZ(pysm3.Model):
        def __init__(self):
            
            self.sz_map = hp.read_map(os.path.join(data_dir, f'sz_y.fits'))
            self.nside = args.nside

            
        @pyu.quantity_input
        def get_emission(self, freqs: pyu.GHz, weights=None) -> pyu.uK_RJ:
            
            #https://pysm3.readthedocs.io/en/3.2.0/_modules/pysm/models/cmb.html
            
            
            freqs_num = pyutils.check_freq_input(freqs)
            weights_norm = pyutils.normalize_weights(freqs_num, weights)
            v = gnu_Kcmb(freqs_num) * Tcmb * pyu.K_CMB
            convert_to_uK_RJ = (v).to_value(pyu.uK_RJ, equivalencies=pyu.cmb_equivalencies(freqs))

            if freqs_num.size == 1:
                scaling_factor = convert_to_uK_RJ
            else:
                scaling_factor = np.trapz(convert_to_uK_RJ * weights_norm, x=freqs_num)

            
            val = self.sz_map * scaling_factor
            val = np.array([val,val,val])

            self.scaling_factor = scaling_factor
            
            return pyu.Quantity(val, unit=pyu.uK_RJ, copy=False)
        

    # Radio
    

    @njit
    def get_emission_radio_numba(freqs,weights,amps,slopes):
        val = np.zeros_like(amps)
        for i in range(len(amps)):
            #print(i)
            temp = amps[i]*(freqs/90)**slopes[i]
            val[i] = np.trapz(temp*weights,freqs)
        return val

    def get_emission_radio(freqs,amps,slopes):
        val = np.zeros_like(amps)
        for i in range(len(amps)):
            val[i] = amps[i]*(freqs/90)**slopes[i]
        return val
    
    class Radio(pysm3.Model):
        def __init__(self):
            fn_params = os.path.join(data_dir,f'radio_params.npz')
            d = np.load(fn_params)['arr_0']
            slopes, amps = d.T
            self.slopes = slopes.copy()
            self.amps = amps.copy()
            self.nside = args.nside

            
        @pyu.quantity_input
        def get_emission(self, freqs: pyu.GHz, weights=None) -> pyu.uK_RJ:
            
            #https://pysm3.readthedocs.io/en/3.2.0/_modules/pysm/models/cmb.html

            freqs_num = pyutils.check_freq_input(freqs)
            
            if weights is None:
                val = get_emission_radio(freqs_num,self.amps,self.slopes)
            else:
                weights_norm = pyutils.normalize_weights(freqs_num, weights)
                val = get_emission_radio_numba(freqs_num, weights_norm, self.amps, self.slopes)
            
            val = np.array([val,val,val])
            
            return pyu.Quantity(val, unit=pyu.uK_RJ, copy=False)
        

    # IR

    @njit
    def get_emission_ir_numba(freqs,weights,amps1,slopes1,amps2,slopes2):
        val = np.zeros_like(amps1)
        for i in range(len(amps1)):
            #print(i)
            temp = amps1[i]*(freqs/148)**slopes1[i] + amps2[i]*(freqs/148)**slopes2[i]
            val[i] = np.trapz(temp*weights,freqs)
        return val


    def get_emission_ir(freqs,amps1,slopes1,amps2,slopes2):
        val = np.zeros_like(amps1)
        for i in range(len(amps1)):
            #print(i)
            val[i] = amps1[i]*(freqs/148)**slopes1[i] + amps2[i]*(freqs/148)**slopes2[i]
        return val

    class IR(pysm3.Model):
        def __init__(self):
            #fn_params = f'/nfs/turbo/lsa-jbregman/campratt/UNET/data/reduce_Han21/{args.ID}/ir_params.npz'
            fn_params = os.path.join(data_dir,f'ir_params.npz')
            d = np.load(fn_params)['arr_0']
            slopes1, amps1, slopes2, amps2 = d.T
            self.slopes1 = slopes1.copy()
            self.amps1 = amps1.copy()
            self.slopes2 = slopes2.copy()
            self.amps2 = amps2.copy()
            self.nside = args.nside

            
        @pyu.quantity_input
        def get_emission(self, freqs: pyu.GHz, weights=None) -> pyu.uK_RJ:
            
            #https://pysm3.readthedocs.io/en/3.2.0/_modules/pysm/models/cmb.html
            

            freqs_num = pyutils.check_freq_input(freqs)
            
            if weights is None:
                val_OG = get_emission_ir(freqs_num,self.amps1,self.slopes1,self.amps2,self.slopes2)
                
            else:
                weights_norm = pyutils.normalize_weights(freqs_num, weights)
                val_OG = get_emission_ir_numba(freqs_num, weights_norm, self.amps1, self.slopes1, self.amps2, self.slopes2)
            
            val = np.array([val_OG,val_OG,val_OG])
            
            return pyu.Quantity(val, unit=pyu.uK_RJ, copy=False)
        


    # Instrumental

    class Instrument(pysm3.Model):
        def __init__(self):
            np.random.seed(seed)
            self.nside = args.nside
            
        @pyu.quantity_input
        def get_emission(self, freqs: pyu.GHz, weights=None) -> pyu.uK_RJ:
            
            freqs_num = pyutils.check_freq_input(freqs)
            weights_norm = pyutils.normalize_weights(freqs_num, weights)
            
            if freqs_num.size>1:
                a = np.argmin(abs(np.trapz(freqs_num*weights_norm, x=freqs_num)/np.trapz(weights_norm, x=freqs_num) - freqs_planck))
                f = freqs_planck[a]
                C_ell = noise_freq_planck[f]

            else:
                f = int(freqs_num)
                print(f)
                C_ell = noise_freq_planck[f]

            nside_use = nside_freq[f]
            pixel_area = hp.nside2pixarea(nside=nside_use)

            noise_per_pixel = np.sqrt(C_ell / pixel_area)
            npix = hp.nside2npix(nside_use)
            val = np.random.normal(scale = noise_per_pixel, size=npix)

            val = hp.ud_grade(val, self.nside)

            val = (val * pyu.uK_CMB).to_value(
            (pyu.uK_RJ), equivalencies=pyu.cmb_equivalencies(f * pyu.GHz))
            
            val = np.array([val,val,val])

            return pyu.Quantity(val, unit=pyu.uK_RJ, copy=False)
        

    # Dust
    dust = pysm3.ModifiedBlackBodyRealization(
    nside=args.nside,
    amplitude_modulation_temp_alm = "dust_gnilc/raw/gnilc_dust_temperature_modulation_alms_lmax768.fits.gz",
    amplitude_modulation_pol_alm = "dust_gnilc/raw/gnilc_dust_polarization_modulation_alms_lmax768.fits.gz",
    largescale_alm = "dust_gnilc/raw/gnilc_dust_largescale_template_logpoltens_alm_nside2048_lmax1024_complex64.fits.gz",
    small_scale_cl = "dust_gnilc/raw/gnilc_dust_small_scales_logpoltens_cl_lmax16384.fits.gz",
    largescale_alm_mbb_index = "dust_gnilc/raw/gnilc_dust_largescale_template_beta_alm_nside2048_lmax1024.fits.gz",
    small_scale_cl_mbb_index = "dust_gnilc/raw/gnilc_dust_small_scales_beta_cl_lmax16384_2023.06.06.fits.gz",
    largescale_alm_mbb_temperature = "dust_gnilc/raw/gnilc_dust_largescale_template_Td_alm_nside2048_lmax1024.fits.gz",
    small_scale_cl_mbb_temperature = "dust_gnilc/raw/gnilc_dust_small_scales_Td_cl_lmax16384_2023.06.06.fits.gz",
    freq_ref = "353 GHz",
    has_polarization = True,
    # Remove the galplane_fix option to skip applying the galactic plane fix
    galplane_fix = "dust_gnilc/raw/gnilc_dust_galplane.fits.gz",
    #galplane_fix_beta_Td = "dust_gnilc/raw/gnilc_dust_beta_Td_galplane.fits.gz"
    # Configuration for reproducing d10
    #seeds = [8192,777,888]
    seeds = [seed,seed,seed],
    synalm_lmax = 3*args.nside,
    max_nside = args.nside)

    # Synchrotron
    sync = pysm3.PowerLawRealization(
    nside=args.nside,
    largescale_alm = "synch/raw/synch_largescale_template_logpoltens_alm_lmax128_2023.02.24.fits.gz",
    amplitude_modulation_temp_alm = "synch/raw/synch_temperature_modulation_alms_lmax64_2023.02.24.fits.gz",
    amplitude_modulation_pol_alm = "synch/raw/synch_polarization_modulation_alms_lmax64_2023.02.24.fits.gz",
    #amplitude_modulation_beta_alm = "synch/raw/synch_amplitude_modulation_alms_lmax768.fits.gz",
    small_scale_cl = "synch/raw/synch_small_scales_cl_lmax16384_2023.02.24.fits.gz",
    largescale_alm_pl_index = "synch/raw/synch_largescale_beta_alm_nside512_lmax768.fits.gz",
    small_scale_cl_pl_index = "synch/raw/synch_small_scales_beta_cl_lmax16384.fits.gz",
    freq_ref = "23 GHz",
    max_nside = args.nside,
    has_polarization = True,
    # Remove the galplane_fix option to skip applying the galactic plane fix
    #galplane_fix = "synch/raw/synch_galplane.fits.gz",
    # Configuration for reproducing s5
    seeds = [seed,seed],
    synalm_lmax = 3*args.nside)

    
    sz = SZ()
    cmb = CMB()
    radio = Radio()
    ir = IR()
    inst = Instrument()

    components_list = [sz, cmb, radio, ir, dust, sync, inst]
    components_names = ['sz','cmb','radio', 'ir','dust', 'sync', 'inst']

    for name, component in zip(components_names, components_list):
        print(f"Component: {name}")

        sky = pysm3.Sky(nside=args.nside,component_objects=[component],output_unit=pyu.K_CMB)

        at = time.perf_counter()

        freqs = np.array([args.freq])
        map_freq = sky.get_emission(get_trans(args.freq)[0]*pyu.GHz, get_trans(args.freq)[1])
        #X_full = map_freq[0].value

        hp.write_map(os.path.join(save_dir,f'{name}_{args.freq}GHz.fits'), map_freq[0].value, overwrite=True)
        print(f"Time taken for {name}: {time.perf_counter()-at:.2f} seconds")

        if name == 'sz' and args.freq == 217:
            hp.write_map(os.path.join(save_dir,'sz_y.fits'), sz.sz_map, overwrite=True)

            
            

