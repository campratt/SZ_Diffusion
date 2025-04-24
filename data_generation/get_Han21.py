import numpy as np
import healpy as hp
import os
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve Han21 data")
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save outputs', default='./Han21_raw')
    parser.add_argument('--ID', type=str, required=False, help='Full-sky ID', default='00000')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    save_dir = os.path.join(args.output_dir,args.ID)
    os.makedirs(save_dir, exist_ok=True)

    d = 'https://portal.nersc.gov/project/cmb/data/generic/mmDL/healpix/%s/'%(args.ID)

    components = [f'tsz_148ghz_{args.ID}.fits',
                  f'lensed_cmb_T_{args.ID}.fits',
                  f'rad_pts_030ghz_{args.ID}.fits',
                  f'rad_pts_090ghz_{args.ID}.fits',
                  f'rad_pts_148ghz_{args.ID}.fits',
                  f'ir_pts_030ghz_{args.ID}.fits',
                  f'ir_pts_090ghz_{args.ID}.fits',
                  f'ir_pts_148ghz_{args.ID}.fits',
                  f'ir_pts_219ghz_{args.ID}.fits',
                  f'ir_pts_277ghz_{args.ID}.fits',
                  f'ir_pts_350ghz_{args.ID}.fits']
    
    def get_data(comp):
        url = os.path.join(d,comp)
        fn_save = os.path.join(save_dir,comp)
        # Download original Han21 data
        os.system('wget -O %s %s'%(fn_save,url))
        # Resize to NSIDE=2048
        comp_map = hp.read_map(fn_save)
        comp_map = hp.ud_grade(comp_map,2048)
        hp.write_map(fn_save,comp_map,overwrite=True)
        
    for comp in components:
        get_data(comp)



