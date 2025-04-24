import numpy as np
import healpy as hp
import os
from multiprocessing import Pool
import argparse


# Make small set of test data

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Hyperparameter Tuning")
    parser.add_argument('--data_root', type=str, required=False, help='Data directory', default='./fullsky_data')
    parser.add_argument('--output_root', type=str, required=False, help='Directory to save outputs', default='./train_test_data/')
    parser.add_argument('--coords_path', type=str, required=False, help='Path to coordinate file', default='./sample_coordinates.txt')
    parser.add_argument('--signal_components', type=str, nargs='+', required=False, help='Path to coordinate file', default=['sz'])
    parser.add_argument('--noise_components', type=str, nargs='+', required=False, help='Path to coordinate file', default=['cmb','dust','sync','radio','ir','inst'])
    parser.add_argument('--resolution', type=float, required=False, help='Pixel resolution in arcminutes per pixel', default=2)
    parser.add_argument('--size', type=int, required=False, help='Size of cutout', default=128)
    parser.add_argument('--cores', type=int, required=False, help='Number of cores for parallel processing', default=1)
    args = parser.parse_args()
    return args




args = parse_args()

observed_dir = os.path.join(args.output_root, 'observed')
signal_dir = os.path.join(args.output_root, 'signal')
sz_dir = os.path.join(args.output_root, 'sz')
noise_dir = os.path.join(args.output_root, 'noise')

os.makedirs(observed_dir, exist_ok=True)
os.makedirs(signal_dir, exist_ok=True)
os.makedirs(noise_dir, exist_ok=True)
os.makedirs(sz_dir, exist_ok=True)

freqs = [30,44,70,100,143,217,353,545]

sample_ids, map_ids, valid_pixs, lon, lat = np.genfromtxt(args.coords_path,dtype='<U50').T
sample_ids = sample_ids.astype(int)
map_ids = map_ids.astype(str)
valid_pixs = valid_pixs.astype(int)
lon = lon.astype(float)
lat = lat.astype(float)

map_ids_unique = np.unique(map_ids)

nside = 2048

signal = np.empty((len(args.signal_components), len(freqs), 12*nside**2))
signal_y = np.empty((12*nside**2))
noise = np.empty((len(args.noise_components), len(freqs), 12*nside**2))


for map_id in map_ids_unique:

    sz = hp.read_map(os.path.join(args.data_root, map_id, f'sz_y.fits'))

    for i, sig in enumerate(args.signal_components):
        for j, freq in enumerate(freqs):
            signal[i, j] = hp.read_map(os.path.join(args.data_root, map_id, f'{sig}_{freq}GHz.fits'))

    for i, n in enumerate(args.noise_components):
        for j, freq in enumerate(freqs):
            noise[i, j] = hp.read_map(os.path.join(args.data_root, map_id, f'{n}_{freq}GHz.fits'))


    signal_total = np.sum(signal, axis=0)
    noise_total = np.sum(noise, axis=0)
    observed_map = signal_total + noise_total

    where_map_id = np.where(map_ids == map_id)[0]
    def func(w):

        # SZ data
        data = hp.gnomview(sz, rot=(lon[w], lat[w]), xsize=args.size, reso=args.resolution, return_projected_map=True,no_plot=True)      
        np.save(os.path.join(sz_dir, f'sz_{w}.npy'), np.array(data))
        
        # Signal data
        data = []
        for i in range(len(signal_total)):
            data.append(hp.gnomview(signal_total[i], rot=(lon[w], lat[w]), xsize=args.size, reso=args.resolution, return_projected_map=True,no_plot=True))       
        np.save(os.path.join(signal_dir, f's_{w}.npy'), np.array(data))

        # Noise data
        data = []
        for i in range(len(noise_total)):
            data.append(hp.gnomview(noise_total[i], rot=(lon[w], lat[w]), xsize=args.size, reso=args.resolution, return_projected_map=True,no_plot=True))       
        np.save(os.path.join(noise_dir, f'n_{w}.npy'), np.array(data))

        # Observed data
        data = []
        for i in range(len(observed_map)):
            data.append(hp.gnomview(observed_map[i], rot=(lon[w], lat[w]), xsize=args.size, reso=args.resolution, return_projected_map=True,no_plot=True))       
        np.save(os.path.join(observed_dir, f'y_{w}.npy'), np.array(data))


    # Parallel processing
    p = Pool(int(args.cores))
    p.map(func, where_map_id)



