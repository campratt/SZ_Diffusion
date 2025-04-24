# SZ_Diffusion

Use a diffusion model to separate the thermal Sunyaev-Zel'dovich effect from structured noise i.e., astrophysical foregrounds and instrumental uncertainties.

This work closely follows the method described [here]([https://your-package-docs.com](https://arxiv.org/pdf/2302.05290).


## Data generation instructions

### Obtain necessary data
- Download simulated data products from Han et al. (2021)
- Retrieve SZ, CMB, radio sources, and infrared sources
- Data location: https://portal.nersc.gov/project/cmb/data/generic/mmDL/healpix
- Reference: https://ui.adsabs.harvard.edu/abs/2021PhRvD.104l3521H/abstract
  ```
  python3 get_Han21.py --ID 00000 --output_dir ./Han21_raw
  ```
### Reduce Han+21 data
- Fit toy emission models to the radio and infrared sources to obtain continuous functions
- Radio sources use a single power law while infrared sources use a double power law
- Change all full-sky maps to have NSIDE=2048
```
python3 reduce_Han21.py --ID 00000 --data_root ./Han21_raw --save_root ./Han21_reduce --cores 20
```
### Generate mock Planck observations
- Uses the reduced Han+21 products, generates random instrumental noise, and uses the PySM to generate thermal dust emission and synchrotron radiation models.
- Reference for PySM: https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.2821T/abstract
- Integrates the emission models over the bandpass for each Planck instrument.
- Everything is calculated in Rayleigh-Jeans temperature units and converted into CMB temperature units Kcmb
- Beam effects are ignored
```
python3 make_components.py --ID 00000 --freq 100 --data_root ./Han21_reduce --output_root ./SkyModels_no_beam
```

### Create train/test data
- Make cutouts of the full-sky simulations
- First run "generate_coordinate_list.ipynb" with the desired number of full-sky maps and number of cutouts per map to create "sample_coordinates.txt"
```
python3 generate_train_test_samples.py --data_root ./SkyModels_no_beam --output_root ./train_test_data --coords_path ./sample_coordinates.txt --cores 8

```




