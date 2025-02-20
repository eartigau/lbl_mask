from astropy.table import Table
import numpy as np
import glob
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.constants import c
from astropy.io import fits
import os
import argparse

def roll_frac(input_map, dv):
    """
    Roll an array by a fraction of an element
    :param input_map: array to be rolled
    :param dv: fraction of an element to roll by
    :return: rolled array
    """
    roll_prev = np.floor(dv).astype(int)
    roll_next = np.ceil(dv).astype(int)
    frac = dv - roll_prev
    output_map = np.zeros_like(input_map)

    for i in range(len(input_map)):
        a = np.roll(input_map[i], roll_prev[i])
        b = np.roll(input_map[i], roll_next[i])
        output_map[i] = a * (1 - frac) + b * frac

    return output_map

def gauss(v, mu, fwhm, depth):
    """
    Generate a Gaussian profile
    :param v: velocity array
    :param mu: mean of the Gaussian
    :param fwhm: full width at half maximum
    :param depth: depth of the Gaussian
    :return: Gaussian profile
    """
    ew = fwhm / np.sqrt(2 * np.log(2))
    return depth * np.exp(-0.5 * ((v - mu) / ew) ** 2)

def doppler(wave, velocity):
    """
    Apply relativistic Doppler shift to a wavelength array
    :param wave: wavelength array
    :param velocity: velocity in m/s
    :return: Doppler-shifted wavelength array
    """
    return wave * np.sqrt((1 + velocity / c) / (1 - velocity / c))

def compute_ccf(file, mask_file, bands=None, velorange=100, velostep=1, outdir=None):
    """
    Compute the cross-correlation function (CCF) for a given spectrum and mask
    :param file: path to the input spectrum file
    :param mask: mask table
    :param bands: dictionary of wavelength bands
    :param velorange: velocity range for the CCF
    :param velostep: velocity step for the CCF
    :param outdir: output directory for the CCF file
    :return: CCF table
    """
    if outdir is None:
        outdir = os.path.dirname(file) + '/ccf'
    if not os.path.exists(outdir):
        print(f'Creating directory {outdir}')
        os.makedirs(outdir)

    outname = os.path.basename(file).replace('.fits', '_ccf.fits')
    outname = os.path.join(outdir, outname)

    # test if outname exists and skip if it does
    if os.path.exists(outname):
        print(f'{outname} already exists, skipping')
        tbl = Table.read(outname)

        return tbl


    if bands is None:
        bands = dict()
        bands['Y'] = [1000, 1100]
        bands['J'] = [1100, 1300]
        bands['H'] = [1300, 1500]
        bands['K'] = [2100, 2300]

    dvs = np.arange(-velorange, velorange, velostep, dtype=float)
    ccf = dict()
    ccf['dvs'] = dvs
    for key in bands.keys():
        ccf[key] = np.zeros_like(dvs, dtype=float)

    tbl_spectrum = Table.read(file)
    h = fits.getheader(file)
    berv = h['BERV']

    mask = Table.read(mask_file)

    for band in bands.keys():
        print(f'Processing band {band}')
        wave1 = bands[band][0]
        wave2 = bands[band][1]

        valid = (mask['ll_mask_s'] > wave1) & (mask['ll_mask_s'] < wave2)
        mask_band = mask[valid]

        w = tbl_spectrum['wavelength'].astype(float)
        f = tbl_spectrum['flux'].astype(float)
        valid = np.isfinite(f)
        spl = ius(w[valid], f[valid], k=3)
        for idv in range(len(dvs)):
            ccf[band][idv] = np.nansum(spl(doppler(mask_band['ll_mask_s'],
                                                   (dvs[idv] - berv) * 1e3)) * mask_band['w_mask'])

    tbl = Table(ccf)
    tbl['dvs'].unit = 'km/s'

    print('Writing to ', outname)
    for key in ['BITPIX', 'NAXIS', 'EXTEND']:
        if key in h:
            del h[key]

    with fits.HDUList([fits.PrimaryHDU()]) as hdul:
        bin_table_hdu = fits.BinTableHDU(tbl)
        for key, value in h.items():
            if key not in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND']:
                bin_table_hdu.header[key] = value
        bin_table_hdu.header['TFIELDS'] = len(tbl.columns)
        for i, col in enumerate(tbl.columns, start=1):
            bin_table_hdu.header[f'TFORM{i}'] = tbl[col].format
        hdul.append(bin_table_hdu)
        hdul.writeto(outname, overwrite=True)

    return tbl

def main():
    parser = argparse.ArgumentParser(description='Compute the cross-correlation function (CCF) for a given spectrum and mask.')
    parser.add_argument('files', type=str, nargs='+', help='Path(s) to the input spectrum file(s)')
    parser.add_argument('mask', type=str, help='Path to the mask file')
    parser.add_argument('--velorange', type=float, default=100, help='Velocity range for the CCF (default: 100)')
    parser.add_argument('--velostep', type=float, default=1, help='Velocity step for the CCF (default: 1)')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for the CCF file (default: subfolder in input file directory called ccf/)')

    args = parser.parse_args()

    # Compute the CCF for each input file
    for file in args.files:
        compute_ccf(file, args.mask, velorange=args.velorange, velostep=args.velostep, outdir=args.outdir)

if __name__ == '__main__':
    main()