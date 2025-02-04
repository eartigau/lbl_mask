import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
import os
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import pickle
from tqdm import tqdm
import warnings
from scipy.ndimage import binary_erosion,binary_dilation
import warnings
import argparse
import sys
from astropy.table import Table

def median_sigma_n():

    """
    Compute the median and sigma of the median of n random values.

    This function generates n random values and computes the median and sigma of the median.

    Parameters:
    n (int, optional): The number of random values to generate. Default is 100.

    Returns:
    float: The median of the median of n random values.
    float: The sigma of the median of n random values.
    """

    vals =  np.array(10**(np.arange(7)/2.+1),dtype =int)

    # Generate n random values
    for n in vals:
        factor = np.nanstd(np.nanmedian(np.random.normal(0,1, [n,1000]),axis=0))*np.sqrt(n)
        print(factor, n)


# Suppress RuntimeWarnings globally. We have already handled NaN values in the code.
# and expect some slices to be all-NaN.
warnings.filterwarnings("ignore", category=RuntimeWarning)

def med_berv_bin(cube, bervs, binsize=1):
    """
    Compute the median of a data cube binned by BERV values.

    This function bins the input data cube based on BERV (Barycentric Earth Radial Velocity) values,
    computes the median for each bin, and then returns the median of these medians. This is useful
    for reducing the impact of having lots of files at a single BERV value.

    Parameters:
    cube (array): A 2D array where each row represents a spectrum and each column represents a wavelength.
    bervs (array): A 1D array of BERV values corresponding to each spectrum in the cube.
    binsize (int, optional): The size of the bins for grouping BERV values. Default is 1.

    Returns:
    array: A 1D array representing the median spectrum after binning by BERV values.
    """
    # Bin the BERV values by dividing by the binsize and taking the floor division
    bins = bervs // binsize
    
    # Get the unique bin values
    ubins = np.unique(bins)
    
    # Initialize an array to store the intermediate medians for each unique bin
    med_intermediate = np.zeros((len(ubins), cube.shape[1]))

    # Loop over each unique bin
    for iubin, ubin in enumerate(ubins):
        # Compute the median of the spectra that fall into the current bin
        med_intermediate[iubin] = np.nanmedian(cube[bins == ubin], axis=0)

    # Compute the median of the intermediate medians across all bins
    return np.nanmedian(med_intermediate, axis=0)

def write_t(data, file):
    """
    Write a dictionary of data to a Multi-Extension FITS (MEF) file.

    This function opens a FITS file for reading and writing ('rw' mode), and allows overwriting if the file already exists.
    It iterates over each key in the data dictionary, retrieves the data and its associated header, and writes them
    to the FITS file as new extensions.

    Parameters:
    data (dict): A dictionary containing the data and headers to be written to the FITS file.
    file (str): The path to the FITS file to be written.

    Returns:
    None
    """

    # set primary hdu
    hdul0 = fits.PrimaryHDU(header=data['_header'])

    hdus = [hdul0]
    for key in data.keys():
        if '_header' in key:
            continue

        data_fits = data[key]
        header = data[key + '_header']
        header['EXTNAME'] = (key,'Name of the extension')
        # find if it is a table
        if isinstance(data_fits, Table):
            hdu = fits.BinTableHDU(data_fits, header=header, name=key)
        else:
            hdu = fits.ImageHDU(data_fits, header=header, name=key)
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(file, overwrite=True)        

def read_t(file):
    """
    Read a Multi-Extension FITS (MEF) file and create a dictionary containing all the extensions and their headers.

    This function opens a FITS file for reading and iterates over each Header Data Unit (HDU) in the file.
    It reads the data and headers from each HDU and stores them in a dictionary, with the extension names as keys.

    Parameters:
    file (str): The path to the FITS file to be read.

    Returns:
    dict: A dictionary containing the data and headers from all the extensions in the FITS file.
    """
    
    data = dict()

    with fits.open(file) as hdul:
        for hdu in hdul:
            if 'EXTNAME' in hdu.header:
                key = hdu.header['EXTNAME']
                data[key] = hdu.data
                data[key + '_header'] = hdu.header
            else:
                data['_header'] = hdu.header
    
    return data

def save_pickle(filename, variable):
    """
    Save a variable to a pickle file.

    This function saves a given variable to a specified pickle file.

    Parameters:
    filename (str): The path to the pickle file.
    variable (any): The variable to be saved.

    Returns:
    None
    """
    with open(filename, 'wb') as handle:
        pickle.dump(variable, handle)

def read_pickle(filename):
    """
    Read a variable from a pickle file.

    This function reads and returns a variable from a specified pickle file.

    Parameters:
    filename (str): The path to the pickle file.

    Returns:
    any: The variable read from the pickle file.
    """
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def smart_get(file, extension):
    """
    Retrieve data from a FITS file or a cached pickle file.

    This function checks if a pickle file exists for the given FITS file and extension.
    If the pickle file exists, it reads the data from the pickle file.
    If the pickle file does not exist, it reads the data from the FITS file and saves it to a pickle file.

    Parameters:
    file (str): The path to the FITS file.
    extension (str): The extension name to retrieve from the FITS file.

    Returns:
    tuple: A tuple containing the data and header for the specified extension.
    """
    # Create the name for the pickle file
    pkl_outname = file.replace('.fits', '.pkl')

    # Check if the pickle file exists
    if not os.path.exists(pkl_outname):
        # Read data from the FITS file
        dict_file = read_t(file)
        # Save data to the pickle file
        save_pickle(pkl_outname, dict_file)
    else:
        # Read data from the pickle file
        dict_file = read_pickle(pkl_outname)
    
    # Return the data and header for the specified extension
    return dict_file[extension], dict_file[extension + '_header']

def lowpassfilter(input_vect, width=101):
    """
    Compute a low-pass filter of an input vector while handling NaN values.

    This function computes a running NaN median over a box of a given length (width value).
    The running median is computed at steps of 1/4th of the width value.
    The resulting median values are used in a spline to recover a vector for all pixel positions within the input vector.

    Parameters:
    input_vect (array): The input vector to be filtered.
    width (int): The width of the box for computing the running median.

    Returns:
    array: The low-pass filtered vector.
    """
    # Indices along input vector
    index = np.arange(len(input_vect))

    # Placeholders for x and y position along vector
    xmed = []
    ymed = []

    # Loop through the length of the input vector
    for i in np.arange(-width // 2, len(input_vect) + width // 2, width // 4):
        # Define the bounds of the box
        low_bound = i
        high_bound = i + int(width)

        # Adjust bounds if they go beyond the edges of the vector
        if low_bound < 0:
            low_bound = 0
        if high_bound > (len(input_vect) - 1):
            high_bound = (len(input_vect) - 1)

        # Get the pixel values within the bounds
        pixval = index[low_bound:high_bound]

        # Skip if there are less than 3 pixel values
        if len(pixval) < 3:
            continue

        # Skip if there are no finite values
        if np.max(np.isfinite(input_vect[pixval])) == 0:
            continue

        # Compute the mean position and NaN median value of the points
        xmed.append(np.nanmean(pixval))
        ymed.append(np.nanmedian(input_vect[pixval]))

    # Convert xmed and ymed to arrays
    xmed = np.array(xmed, dtype=float)
    ymed = np.array(ymed, dtype=float)

    # Return a vector of NaNs if there are less than 3 valid points
    if len(xmed) < 3:
        return np.zeros_like(input_vect) + np.nan

    # Handle duplicate xmed values
    if len(xmed) != len(np.unique(xmed)):
        xmed2 = np.unique(xmed)
        ymed2 = np.zeros_like(xmed2)
        for i in range(len(xmed2)):
            ymed2[i] = np.mean(ymed[xmed == xmed2[i]])
        xmed = xmed2
        ymed = ymed2

    # Create a spline of the median values
    spline = ius(xmed, ymed, k=2, ext=3)
    lowpass = spline(np.arange(len(input_vect)))

    return lowpass

def doppler_shift(wave, v):
    """
    Apply Doppler shift to a wavelength array.

    This function shifts the wavelength array by a given velocity using the Doppler formula.

    Parameters:
    wave (array): The input wavelength array.
    v (float): The velocity in km/s.

    Returns:
    array: The Doppler-shifted wavelength array.
    """
    c = 299792.458  # Speed of light in km/s
    wave_shifted = wave * np.sqrt((1 + v/c)/(1 - v/c))
    return wave_shifted

def doppler_spline(wave_sp, sp, dv):
    """
    Apply a Doppler shift to a spectrum and interpolate the result.

    This function shifts the input wavelength array by a given velocity using the Doppler formula,
    then interpolates the shifted spectrum back onto the original wavelength grid. It also creates
    a mask to handle invalid values.

    Parameters:
    wave_sp (array): The input wavelength array.
    sp (array): The input spectrum array.
    dv (float): The velocity in km/s for the Doppler shift.

    Returns:
    array: The Doppler-shifted and interpolated spectrum.
    """
    # Apply Doppler shift to the wavelength array
    wave3 = doppler_shift(wave_sp, -dv)
    
    # Create a mask for finite values in the spectrum
    g = np.isfinite(sp)
    
    # Convert the mask to float for interpolation
    g_float = np.isfinite(sp).astype(float)
    
    # Interpolate the shifted spectrum back onto the original wavelength grid
    sp2 = ius(wave3[g], sp[g], k=3, ext=1)(wave_sp)
    
    # Interpolate the mask to handle invalid values
    mask = ius(wave3, g_float, k=1, ext=1)(wave_sp)
    
    # Set values to NaN where the mask is less than 0.5
    # pixels there would be interpolated in big gaps of NaNs
    sp2[mask < 0.5] = np.nan

    return sp2

def sig(vals):
    """
    Calculate the sigma value (standard deviation) from percentiles.

    This function calculates the sigma value as half the difference between the 84th and 16th percentiles.

    Parameters:
    vals (array): The input array of values.

    Returns:
    float: The sigma value.
    """

    with warnings.catch_warnings() as _: # Suppress warnings
        n1, p1 = np.nanpercentile(vals, [16, 84])
    return (p1-n1)/2

def construct_residuals(obj, nsig_cuts = [3], doplot = False):
    """
    Construct residuals for a given object and nsig_cut value.

    This function processes a set of FITS files for a specified object, applies Doppler shifts,
    computes residuals, and generates masks for outliers based on sigma cuts. The residuals are
    calculated by comparing the observed data to a model and applying a low-pass filter to remove
    high-frequency noise. The function can also generate plots for visual inspection.

    Parameters:
    obj (str): The name of the object, which is used to locate the directory containing the FITS files.
    nsig_cuts (list): A list of sigma cut values to use for masking outliers. Default is [3].
    doplot (bool): A flag to control whether plots are generated for visual inspection. Default is False.

    Returns:
    None
    """

    # test if nsig_cuts is a list and if not, convert it to a list
    if not isinstance(nsig_cuts, list):
        nsig_cuts = [nsig_cuts]


    # Get list of FITS files in the directory
    files = np.array(glob.glob(f'{obj}/*t.fits'))
    if len(files) == 0:
        print(f'No FITS files found in directory: {obj}')
        return
    
    files = files[np.argsort(files)]

    hdr0 = smart_get(files[0], 'FluxA')[1]
    if 'NIRPS' in hdr0['INSTRUME'].upper():
        fiber_setup = 'A'
        instrument = 'NIRPS_HE'
    elif 'SPIROU' in hdr0['INSTRUME'].upper():
        fiber_setup = 'AB'
        instrument = 'SPIROU'
    else:
        raise ValueError('Unknown instrument in header')

    # Number of orders in the data
    n_orders = smart_get(files[0], f'Flux{fiber_setup}')[0].shape[0]

    outpaths = []
    for nsig_cut in nsig_cuts:
        # Output path for masked files
        outpath = f'{obj}mask{nsig_cut:.1f}sig'
        # Create the output directory if it does not exist
        if not os.path.exists(outpath):
            print(f'Creating directory: {outpath}')
            os.makedirs(outpath)
        outpaths.append(outpath)

    # Width parameter for lowpass filter
    wpwidth = 51

    # Initialize array to store BERV values
    bervs = np.zeros(len(files))
    for i in tqdm(range(len(files)), desc='Reading BERV values', leave=False):
        # Read BERV value from each file


        hdr = smart_get(files[i], f'Flux{fiber_setup}')[1]
        if 'BERV' in hdr:
            bervs[i] = hdr['BERV']

            if 'MKT_ARV' in hdr:
                # if we have the latest version of APERO, we can use the MKT_ARV keyword
                # that gives the stars's velocity in m/s. We convert it to km/s and add it to the BERV
                # to get the total velocity. This provides a slightly better correction of the full
                # motion of the star and a slightly better template
                bervs[i] -= hdr['MKT_ARV']/1000
        else:
            print(f'No BERV value found in file: {files[i]}')
            # raise an error if no BERV value is found
            # and we cannot correct for the star's velocity. We will need to update the codes
            # to handle this case.
            raise ValueError('No BERV value found in file')


    if not doplot:
        orders = np.arange(n_orders)
    else:
        # in steps of 10, starting at 5
        orders = np.arange(5, n_orders, 10)

    nsigma_map_outname = 'nsig_residuals_'+obj+'.fits'

    if not os.path.exists(nsigma_map_outname):
        nsig_map = np.zeros((n_orders, 4088))
        for iord in tqdm(orders, desc='Processing orders', leave=False):
            # Initialize arrays to store data
            cube_detector = np.zeros((len(files), 4088))+np.nan
            cube_stellar = np.zeros((len(files), 4088))+np.nan
            cube_model_star = np.zeros((len(files), 4088))+np.nan
            wave = np.zeros((len(files), 4088))

            # Initialize array to track skipped files
            skipped = np.zeros(len(files))
            # Loop through each file to read data and apply Doppler shift
            for i in tqdm(range(len(files)), desc='Reading data', leave=False):
                # Read wavelength data
                wave[i] = smart_get(files[i], f'Wave{fiber_setup}')[0][iord]
                # Read flux data
                tmp = smart_get(files[i], f'Flux{fiber_setup}')[0][iord]

                mean_bad = np.nanmean(~np.isfinite(tmp))
                if mean_bad > 0.9:
                    skipped[i] = 1
                    continue

                # Normalize flux data
                with np.errstate(all='ignore'):
                    tmp/=np.nanmedian(tmp)
                
                cube_detector[i] = tmp
                cube_stellar[i] = doppler_spline(wave[i], tmp, -bervs[i])

            # Skip the order if more than 50% of the files were skipped
            if np.mean(skipped) > 0.5:
                continue

            # Suppress the specific RuntimeWarning about All-NaN slices
            with np.errstate(all='ignore'):
                # Calculate median of processed data
                med = med_berv_bin(cube_stellar, bervs)

            g = np.isfinite(med)

            if np.mean(g)<0.1:
                print('Too few valid pixels')
                continue


            for i in tqdm(range(len(files)), desc='Processing model', leave=False):
                cube_model_star[i] = doppler_spline(wave[i], med, bervs[i])

            # Calculate residuals
            residual = cube_detector - cube_model_star

            for i in tqdm(range(len(files)), desc='Processing residuals', leave=False):
                # Apply lowpass filter to residuals
                res = np.array(residual[i])
                res -= lowpassfilter(res, wpwidth)
                nsig = res/sig(res)
                mask = np.zeros_like(res, dtype = float)
                mask[np.abs(nsig)>3] = np.nan
                res -= lowpassfilter(res+mask, wpwidth)
                residual[i] = res

            with warnings.catch_warnings() as _: # Suppress warnings
                p16, med_res, p84 = np.nanpercentile(residual, [16,50, 84], axis=0)

            # Apply lowpass filter to effective RMS
            effective_rms = lowpassfilter((p84 - p16) / 2, wpwidth)
            nvalid = np.sum(np.isfinite(residual), axis=0)
            p16 /= effective_rms
            med_res /= effective_rms
            p84 /= effective_rms

            nsig = med_res*np.sqrt(nvalid)/1.25
            # this is a factor to correct for the fact that the residuals are not normally distributed

            # Plot data cubes and residuals if required
            if doplot:
                fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharex=True, sharey=False)

                # Share y-axis between rows 0 and 1
                for col in range(2):
                    ax[1, col].get_shared_y_axes().joined(ax[0, col], ax[1, col])
                p1,p99 = np.nanpercentile(cube_detector, [1, 99])
                ax[0,0].imshow(cube_detector, aspect='auto', origin='lower', vmin=p1, vmax=p99)
                ax[0,0].set(xlim=[2000, 2500])
                ax[0,0].set(title='Detector-space data')

                ax[1,0].imshow(cube_model_star, aspect='auto', origin='lower', vmin=p1, vmax=p99)
                ax[1,0].set(xlim=[2000, 2500])
                ax[1,0].set(title='Stellar-restframe model')

                amp = (p99-p1)
                #moy = (p99+p1)/2
                p1,p99 = -amp/2,amp/2
                ax[0,1].imshow(residual, aspect='auto', origin='lower', vmin=p1, vmax=p99)
                ax[0,1].set(xlim=[2000, 2500])
                ax[0,1].set(title='Residuals in detector space')

                residual2 = residual.copy()
                for i in tqdm(range(len(files)), desc='Processing model', leave=False):
                    residual2[i] = doppler_spline(wave[i], residual2[i], -bervs[i])
                ax[1,1].imshow(residual2, aspect='auto', origin='lower', vmin=p1, vmax=p99)
                ax[1,1].set(xlim=[2000, 2500])
                ax[1,1].set(title='Residuals in stellar restframe')

                ax[2,0].plot(med)
                ax[2,0].set(xlim=[2000, 2500])
                ax[2,0].set(title='Median stellar spectrum')

                ax[2,1].set(ylim = [-nsig_cuts[0]*1.5,nsig_cuts[0]*1.5])
                ax[2,1].plot(p16, color = 'grey',alpha = 0.5)
                ax[2,1].plot(med_res, color = 'purple',alpha = 0.5)
                ax[2,1].plot(p84, color = 'grey',alpha = 0.5)
                ax[2,1].set(title = '1-sigma stats')

                #plt.show()

            for isig,nsig_cut in enumerate(nsig_cuts):
                mask_ini = (np.abs(nsig)>nsig_cut) | (~np.isfinite(nsig))
                mask = np.array(mask_ini)
                #singles = (np.convolve(mask,np.ones(3),mode = 'same') == 1) & mask
                #mask[singles] = False
                #mask = binary_dilation(mask, structure=np.ones(5), output=mask)

                if isig ==0:
                    if doplot:
                        #have a 'fill' mask to show the pixels that are masked. The masking in done in blocks
                        #so we can see the blocks that are masked
                        mask2 = np.zeros_like(mask, dtype = float)+np.nan
                        mask2[np.where(mask == 1)] = 0
                        ax[2,1].plot(p16+mask2, color = 'red',alpha = 1.0)
                        ax[2,1].plot(med_res+mask2, color = 'red',alpha = 1.0)
                        ax[2,1].plot(p84+mask2, color = 'red',alpha = 1.0)

            if doplot:
                plt.show()

            nsig_map[iord, :] = nsig
        
        fits.writeto(nsigma_map_outname, nsig_map, overwrite=True)

    if doplot:
        return


    nsig = fits.getdata(nsigma_map_outname)
    for isig,nsig_cut in enumerate(nsig_cuts):
        # Convert the mask to boolean
        mask = (np.abs(nsig) > nsig_cut).astype(bool)

        # Loop through each file to apply the mask
        for ifile, file in enumerate(files):
            # Create the output file name
            outname = f'{outpaths[isig]}/{os.path.basename(file)}'

            # Skip the file if it already exists
            if os.path.exists(outname):
                print(f'Skipping {outname}')
                continue

            # Read the data from the FITS file
            dict_file = read_t(file)
            # Apply the mask to the flux data
            dict_file[f'Flux{fiber_setup}'][mask] = np.nan

            hdr = dict_file[f'Flux{fiber_setup}_header']
            if 'WAVEFILE' in hdr:
                wave_sol_path = f'/space/spirou/SLINKY/data_{instrument}_updatedwavesol/'+hdr['WAVEFILE']
                if os.path.exists(wave_sol_path):
                    print('We have an updated wave solution and will apply it')
                    wave_sol = fits.getdata(wave_sol_path)
                    dict_file[f'Wave{fiber_setup}'] = wave_sol

            # Write the masked data to the output file
            print(f'sigma [{isig+1} / {len(nsig_cuts)}] file [{ifile+1} / {len(files)}]\tWriting file {outname}')
            write_t(dict_file, outname)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Construct residuals for specified objects.")
    
    # Add arguments
    parser.add_argument('objs', type=str, help="Comma-separated list of object names.")
    parser.add_argument('nsig_cuts', type=str, help="Comma-separated list of sigma cut values.")
    parser.add_argument('--doplot', action='store_true', help="Flag to control whether plots are generated for visual inspection.")
    
    # If no arguments are provided, print the help message and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Split the comma-separated inputs into lists
    objs = args.objs.split(',')
    nsig_cuts = [float(x) for x in args.nsig_cuts.split(',')]
    doplot = args.doplot
    
    print(f'Constructing residuals for objects: {objs}')
    print(f'Sigma cut values: {nsig_cuts}')

    for obj in objs:
        # Construct the residuals for the specified object and frac_noise_increase
        construct_residuals(obj, nsig_cuts, doplot=doplot)