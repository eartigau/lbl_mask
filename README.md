# Code to perform a masking of spectra for input to the LBL code

## Installation

There's no proper installation script yet (we're working on that). You simply download the Python script and put it in the science folder of your LBL installation. 

## Running the script

To generate the masked files corresponding to a 5 sigma thresholding of residuals, for a given star (lets say PROXIMA), you run the following command:

``` python lbl_resmask.py PROXIMA 6.0 ```

To generate the masked files for two stars and 3 different sigma cuts:

``` python lbl_resmask.py 'PROXIMA,GL699' '6.0,9.0,15.0' ```

You can also show debug plots if you want :

``` python lbl_resmask.py PROXIMA 6.0 --doplot```

The code will save a fits file of 0s and 1s with the name of the object and thresholding. For the above command, you would get  ```mask_PROXIMA_6.0sig.fits```. All masked files will be in the folder ```PROXIMAmask6.0sig/```

## Compatibility with the main LBL code

Inside the LBL code, you will simply need to pass the new object name in the wrapper file :

```rparams['OBJECT_SCIENCE'] = ['PROXIMA','PROXIMAmask6.0sig']```

You can then pass the same object as the template

```rparams['OBJECT_TEMPLATE'] = ['PROXIMA','PROXIMAmask6.0sig']```
