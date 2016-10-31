# Implementaion-1

Implementation for CVPR 2015 Paper: "An Improved Deep Learning Architecture for Person Re-Identification".

[`Paper link`](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf)

This architechture is implemented based on `Keras` with `Tensorflow` backen using `Python` Programming Language.

## How to use this code

1. Download the CUHK03 dataset and unpack it to local disk. 

2. Find the .mat file named `cuhk-03.mat`.

2. Move the `cuhk-03.mat` file to the same path with `make_cuhk03_hdf5_dataset.py`.

4. Run the `make_cuhk03_hdf5_dataset.py` file. (about half an hour)

5. Get a hdf5 file named `cuhk_03_for_CNN.h5`.

6. Move the `model_def_compile.py` file to the same path with `cuhk_03_for_CNN.h5`.

7. run the `model_def_compile.py` and it will begin to train the model.(You'd better run this file through Ipython Environment)

## The dataset HDF5 structure

>cuhk03_for_CNN.h5
>>'train'
>>>'x1'
>>>'x2'
>>>'y'

>>'validation'
>>>'x1'
>>>'x2'
>>>'y'

>>'test'
>>>'a'
>>>'b'
