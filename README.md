# Implementaion-1

Implementation for CVPR 2015 Paper: "An Improved Deep Learning Architecture for Person Re-Identification".

Paper link: http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf.

This architechture is implemented based on Keras with Tensorflow backen using Python Programming Language.

# How to use this code

1. Download the CUHK03 dataset and unpack it to local disk. Find the .mat file named 'cuhk-03.mat'.

2. Move the 'cuhk-03.mat' file to the same path with 'make_cuhk03_hdf5_dataset.py' and run the .py file.
   The whole process will take about half an hour and finally get a hdf5 file named 'cuhk_03_for_CNN.h5'

3. Once the 'cuhk03_for_CNN.h5' and 'model_def_compile.py' are in the same path, you could run 'the model_def_compile.py' and it will begin    to train the model. 
