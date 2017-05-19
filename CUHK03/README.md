## How to use this code

1. Download the CUHK03 dataset and unpack it to local disk. 

2. Find the .mat file named **`cuhk-03.mat`**.

2. Find the script **`create_dataset.py`**.

4. Run the **`create_dataset.py`** with the path of `cuhk-03.mat`

5. Get a hdf5 file named **`cuhk_03.h5`**.

6. Run the **`main.py`** with the path of **`cuhk_03.h5`** file.

## The dataset HDF5 structure

>**`cuhk-03.h5`**
>>**`'a'`**
>>>`'train'`
>>>`'validation'`
>>>`'test'`

>>**`'b'`**
>>>**`'train'`**
>>>**`'validation'`**
>>>**`'test'`**
