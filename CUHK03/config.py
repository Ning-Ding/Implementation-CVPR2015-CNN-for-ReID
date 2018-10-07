from easydict import EasyDict

__C = EasyDict()
cfg = __C


__C.DATA = EasyDict()
__C.DATA.ORIGINAL_FILE = "cuhk-03.mat"
__C.DATA.CREATED_FILE = "cuhk-03.hdf5"
__C.DATA.INDEX_FILE = "cuhk-03-index.hdf5"
__C.DATA.IMAGE_SIZE = (60,160)
__C.DATA.ARRAY_SIZE = (160,60)
__C.DATA.PATTERN = EasyDict()
__C.DATA.PATTERN.TRAIN = [1,0,0]
__C.DATA.PATTERN.VALID = [1,0]

__C.TRAIN = EasyDict()
__C.TRAIN.BATCHSIZE = 150
__C.TRAIN.STEPS = 2100
__C.TRAIN.WEIGHT_DECAY = 0.00025
__C.TRAIN.GPU_INDEX = 0


__C.VALID = EasyDict()
__C.VALID.STEPS = 1
