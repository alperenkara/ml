# Import the relevant modules to be used later
import urllib
import re
import os
import numpy as np
from PIL import Image
import sys
import cntk as C

try:
    from urllib.request import urlretrieve, urlopen
except ImportError:
    from urllib import urlretrieve, urlopen

# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

isFast = True

# Determine the data path for testing
# Check for an environment variable defined in CNTK's test infrastructure
envvar = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'


def is_test(): return envvar in os.environ


if is_test():
    test_data_path_base = os.path.join(os.environ[envvar], "Tutorials", "data")
    test_data_dir = os.path.join(
        test_data_path_base, "BerkeleySegmentationDataset")
    test_data_dir = os.path.normpath(test_data_dir)

# Default directory in a local folder where the tutorial is run
data_dir = os.path.join("data", "BerkeleySegmentationDataset")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# folder with images to be evaluated
example_folder = os.path.join(data_dir, "example_images")
if not os.path.exists(example_folder):
    os.makedirs(example_folder)

# folders with resulting images
results_folder = os.path.join(data_dir, "example_results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# training configuration
MINIBATCH_SIZE = 8 if isFast else 16
NUM_MINIBATCHES = 200 if isFast else 1000000

# Ensure the training and test data is generated and available for this tutorial.
# We search in two locations for the prepared Berkeley Segmentation Dataset.
data_found = False

for data_dir in [os.path.join("data", "BerkeleySegmentationDataset")]:
    train_hr_path = os.path.join(data_dir, "train224")
    train_lr_path = os.path.join(data_dir, "train112")
    if os.path.exists(train_hr_path) and os.path.exists(train_lr_path):
        data_found = True
        break

if not data_found:
    raise ValueError(
        "Please generate the data by completing the first part of this notebook.")

print("Data directory is {0}".format(data_dir))

# folders with training data (high and low resolution images) and paths to map files
training_folder_HR = os.path.join(data_dir, "train224")
training_folder_LR = os.path.join(data_dir, "train112")
MAP_FILE_Y = os.path.join(data_dir, "train224", "map.txt")
MAP_FILE_X = os.path.join(data_dir, "train112", "map.txt")

# image dimensions
NUM_CHANNELS = 3
LR_H, LR_W, HR_H, HR_W = 112, 112, 224, 224
LR_IMAGE_DIMS = (NUM_CHANNELS, LR_H, LR_W)
HR_IMAGE_DIMS = (NUM_CHANNELS, HR_H, HR_W)

# basic resnet block


def resblock_basic(inp, num_filters):
    c1 = C.layers.Convolution(
        (3, 3), num_filters, init=C.he_normal(), pad=True, bias=False)(inp)
    c1 = C.layers.BatchNormalization(map_rank=1)(c1)
    c1 = C.param_relu(C.Parameter(c1.shape, init=C.he_normal()), c1)

    c2 = C.layers.Convolution(
        (3, 3), num_filters, init=C.he_normal(), pad=True, bias=False)(c1)
    c2 = C.layers.BatchNormalization(map_rank=1)(c2)
    return inp + c2


def resblock_basic_stack(inp, num_stack_layers, num_filters):
    assert (num_stack_layers >= 0)
    l = inp
    for _ in range(num_stack_layers):
        l = resblock_basic(l, num_filters)
    return l

# SRResNet architecture


def SRResNet(h0):
    print('Generator inp shape: ', h0.shape)
    with C.layers.default_options(init=C.he_normal(), bias=False):

        h1 = C.layers.Convolution((9, 9), 64, pad=True)(h0)
        h1 = C.param_relu(C.Parameter(h1.shape, init=C.he_normal()), h1)

        h2 = resblock_basic_stack(h1, 16, 64)

        h3 = C.layers.Convolution((3, 3), 64, activation=None, pad=True)(h2)
        h3 = C.layers.BatchNormalization(map_rank=1)(h3)

        h4 = h1 + h3
        # here

        h5 = C.layers.ConvolutionTranspose2D(
            (3, 3), 64, pad=True, strides=(2, 2), output_shape=(224, 224))(h4)
        h5 = C.param_relu(C.Parameter(h5.shape, init=C.he_normal()), h5)

        h6 = C.layers.Convolution((3, 3), 3, pad=True)(h5)

        return h6


def build_SRResNet_graph(lr_image_shape, hr_image_shape, net):
    inp_dynamic_axes = [C.Axis.default_batch_axis()]
    real_X = C.input(
        lr_image_shape, dynamic_axes=inp_dynamic_axes, name="real_X")
    real_Y = C.input(
        hr_image_shape, dynamic_axes=inp_dynamic_axes, name="real_Y")

    real_X_scaled = real_X/255
    real_Y_scaled = real_Y/255

    genG = net(real_X_scaled)

    G_loss = C.reduce_mean(C.square(real_Y_scaled - genG))

    G_optim = C.adam(G_loss.parameters,
                     lr=C.learning_rate_schedule(
                         [(1, 0.01), (1, 0.001), (98, 0.0001)], C.UnitType.minibatch, 10000),
                     momentum=C.momentum_schedule(0.9), gradient_clipping_threshold_per_sample=1.0)

    G_G_trainer = C.Trainer(genG, (G_loss, None), G_optim)

    return (real_X, real_Y, genG, real_X_scaled, real_Y_scaled, G_optim, G_G_trainer)


# create a map file from a flat folder
import cntk.io.transforms as xforms

def create_map_file_from_flatfolder(folder):
    file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    map_file_name = os.path.join(folder, "map.txt")
    with open(map_file_name , 'w') as map_file:
        for entry in os.listdir(folder):
            filename = os.path.join(folder, entry)
            if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                tempName = '/'.join(filename.split('\\'))
                tempName = '/'.join(tempName.split('//'))
                tempName = '//'.join(tempName.split('/'))
                map_file.write("{0}\t0\n".format(tempName))
    return map_file_name


# creates a minibatch source for training or testing
def create_mb_source(map_file, width, height, num_classes = 10, randomize = True):
    transforms = [xforms.scale(width = width,  height = height, channels = NUM_CHANNELS, interpolations = 'linear')]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features = C.io.StreamDef(field = 'image', transforms = transforms),
        labels = C.io.StreamDef(field = 'label', shape = num_classes))), randomize = randomize)

# training
def train(arch, lr_dims, hr_dims, build_graph):
    create_map_file_from_flatfolder(training_folder_LR)
    create_map_file_from_flatfolder(training_folder_HR)

    print("Starting training")

    reader_train_X = create_mb_source(MAP_FILE_X, lr_dims[1], lr_dims[2])
    reader_train_Y = create_mb_source(MAP_FILE_Y, hr_dims[1], hr_dims[2])
    real_X, real_Y, genG, real_X_scaled, real_Y_scaled, G_optim, G_G_trainer = build_graph(lr_image_shape=lr_dims,
                                                                                           hr_image_shape=hr_dims, net=arch)

    print_frequency_mbsize = 50

    pp_G = C.logging.ProgressPrinter(print_frequency_mbsize)

    input_map_X = {real_X: reader_train_X.streams.features}
    input_map_Y = {real_Y: reader_train_Y.streams.features}

    for train_step in range(NUM_MINIBATCHES):

        X_data = reader_train_X.next_minibatch(MINIBATCH_SIZE, input_map_X)
        batch_inputs_X = {real_X: X_data[real_X].data}

        Y_data = reader_train_Y.next_minibatch(MINIBATCH_SIZE, input_map_Y)
        batch_inputs_X_Y = {
            real_X: X_data[real_X].data, real_Y: Y_data[real_Y].data}

        G_G_trainer.train_minibatch(batch_inputs_X_Y)
        pp_G.update_with_trainer(G_G_trainer)
        G_trainer_loss = G_G_trainer.previous_minibatch_loss_average

    return (G_G_trainer.model, real_X, real_X_scaled, real_Y, real_Y_scaled)


SRResNet_model, real_X, real_X_scaled, real_Y, real_Y_scaled = train(
    SRResNet, LR_IMAGE_DIMS, HR_IMAGE_DIMS, build_SRResNet_graph)
