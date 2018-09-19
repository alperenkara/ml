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
    test_data_dir = os.path.join(test_data_path_base, "BerkeleySegmentationDataset")
    test_data_dir = os.path.normpath(test_data_dir)

# Default directory in a local folder where the tutorial is run
data_dir = os.path.join("data", "BerkeleySegmentationDataset")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#folder with images to be evaluated
example_folder = os.path.join(data_dir, "example_images")
if not os.path.exists(example_folder):
    os.makedirs(example_folder)

#folders with resulting images
results_folder = os.path.join(data_dir, "example_results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

def download_data(images_dir, link):
    #Open the url
    images_html = urlopen(link).read().decode('utf-8')

    #looking for .jpg images whose names are numbers
    image_regex = "[0-9]+.jpg"

    #remove duplicates
    image_list = set(re.findall(image_regex, images_html))
    print("Starting download...")

    num = 0

    for image in image_list:
        num = num + 1
        filename = os.path.join(images_dir, image)

        if num % 25 == 0:
            print("Downloading image %d of %d..." % (num, len(image_list)))
        if not os.path.isfile(filename):
            urlretrieve(link + image, filename)
        else:
            print("File already exists", filename)

    print("Images available at: ", images_dir)


#folder for raw images, before preprocess
images_dir = os.path.join(data_dir, "Images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

#Get the path for pre-trained models and example images
if is_test():
    print("Using cached test data")
    models_dir = os.path.join(test_data_dir, "PretrainedModels")
    images_dir = os.path.join(test_data_dir, "Images")
else:
    models_dir = os.path.join(data_dir, "PretrainedModels")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    images_dir = os.path.join(data_dir, "Images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    #link to BSDS dataset
    link = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/images/plain/normal/color/"

    download_data(images_dir, link)

print("Model directory", models_dir)
print("Image directory", images_dir)


#extract 64 x 64 patches from BSDS dataset
def prep_64(images_dir, patch_h, patch_w, train64_lr, train64_hr, tests):
    if not os.path.exists(train64_lr):
        os.makedirs(train64_lr)

    if not os.path.exists(train64_hr):
        os.makedirs(train64_hr)

    if not os.path.exists(tests):
        os.makedirs(tests)

    k = 0
    num = 0

    print("Creating 64 x 64 training patches and tests from:", images_dir)

    for entry in os.listdir(images_dir):
        filename = os.path.join(images_dir, entry)
        img = Image.open(filename)
        rect = np.array(img)

        num = num + 1

        if num % 25 == 0:
            print("Processing image %d of %d..." % (num, len(os.listdir(images_dir))))

        if num % 50 == 0:
            img.save(os.path.join(tests, str(num) + ".png"))
            continue

        x = 0
        y = 0

        while(y + patch_h <= img.width):
            x = 0
            while(x + patch_w <= img.height):
                patch = rect[x : x + patch_h, y : y + patch_w]
                img_hr = Image.fromarray(patch, 'RGB')

                img_lr = img_hr.resize((patch_w // 2, patch_h // 2), Image.ANTIALIAS)
                img_lr = img_lr.resize((patch_w, patch_h), Image.BICUBIC)

                out_hr = os.path.join(train64_hr, str(k) + ".png")
                out_lr = os.path.join(train64_lr, str(k) + ".png")

                k = k + 1

                img_hr.save(out_hr)
                img_lr.save(out_lr)

                x = x + 42
            y = y + 42
    print("Done!")


#extract 224 x 224 and 112 x 112 patches from BSDS dataset
def prep_224(images_dir, patch_h, patch_w, train112, train224):
    if not os.path.exists(train112):
        os.makedirs(train112)

    if not os.path.exists(train224):
        os.makedirs(train224)

    k = 0
    num = 0

    print("Creating 224 x 224 and 112 x 112 training patches from:", images_dir)

    for entry in os.listdir(images_dir):
        filename = os.path.join(images_dir, entry)
        img = Image.open(filename)
        rect = np.array(img)

        num = num + 1
        if num % 25 == 0:
            print("Processing image %d of %d..." % (num, len(os.listdir(images_dir))))

        x = 0
        y = 0

        while(y + patch_h <= img.width):
            x = 0
            while(x + patch_w <= img.height):
                patch = rect[x : x + patch_h, y : y + patch_w]
                img_hr = Image.fromarray(patch, 'RGB')

                img_lr = img_hr.resize((patch_w // 2, patch_h // 2), Image.ANTIALIAS)

                for i in range(4):
                    out_hr = os.path.join(train224, str(k) + ".png")
                    out_lr = os.path.join(train112, str(k) + ".png")

                    k = k + 1

                    img_hr.save(out_hr)
                    img_lr.save(out_lr)

                    img_hr = img_hr.transpose(Image.ROTATE_90)
                    img_lr = img_lr.transpose(Image.ROTATE_90)

                x = x + 64
            y = y + 64
    print("Done!")

#blurry 64x64 destination
train64_lr = os.path.join(data_dir, "train64_LR")

#original 64x64 destination
train64_hr = os.path.join(data_dir, "train64_HR")

#112x112 patches destination
train112 = os.path.join(data_dir, "train112")

#224x224 pathes destination
train224 = os.path.join(data_dir, "train224")

#tests destination
tests = os.path.join(data_dir, "tests")

#prep
prep_64(images_dir, 64, 64, train64_lr, train64_hr, tests)
prep_224(images_dir, 224, 224, train112, train224)

