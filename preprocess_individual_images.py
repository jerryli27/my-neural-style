
import argparse

import scipy.misc
from joblib import Parallel, delayed
import general_util
import os
import time
import numpy as np
import cv2
from sketches_util import image_to_sketch

parser = argparse.ArgumentParser()

parser.add_argument('--hw', type=int, default=128, help='Height/Width of the generated mask.')
parser.add_argument(
    '--colored_image_dir', default='/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/', help='Where to store generated mask images.')
parser.add_argument(
    '--save_dir', default='/home/ubuntu/pixiv_downloaded_sketches/', help='Where to store generated mask images.')
parser.add_argument(
    '--n_jobs', type=int, default=4, help='Number of worker threads.')

args = parser.parse_args()


colored_image_dir = args.colored_image_dir # '/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/'  # '/home/xor/pixiv_images/test_images/'
save_dir = args.save_dir# '/home/ubuntu/pixiv_downloaded_sketches/'  # '/home/xor/pixiv_images/test_images_sketches/'

colored_save_dir = save_dir+'color/'
sketchs_save_dir = save_dir+'line/'

hw = args.hw
print('height and width of converted images are: %d' %hw)

colored_image_dir_len = len(colored_image_dir)
img_shape = (hw,hw)

assert save_dir[-1] == '/'
all_img_dirs = general_util.get_all_image_paths_in_dir(colored_image_dir)
num_images = len(all_img_dirs)
print('Number of images to convert is : %d.' %(num_images))


def generate(i):
    img = general_util.imread(all_img_dirs[i])
    sketch = image_to_sketch(img)  # Generate the sketch based on full sized image and resize later.

    img_reshaped = cv2.resize(img,img_shape,interpolation = cv2.INTER_AREA)
    sketch_reshaped = cv2.resize(sketch,img_shape,interpolation = cv2.INTER_AREA)

    img_file_name = general_util.get_file_name(all_img_dirs[i])
    img_subdir_name = os.path.dirname(all_img_dirs[i])[colored_image_dir_len:]
    img_subdir_path = img_subdir_name + '/' + img_file_name + '.png'

    img_save_dir = colored_save_dir + img_subdir_name
    try:
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
    except OSError:
        print('Moving on... Probably multithread conflicts.')

    current_sketch_save_dir = sketchs_save_dir + img_subdir_name
    if not os.path.exists(current_sketch_save_dir):
        os.makedirs(current_sketch_save_dir)

    general_util.imsave(colored_save_dir + img_subdir_path, img_reshaped)
    general_util.imsave(sketchs_save_dir + img_subdir_path, sketch_reshaped)
    if i % 100 == 0:
        end_time = time.time()
        remaining_time = 0.0 if i == 0 else (num_images - i) * (float(end_time - start_time) / i)
        print('%.3f%% done. Remaining time: %.1fs' % (float(i) / num_images * 100, remaining_time))
    return img_subdir_path + '\n'



# Generate doodles

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(colored_save_dir):
    os.makedirs(colored_save_dir)
if not os.path.exists(sketchs_save_dir):
    os.makedirs(sketchs_save_dir)
start_time = time.time()

img_subdir_paths = Parallel(n_jobs=args.n_jobs)(delayed(generate)(i)
                                         for i in range(num_images))

with open(save_dir + 'image_files_relative_paths.txt', 'w') as image_files:
    image_files.writelines(img_subdir_paths)
print('Done')