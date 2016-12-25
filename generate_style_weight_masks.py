# Mainly took from Dmitry's github https://github.com/DmitryUlyanov/online-neural-doodle

import argparse
import os

import numpy as np
import scipy.misc
from joblib import Parallel, delayed

import diamond_square as DS

parser = argparse.ArgumentParser()

parser.add_argument('--height', help='Height of the generated mask.')
parser.add_argument('--width', help='Width of the generated mask.')
parser.add_argument(
    '--out_dir', default='style_weight_train_masks/', help='Where to store generated mask images.')
parser.add_argument(
    '--n_jobs', type=int, default=4, help='Number of worker threads.')
parser.add_argument(
    '--n_masks', type=int, default=1000, help='Number of masks to generate.')



args = parser.parse_args()


# get shape
dims = (int(args.height), int(args.width))


def generate():
    np.random.seed(None)

    hmap = np.array(DS.diamond_square((200, 200), 0, 1, 0.35)) + np.array(DS.diamond_square((200, 200), 0, 1, 0.35)) + np.array(DS.diamond_square((200, 200), 0, 1, 0.35))
    hmap_rounded = np.round(hmap, decimals=1)
    return hmap_rounded


# Generate doodles

gen_masks = Parallel(n_jobs=args.n_jobs)(delayed(generate)()
                                         for i in range(args.n_masks))

# Save

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
for i, mask in enumerate(gen_masks):
    mask_rgb = np.transpose(np.repeat(np.array([mask[:,:]]), 3, axis=0),(1,2,0))
    scipy.misc.imsave('%strain_mask_%d.png' % (args.out_dir, i), mask_rgb)
    # result = general_util.imread('%strain_mask_%d_%d.png' % (args.out_dir, i, j), bw=True)
    # print(result)
