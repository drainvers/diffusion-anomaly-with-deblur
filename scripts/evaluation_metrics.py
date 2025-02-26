import sys
sys.path.append("..")
sys.path.append(".")

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch as th
from skimage.filters import threshold_otsu
import cv2, os
from PIL import Image

import argparse
# from guided_diffusion.script_util import add_dict_to_argparser

# Evaluation metrics
from torcheval.metrics import PeakSignalNoiseRatio
from torcheval.metrics import StructuralSimilarity
from torcheval.metrics import FrechetInceptionDistance

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def load_samples(npz_path):
    samples_npz = np.load(npz_path)

    data = {
        'samples': samples_npz.f.samples,
        'labels': samples_npz.f.labels,
        'names': samples_npz.f.names,
        'originals': samples_npz.f.orgs
    }

    return data

'''
#We define the anomaly map as the absolute difference between the original image and the generated healthy reconstruction. We sum up over all channels (4 different MR sequences):

difftot=abs(original-healthyreconstruction).sum(dim=0)

#We compute the Otsu threshold for the anomaly map:

diff = np.array(difftot)
thresh = threshold_otsu(diff)
mask = th.where(th.tensor(diff) > thresh, 1, 0)  #this is our predicted binary segmentation
viz.image(visualize(mask[ 0,...]), opts=dict(caption="mask"))

#We load the ground truth segmetation mask and put all the different tumor labels to 1:

Labelmask_GT = th.where(groundtruth_segmentation > 0, 1, 0)

pixel_wise_cls = visualize(np.array(th.tensor(diff).view(1, -1))[0, :])
pixel_wise_gt = visualize(np.array(th.tensor(Labelmask_GT).view(1, -1))[0, :])

#Then we compute the Dice and AUROC scores

DSC=dice_score(mask.cpu(), Labelmask_GT.cpu()) #predicted Dice score
auc = roc_auc_score(pixel_wise_gt, pixel_wise_cls)
'''

def calculate_metric_scores(input_images, target_images):
    '''
    Inputs:
    - input_images: A Tensor of size (N, C, H, W), C assumed to be 1
    - target_images: A Tensor of size (N, C, H, W), C assumed to be 1
    '''
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarity()
    fid = FrechetInceptionDistance()

    psnr.update(th.from_numpy(input_images), th.from_numpy(target_images))
    ssim.update(th.from_numpy(input_images), th.from_numpy(target_images))
    fid.update(th.from_numpy(np.repeat(input_images, 3, axis=1).clip(0, 1)), True) # FID requires pixel values to be [0,1] and have 3 channels
    fid.update(th.from_numpy(np.repeat(target_images, 3, axis=1).clip(0, 1)), False) # FID requires pixel values to be [0,1] and have 3 channels

    return dict(
        psnr=psnr.compute(),
        ssim=ssim.compute(),
        fid=fid.compute()
        )

def generate_grid_for_mpl(npz_paths, dest_path='', x_labels=None):
    master_image = None
    master_samples = []
    master_diffs = []

    img_grid = []
    y_labels = None

    for i, path in enumerate(npz_paths):
        sample_data = load_samples(path)

        if i == 0:
            batch_size = len(sample_data['names'])
            y_labels = sample_data['names']
            if x_labels is None:
                x_labels = range(len(npz_paths) * 2 + 1)
            assert len(x_labels) == (len(npz_paths) * 2 + 1)

            master_image = np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1)
            if master_image.shape[2] == 512:
                height, width = master_image.shape[1:]
                master_image = cv2.resize(master_image.transpose(1, 2, 0), (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                master_image = master_image.transpose(2, 0, 1)
            img_list = np.array_split(master_image, batch_size, axis=1)
            for img in img_list:
                img_grid.append([img])
        
        master_sample = np.concatenate(np.repeat(sample_data['samples'], 3, axis=1), axis=1)
        if master_sample.shape[2] == 512:
            height, width = master_sample.shape[1:]
            master_sample = cv2.resize(master_sample.transpose(1, 2, 0), (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            master_sample = master_sample.transpose(2, 0, 1)
        master_samples.append(master_sample)
        
        master_diff = generate_diff_batch(sample_data['originals'], sample_data['samples'])
        if master_diff.shape[2] == 512:
            height, width = master_diff.shape[1:]
            master_diff = cv2.resize(master_diff.transpose(1, 2, 0), (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            master_diff = master_diff.transpose(2, 0, 1)
        master_diffs.append(master_diff)

    for sample_col in master_samples:
        img_list = np.array_split(sample_col, batch_size, axis=1)
        for i, img in enumerate(img_list):
            img_grid[i].append(img)

    for sample_col in master_diffs:
        img_list = np.array_split(sample_col, batch_size, axis=1)
        for i, img in enumerate(img_list):
            img_grid[i].append(img)

    nrow = len(img_grid)
    ncol = len(npz_paths) * 2 + 1

    fig = plt.subplots(figsize=(ncol * 2.56, nrow * 2.56), dpi=100, layout='tight')
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0)

    for row in range(nrow):
        for col in range(ncol):
            im = img_grid[row][col]
            ax = plt.subplot(gs[row, col])
            ax.imshow(np.transpose(im, [1, 2, 0]), cmap='gray', aspect='auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(top=False, right=False, bottom=False, left=False, pad=0)
            ax.set(frame_on=False)
            if row == 0:
                ax.set_xlabel(x_labels[col])
                ax.xaxis.set_label_position('top')
            if col == 0:
                ax.set_ylabel(y_labels.tolist()[row], wrap=True)
                ax.yaxis.set_label_position('left')
    
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(dest_path, 'chexpert_eval_mpl_test.png'), dpi=100)

def generate_master_image(npz_paths):
    master_image = None
    master_samples = []
    master_diffs = []
    results = []

    for i, path in enumerate(npz_paths):
        sample_data = load_samples(path)
        results.append(calculate_metric_scores(sample_data['originals'], sample_data['samples']))

        if i == 0:
            master_image = np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1)
            if master_image.shape[2] == 512:
                height, width = master_image.shape[1:]
                master_image = cv2.resize(master_image.transpose(1, 2, 0), (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                master_image = master_image.transpose(2, 0, 1)
        
        master_sample = np.concatenate(np.repeat(sample_data['samples'], 3, axis=1), axis=1)
        if master_sample.shape[2] == 512:
            height, width = master_sample.shape[1:]
            master_sample = cv2.resize(master_sample.transpose(1, 2, 0), (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            master_sample = master_sample.transpose(2, 0, 1)
        master_samples.append(master_sample)
        
        master_diff = generate_diff_batch(sample_data['originals'], sample_data['samples'])
        if master_diff.shape[2] == 512:
            height, width = master_diff.shape[1:]
            master_diff = cv2.resize(master_diff.transpose(1, 2, 0), (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            master_diff = master_diff.transpose(2, 0, 1)
        master_diffs.append(master_diff)
    
    master_samples = np.concatenate(np.array(master_samples), axis=2)
    master_diffs = np.concatenate(np.array(master_diffs), axis=2)
    return np.concatenate([master_image, master_samples, master_diffs], axis=2), results
                                   
        
def generate_diff_batch(original, reconstructed):
    '''
    Inputs:
    - original - N, C, H, W tensor
    - reconstructed - N, C, H, W tensor

    Returns
    - N, C, H, W tensor with the applied color map
    '''
    diff_list = []
    for i in range(original.shape[0]):
        diff_list.append(generate_diff(original[i][0], reconstructed[i][0]))
    return np.concatenate(diff_list, axis=1)

def generate_diff(original, reconstructed):
    '''
    Inputs:
    - original - H, W tensor
    - reconstructed - H, W tensor

    Returns
    - C, H, W tensor with the applied color map
    '''
    cm = plt.get_cmap('jet') # Applying this colormap will return an image in RGBA format
    diff = np.array(abs(original - reconstructed))
    colored_diff = cm(visualize(diff))[:, :, :3] # Remove alpha channel
    return colored_diff.transpose(2, 0, 1) # HWC -> CHW

def main():
    args = create_argparser().parse_args()
    x_labels = ['Original']
    scenarios = args.x_labels
    x_labels.extend(scenarios)
    x_labels.extend([f'Heatmap ({x})' for x in scenarios])
    print(x_labels)
    generate_grid_for_mpl(args.npz_paths, dest_path=args.result_dir, x_labels=x_labels)
    final_image, final_metrics = generate_master_image(args.npz_paths)
    final_image = (final_image.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(final_image).save(os.path.join(args.result_dir, 'compiled_master_image.png'))

    print(f'{"":14s} PSNR  , SSIM , FID   ')
    for scenario, metric in zip(scenarios, final_metrics):
        print(f"{scenario:14s} {metric['psnr'].item():02.3f}, {metric['ssim'].item():02.3f}, {metric['fid'].item():02.3f}")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--result_dir", required=False, default=os.path.join('results', 'latest'), type=str)
    parser.add_argument(f"--npz_paths", required=True, type=str, nargs='+')
    parser.add_argument(f"--x_labels", required=False, default=None, type=str, nargs='+')
    return parser

if __name__ == '__main__':
    main()

