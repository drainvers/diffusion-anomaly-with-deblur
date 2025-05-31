import sys
sys.path.append("..")
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import cv2, os, csv, argparse

from sklearn.metrics import roc_auc_score
from matplotlib import gridspec
from skimage.filters import threshold_otsu
from PIL import Image, ImageDraw
# from visdom import Visdom
# viz = Visdom(port=8850)

from skimage import segmentation
# from guided_diffusion.script_util import add_dict_to_argparser

# Evaluation metrics
from torcheval.metrics import PeakSignalNoiseRatio
from torcheval.metrics import StructuralSimilarity
from torcheval.metrics import FrechetInceptionDistance

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()

def load_samples(npz_path):
    samples_npz = np.load(npz_path, allow_pickle=True)
    sorting_indices = np.argsort(samples_npz.f.names)

    data = {
        'samples': samples_npz.f.samples[sorting_indices],
        'org_labels': samples_npz.f.org_labels[sorting_indices],
        'tgt_labels': samples_npz.f.tgt_labels[sorting_indices],
        'names': samples_npz.f.names[sorting_indices],
        'originals': samples_npz.f.orgs[sorting_indices]
    }

    return data

def rescale_image(image, scale_factor):
    height, width = image.shape[1:]
    image = cv2.resize(image.transpose(1, 2, 0), (np.int(width * scale_factor), np.int(height * scale_factor)), interpolation=cv2.INTER_AREA)
    image = image.transpose(2, 0, 1)
    return image

def calculate_metric_scores(input_images, target_images, names, include_iou=False):
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

    if include_iou:
        pred_masks, gt_masks = create_mask_batch(input_images, target_images, names, args.gt_mask_path, args.pred_mask_threshold)
        ious = []
        conf_matrix = dict(TP=0, TN=0, FP=0, FN=0)

        for pred_mask, gt_mask in zip(list(pred_masks), list(gt_masks)):
            ious.append(calculate_iou(pred_mask, gt_mask))
            curr_matrix = get_conf_matrix(pred_mask, gt_mask)
            for key in conf_matrix.keys():
                conf_matrix[key] += curr_matrix[key]

        ious = np.array(ious)
        _, _, specificity = calculate_precision_recall_specificity(conf_matrix)

        return dict(
            psnr=psnr.compute(),
            ssim=ssim.compute(),
            fid=fid.compute(),
            miou=np.nanmean(ious),
            specificity=specificity
            )

    return dict(
            psnr=psnr.compute(),
            ssim=ssim.compute(),
            fid=fid.compute(),
            )

def cam_to_segmentation(cam_mask, threshold=np.nan, smoothing=False, k=0):
    """
    Threshold a saliency heatmap to binary segmentation mask.
    Args:
        cam_mask (torch.Tensor): heat map in the original image size (H x W).
            Will squeeze the tensor if there are more than two dimensions.
        threshold (np.float64): threshold to use
        smoothing (bool): if true, smooth the pixelated heatmaps using box filtering
        k (int): size of kernel used for box filter smoothing (int); k must be
                 >= 0; if k is > 0, make sure to set if_smoothing to True,
                 otherwise no smoothing would be performed.

    Returns:
        segmentation (np.ndarray): binary segmentation output
    """
    if (len(cam_mask.shape) > 2):
        cam_mask = cam_mask.squeeze()

    assert len(cam_mask.shape) == 2

    # normalize heatmap
    mask = cam_mask - cam_mask.min()
    mask = mask / mask.max()

    # use Otsu's method to find threshold if no threshold is passed in
    if np.isnan(threshold):
        mask = np.uint8(255 * mask)

        if smoothing:
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            gray_img = cv2.boxFilter(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY),
                                     -1, (k, k))
            # mask = 255 - gray_img
            mask = gray_img

        maxval = np.max(mask)
        thresh = cv2.threshold(mask, 0, maxval, cv2.THRESH_OTSU)[1]

        # draw out contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        polygons = []
        for cnt in cnts:
            if len(cnt) > 1:
                polygons.append([list(pt[0]) for pt in cnt])

        # create segmentation based on contour
        img_dims = (mask.shape[1], mask.shape[0])
        segmentation_output = Image.new('1', img_dims)
        for polygon in polygons:
            coords = [(point[0], point[1]) for point in polygon]
            ImageDraw.Draw(segmentation_output).polygon(coords,
                                                        outline=1,
                                                        fill=1)
        segmentation = np.array(segmentation_output, dtype="int")
    else:
        segmentation = np.array(mask > threshold, dtype="int")
        if smoothing:
            smoothed_mask = segmentation.copy().astype(np.float32)
            smoothed_mask = cv2.boxFilter(smoothed_mask, -1, (k, k))
            
            # Threshold again to get binary mask
            segmentation = (smoothed_mask > 0.5).astype(np.uint8)

    return segmentation

def overlay_masks(image, masks, colors=None, alpha=0.5, border_thickness=2, border_color=None, seed=None):
    """
    Overlays multiple segmentation masks on an image with borders around contours.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image as a numpy array of shape (H, W, C) where C is the number of channels.
    masks : list of numpy.ndarray
        List of masks, each as a numpy array of shape (H, W, C) matching the image dimensions.
        Non-zero values in the mask indicate the areas to be colored.
    colors : list of tuple/list or None, optional
        List of RGB colors for the masks, each as a tuple/list of 3 values.
        If None, random colors will be generated for each mask.
    alpha : float, optional
        Transparency of the overlay, between 0 and 1. Default is 0.5.
    border_thickness : int, optional
        Thickness of the contour border in pixels. Default is 2.
    border_color : tuple/list or None, optional
        RGB color for the contour borders. If None, the same color as the mask will be used
        but with increased intensity.
    seed : int or None, optional
        Random seed for reproducible color generation. Default is None.
        
    Returns:
    --------
    numpy.ndarray
        Image with all masks overlaid on it and contour borders drawn.
    """
    # Check if masks is empty
    if not masks:
        return image.copy()
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Make a copy of the input image to avoid modifying the original
    output = image.copy().astype(np.float32)
    
    # Generate random colors if not provided
    if colors is None:
        colors = []
        for _ in range(len(masks)):
            # Generate random RGB values
            color = (np.random.randint(0, 255), 
                     np.random.randint(0, 255), 
                     np.random.randint(0, 255))
            colors.append(color)
    
    # Check that we have enough colors
    if len(colors) < len(masks):
        raise ValueError("Not enough colors provided for all masks")
    
    # Process each mask
    for i, mask in enumerate(masks):
        # Check that image and mask have the same shape
        if image.shape != mask.shape:
            raise ValueError(f"Image and mask {i} must have the same shape. "
                            f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        
        # Create a binary mask for contour detection (sum across channels)
        binary_mask = (np.sum(mask, axis=2) > 0).astype(np.uint8)
        
        # Create a colored mask
        colored_mask = np.zeros_like(image, dtype=np.float32)
        for j in range(min(3, image.shape[2])):
            colored_mask[..., j] = np.where(mask[..., j] > 0, colors[i][j], 0)
        
        # Create a mask for blending (any channel > 0)
        blend_mask = (np.sum(mask, axis=2) > 0)[..., np.newaxis]
        blend_mask = np.repeat(blend_mask, image.shape[2], axis=2)
        
        # Overlay the colored mask on the image
        output = np.where(blend_mask, 
                         (1 - alpha) * output + alpha * colored_mask, 
                         output)
    
    # Ensure output values are within valid range for drawing contours
    output_uint8 = np.clip(output, 0, 255).astype(np.uint8)
    
    # Draw contours for each mask
    for i, mask in enumerate(masks):
        # Create binary mask for contour detection
        binary_mask = (np.sum(mask, axis=2) > 0).astype(np.uint8)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Determine border color
        if border_color is None:
            # Make the border slightly brighter than the mask color
            # but preserve the hue
            b_color = tuple([min(int(c * 1.5), 255) for c in colors[i]])
        else:
            b_color = border_color
            
        # Draw contours on the output image
        cv2.drawContours(output_uint8, contours, -1, b_color, border_thickness)
    
    return output_uint8

def calculate_iou(pred_mask, gt_mask, true_pos_only=True):
    """
    Calculate IoU score between two segmentation masks.

    Args:
        pred_mask (np.array): binary segmentation mask
        gt_mask (np.array): binary segmentation mask
    Returns:
        iou_score (np.float64)
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    if true_pos_only:
        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    else:
        if np.sum(union) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))

    return iou_score

def calculate_precision_recall_specificity(conf_matrix):
    TP = conf_matrix['TP']
    TN = conf_matrix['TN']
    FP = conf_matrix['FP']
    FN = conf_matrix['FN']

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)

    return precision, recall, specificity

def get_conf_matrix(seg_mask, gt_mask):
    TP = np.sum(np.logical_and(seg_mask == 1, gt_mask == 1))
    TN = np.sum(np.logical_and(seg_mask == 0, gt_mask == 0))
    FP = np.sum(np.logical_and(seg_mask == 1, gt_mask == 0))
    FN = np.sum(np.logical_and(seg_mask == 0, gt_mask == 1))

    return dict(TP=TP, TN=TN, FP=FP, FN=FN)

def get_image_with_mask_overlay(input_image, pred_mask, gt_mask, threshold=np.nan):
    org_scan = (np.repeat(input_image.transpose(1, 2, 0), 3, axis=2) * 255.).astype(np.uint8)
    pred_mask_overlay = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)
    gt_mask_overlay = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=2)

    overlaid_scan = overlay_masks(org_scan, [pred_mask_overlay, gt_mask_overlay], colors=[(255, 0, 255), (0, 0, 255)],border_thickness=1)
    overlaid_scan = overlaid_scan.transpose(2, 0, 1)
    return overlaid_scan

def create_mask_batch(input_images, target_images, names, gt_mask_dir, threshold=np.nan):
    pred_mask_list = []
    gt_mask_list = []
    for i in range(input_images.shape[0]):
        pred_mask, gt_mask = create_mask(input_images[i], target_images[i], names[i], gt_mask_dir, threshold)
        pred_mask_list.append(pred_mask)
        gt_mask_list.append(gt_mask)
    return np.array(pred_mask_list), np.array(gt_mask_list)

def create_mask(input_image, target_image, name, gt_mask_dir, threshold=np.nan):
    '''
    Args:
    -----
        input_image, target_image (numpy.ndarray): (C x H x W) image
        name: str
        gt_mask_dir: str
        threshold: float or np.nan

    Returns:
    --------
        pred_mask
            predicted mask generated from heatmap
        gt_mask
            ground truth mask loaded from chexlocalize
    '''
    diff = np.array(abs(input_image[0] - target_image[0]))
    pred_mask = cam_to_segmentation(diff, threshold=threshold, smoothing=True, k=7).astype(np.uint8)
    
    try:
        gt_mask_img = Image.open(f'{gt_mask_dir}/{name}_Pleural Effusion_mask.png')
    except FileNotFoundError:
        print('mask not found')
        gt_mask_img = Image.new('1', input_image.squeeze().shape)
    gt_mask = np.array(gt_mask_img, dtype=np.uint8)

    return pred_mask, gt_mask

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

def generate_grid_for_mpl(npz_paths, dest_path='', x_labels=None, generate_masks=False):
    master_image = None
    master_samples = []
    master_diffs = []
    master_overlays = []

    img_grid = []
    y_labels = None
    y_label_colors = None
    if generate_masks:
        col_attr_length = len(npz_paths) * 3 + 1
    else:
        col_attr_length = len(npz_paths) * 2 + 1

    for i, path in enumerate(npz_paths):
        sample_data = load_samples(path)

        if i == 0:
            batch_size = len(sample_data['names'])
            y_labels = sample_data['names']
            y_label_colors = sample_data['org_labels']
            if x_labels is None:
                x_labels = range(col_attr_length)
            assert len(x_labels) == (col_attr_length)

            master_image = np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1)
            if master_image.shape[2] == 512:
                master_image = rescale_image(master_image, 0.5)
            img_list = np.array_split(master_image, batch_size, axis=1)
            for img in img_list:
                img_grid.append([img])
        
        master_sample = np.concatenate(np.repeat(sample_data['samples'], 3, axis=1), axis=1)
        if master_sample.shape[2] == 512:
            master_sample = rescale_image(master_sample, 0.5)
        master_samples.append(master_sample)
        
        master_diff = generate_diff_batch(sample_data['originals'], sample_data['samples'])
        if master_diff.shape[2] == 512:
            master_diff = rescale_image(master_diff, 0.5)
        master_diffs.append(master_diff)

        if generate_masks:
            pred_masks, gt_masks = create_mask_batch(sample_data['originals'],
                                                    sample_data['samples'],
                                                    sample_data['names'],
                                                    args.gt_mask_path,
                                                    args.pred_mask_threshold)
            pred_masks = np.repeat(np.concatenate(pred_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            gt_masks = np.repeat(np.concatenate(gt_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            master_org = (np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1).transpose(1, 2, 0) * 255).astype(np.uint8)
            master_overlay = overlay_masks(master_org, [pred_masks, gt_masks], [(255, 0, 255), (0, 0, 255)], border_thickness=1)
            master_overlays.append((master_overlay / 255.).transpose(2, 0, 1))

    for sample_col in master_samples:
        img_list = np.array_split(sample_col, batch_size, axis=1)
        for i, img in enumerate(img_list):
            img_grid[i].append(img)

    for sample_col in master_diffs:
        img_list = np.array_split(sample_col, batch_size, axis=1)
        for i, img in enumerate(img_list):
            img_grid[i].append(img)
    
    if generate_masks:
        for sample_col in master_overlays:
            img_list = np.array_split(sample_col, batch_size, axis=1)
            for i, img in enumerate(img_list):
                img_grid[i].append(img)

    nrow = len(img_grid)
    ncol = col_attr_length

    fig = plt.subplots(figsize=(ncol * 2.56, nrow * 2.56), dpi=100, layout='tight')
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0)

    fontdict = {'size': 16}

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
                ax.set_xlabel(x_labels[col], wrap=True, fontdict=fontdict)
                ax.xaxis.set_label_position('top')
            if col == 0:
                ax.set_ylabel(y_labels.tolist()[row].split('_')[0], wrap=True, fontdict=fontdict)
                if y_label_colors.tolist()[row] == 1:
                    ax.yaxis.label.set_color('red')
                ax.yaxis.set_label_position('left')
    
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(dest_path, 'chexpert_eval_mpl_test.png'), dpi=100)

def generate_metrics(npz_paths, include_iou=False):
    results = []
    for i, path in enumerate(npz_paths): # This is a column
        sample_data = load_samples(path)
        results.append(calculate_metric_scores(sample_data['originals'], sample_data['samples'], sample_data['names'], include_iou=include_iou))
    return results

def generate_master_image(npz_paths, generate_masks=False):
    master_image = None
    master_samples = []
    master_diffs = []
    master_overlays = []
    results = []

    for i, path in enumerate(npz_paths): # This is a column
        sample_data = load_samples(path)
        results.append(calculate_metric_scores(sample_data['originals'], sample_data['samples'], sample_data['names'], 0.7))

        if i == 0:
            master_image = np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1)
            if master_image.shape[2] == 512:
                master_image = rescale_image(master_image, 0.5)
        
        master_sample = np.concatenate(np.repeat(sample_data['samples'], 3, axis=1), axis=1)
        if master_sample.shape[2] == 512:
            master_sample = rescale_image(master_sample, 0.5)
        master_samples.append(master_sample)
        
        master_diff = generate_diff_batch(sample_data['originals'], sample_data['samples'])
        if master_diff.shape[2] == 512:
            master_diff = rescale_image(master_diff, 0.5)
        master_diffs.append(master_diff)

        if generate_masks:
            pred_masks, gt_masks = create_mask_batch(sample_data['originals'],
                                                    sample_data['samples'],
                                                    sample_data['names'],
                                                    args.gt_mask_path,
                                                    args.pred_mask_threshold)
            pred_masks = np.repeat(np.concatenate(pred_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            gt_masks = np.repeat(np.concatenate(gt_masks, axis=0)[:, :, np.newaxis], 3, axis=2)
            master_org = (np.concatenate(np.repeat(sample_data['originals'], 3, axis=1), axis=1).transpose(1, 2, 0) * 255).astype(np.uint8)
            master_overlay = overlay_masks(master_org, [pred_masks, gt_masks], [(255, 0, 255), (0, 0, 255)], border_thickness=1)
            master_overlays.append((master_overlay / 255.).transpose(2, 0, 1))

    
    master_samples = visualize(np.concatenate(np.array(master_samples), axis=2))
    master_diffs = visualize(np.concatenate(np.array(master_diffs), axis=2))
    if generate_masks:
        master_overlays = visualize(np.concatenate(np.array(master_overlays), axis=2))
        return np.concatenate([master_image, master_samples, master_diffs, master_overlays], axis=2), results
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
        diff = generate_diff(original[i][0], reconstructed[i][0])
        diff_list.append(diff)
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
    os.makedirs(args.result_dir, exist_ok=True)
    x_labels = ['Original']
    scenarios = args.x_labels
    x_labels.extend(scenarios)
    x_labels.extend([f'Heatmap ({x})' for x in scenarios])
    if args.include_miou:
        x_labels.extend([f'Overlay ({x})' for x in scenarios])
    generate_grid_for_mpl(args.npz_paths, dest_path=args.result_dir, x_labels=x_labels, generate_masks=args.include_miou)

    # final_image = generate_master_image(args.npz_paths, generate_masks=args.include_miou)
    # final_image = (final_image.transpose(1, 2, 0) * 255).astype(np.uint8)
    # Image.fromarray(final_image).save(os.path.join(args.result_dir, 'compiled_master_image.png'))

    final_metrics = generate_metrics(args.npz_paths, args.include_miou)

    if args.include_miou:
        print(f'Model,PSNR,SSIM,FID,mIoU,Specificity')
        for scenario, metric in zip(scenarios, final_metrics):
            print(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f},{metric['miou'].item():02.3f},{metric['specificity'].item():02.3f}")

        with open(os.path.join(args.result_dir, 'metrics.csv'), 'w') as csvfile:
            csvfile.write(f'Model,PSNR,SSIM,FID,mIoU,Specificity\n')
            for scenario, metric in zip(scenarios, final_metrics):
                csvfile.write(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f},{metric['miou'].item():02.3f},{metric['specificity'].item():02.3f}\n")
    else:
        print(f'Model,PSNR,SSIM,FID')
        for scenario, metric in zip(scenarios, final_metrics):
            print(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f}")

        with open(os.path.join(args.result_dir, 'metrics.csv'), 'w') as csvfile:
            csvfile.write(f'Model,PSNR,SSIM,FID,mIoU,Specificity\n')
            for scenario, metric in zip(scenarios, final_metrics):
                csvfile.write(f"{scenario},{metric['psnr'].item():02.3f},{metric['ssim'].item():02.3f},{metric['fid'].item():02.3f}\n")

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--result_dir", required=False, default=os.path.join('results', 'latest'), type=str)
    parser.add_argument(f"--npz_paths", required=True, type=str, nargs='+')
    parser.add_argument(f"--x_labels", required=False, default=None, type=str, nargs='+')
    parser.add_argument(f"--include_miou", required=False, default=False, type=str2bool)
    parser.add_argument(f"--gt_mask_path", type=str, default='/workspace/CheXlocalize/256_segmentations_pleural_effusion')
    parser.add_argument(f"--pred_mask_threshold", type=float, default=np.nan)
    return parser

if __name__ == '__main__':
    args = create_argparser().parse_args()
    main()

