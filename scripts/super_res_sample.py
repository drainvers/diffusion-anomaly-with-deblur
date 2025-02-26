"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
from visdom import Visdom
viz = Visdom(port=8850)
import sys
sys.path.append("..")
sys.path.append(".")
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.train_util import visualize
from guided_diffusion.image_datasets import load_data, _list_image_files_recursively
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    # datal = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    if args.dataset=='brats':
        ds = BRATSDataset(args.base_samples, test_flag=True)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False)
    
    elif args.dataset=='chexpert':
        data = load_superres_data_sample(
            data_dir=args.base_samples,
            batch_size=args.batch_size,
            small_size=args.small_size,
            class_cond=False,
            deterministic=True
        )
        datal = iter(data)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        img = next(datal)
        viz.image(visualize(img['low_res']), opts=dict(caption=f'img input')) # {img["path"]}

        model_kwargs = img
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        viz.image(visualize(sample), opts=dict(caption=f'sampled output 2x'))
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

def load_superres_data_sample(data_dir, batch_size, small_size, class_cond=False, deterministic=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=small_size,
        class_cond=class_cond,
        deterministic=deterministic
    )
    for small_batch, model_kwargs in data:
        batch = small_batch * 2.0 - 1.0 # Make [0., 1.] -> [-1., 1.]
        res = dict(low_res=batch)
        print(res["low_res"].dtype)
        print(res["low_res"])
        if class_cond:
            res["y"] = model_kwargs["y"]
            res["path"] = model_kwargs["path"]
        yield res

def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        result_dir='./results',
        dataset='chexpert',
        small_size=256
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()