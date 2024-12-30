import numpy as np
import torch as th
import pytorch_wavelets
from visdom import Visdom
viz = Visdom(port=8850)
from .train_util import visualize

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    print('num_timesteps', num_timesteps)
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            print('desired_cound', desired_count )
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    print('sectioncount', section_counts)
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    print('all steps', set(all_steps))
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
        print('self.orig',self.original_num_steps )
        print('use_timesteps',set(use_timesteps))

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )
   

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t

class MASFDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
        print('self.orig',self.original_num_steps )
        print('use_timesteps',set(use_timesteps))

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

        # Parameters for DWT, IDWT, and MA calculations
        self.prior_sub_bands = None
        self.wavelet = 'haar'
        self.padding_scheme = 'symmetric'
        self.xfm = pytorch_wavelets.DWTForward(J=1, wave=self.wavelet, mode=self.padding_scheme).cuda()
        self.ifm = pytorch_wavelets.DWTInverse(wave=self.wavelet, mode=self.padding_scheme).cuda()
        self.channels = 1
        self.use_adaptive_weighting = False

    def _decompose_image(self, img):
        '''
        yh is a tensor of shape (N, C, 3, H, W), where 3 represents the 3 sub bands of LH, HL, HH

        Args:
            Image tensor of shape (N, C, H, W)

        Returns:
            Frequency sub-bands of shape (N, C, H, W) for LL, LH, HL, HH
        '''
        ll, high = self.xfm(img)
        lh, hl, hh = th.unbind(high[0], dim=2)
        return (ll, [lh, hl, hh])

    def _recompose_image(self, sub_bands):
        '''
        yh is a tensor of shape (N, C, 3, H, W), where 3 represents the 3 sub bands of LH, HL, HH

        Args:
            Frequency sub-bands of shape (N, C, H, W) for LL, LH, HL, HH

        Returns:
            Image tensor of shape (N, C, H, W)
        '''
        ll, high = sub_bands
        # print(len(high), high[0].shape)
        high = [th.stack(tuple(high), axis=2)]
        # print(high[0].shape)
        return self.ifm((ll, high))

    def _update_sample_ma(self, sub_bands, gamma=0.9):
        """
        Args:
            sub_bands: Tuple of (LL, [LH, HL, HH]).
            gamma: Weight for moving average (default: 0.9).

        Returns:
            Updated frequency sub-bands (LL, [LH, HL, HH]).
        """
        if self.prior_sub_bands is None:
            self.prior_sub_bands = sub_bands
        
        updated_sub_bands = []
        for prev, curr in zip(self.prior_sub_bands, sub_bands):
            if isinstance(curr, list): # list(LH, HL, HH)
                updated_sub_band = [(1 - gamma) * p + gamma * c for p, c in zip(prev, curr)]
                updated_sub_bands.append(updated_sub_band)
            else: # LL
                updated_sub_bands.append((1 - gamma) * prev + gamma * curr)
        
        self.prior_sub_bands = updated_sub_bands
        return updated_sub_bands

    def perform_moving_average(self, x):
        sub_bands = self._decompose_image(x)
        new_sub_bands = self._update_sample_ma(sub_bands)
        x_bar = self._recompose_image(new_sub_bands)

        return x_bar

    def ddim_sample_loop_known(
        self,
        model,
        shape,
        img,
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        noise_level=500,
        progress=False,
        conditioning=False,
        conditioner=None,
        classifier=None,
        eta=0
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(0,1, (b,), device=device).long().to(device)
        org = img[0].to(device)
        img = img[0].to(device)
        
        indices = list(range(t))[::-1]
        noise = th.randn_like(img).to(device)
        x_noisy = self.q_sample(x_start=img, t=t, noise=noise).to(device)
        print('xnoisy', x_noisy.shape)

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=noise_level,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        # viz.image(visualize(final["sample"].cpu()[0,0, ...]), opts=dict(caption="final 0" ))
        # Comment the following three lines if running with CheXpert
        # viz.image(visualize(final["sample"].cpu()[0,1, ...]), opts=dict(caption="final 1" ))
        # viz.image(visualize(final["sample"].cpu()[0,2, ...]), opts=dict(caption="final 2" ))
        # viz.image(visualize(final["sample"].cpu()[0,3, ...]), opts=dict(caption="final 3" ))
        self.prior_sub_bands = None

        return final["sample"], x_noisy, img
    
    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(time-1))[::-1]
        print('indices', indices)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:

            k=abs(time-1-i)
            if k%20==0:
                print('k',k)

            t = th.tensor([k] * shape[0], device=device)
            with th.no_grad():

                out = self.ddim_reverse_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

                yield out
                img = out["sample"]
                # if k%50==0:
                #     viz.image(visualize(img.cpu()[0,0, ...]), opts=dict(caption=f"reversesample {k}"))

        for i in indices:
            k=abs(time-1-i)
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=self.perform_moving_average,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
            yield out
            img = out["sample"]
            saliency=out['saliency']
            # if k%50==0:
            #     viz.image(visualize(img.cpu()[0,0, ...]), opts=dict(caption=f"sample {k}"))
        
        self.prior_sub_bands = None

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )
   

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t

class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps


    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)



