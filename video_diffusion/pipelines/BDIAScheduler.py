from diffusers.schedulers import DDIMScheduler

from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from typing import Optional, Union, Tuple, List

import torch

# class BDIASpatioTemporalStableDiffusionPipeline(DDIMSpatioTemporalStableDiffusionPipeline):
    # def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    #     """
    #     Assume the eta in DDIM=0
    #     """
    #     timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
    #     alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
    #     alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
    #     beta_prod_t = 1 - alpha_prod_t
    #     next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    #     next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    #     next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    #     return next_sample




class BDIAScheduler(DDIMScheduler):
    # def __init__(
    #         self,
    #         num_train_timesteps: int = 1000,
    #         beta_start: float = 0.0001,
    #         beta_end: float = 0.02,
    #         beta_schedule: str = "linear",
    #         trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
    #         clip_sample: bool = True,
    #         set_alpha_to_one: bool = True,
    #         steps_offset: int = 0,
    #         prediction_type: str = "epsilon",
    #         **kwargs,
    # ):
    #     message = (
    #         "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
    #         " DDIMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
    #     )
    #     predict_epsilon = deprecate("predict_epsilon", "0.12.0", message, take_from=kwargs)
    #     if predict_epsilon is not None:
    #         self.register_to_config(prediction_type="epsilon" if predict_epsilon else "sample")
    #
    #     if trained_betas is not None:
    #         self.betas = torch.tensor(trained_betas, dtype=torch.float32)
    #     elif beta_schedule == "linear":
    #         self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    #     elif beta_schedule == "scaled_linear":
    #         # this schedule is very specific to the latent diffusion model.
    #         self.betas = (
    #                 torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
    #         )
    #     elif beta_schedule == "squaredcos_cap_v2":
    #         # Glide cosine schedule
    #         self.betas = betas_for_alpha_bar(num_train_timesteps)
    #     else:
    #         raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
    #
    #     self.alphas = 1.0 - self.betas
    #     self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    #
    #     # At every step in ddim, we are looking into the previous alphas_cumprod
    #     # For the final step, there is no previous alphas_cumprod because we are already at 0
    #     # `set_alpha_to_one` decides whether we set this parameter simply to one or
    #     # whether we use the final alpha of the "non-previous" one.
    #     self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
    #
    #     # standard deviation of the initial noise distribution
    #     self.init_noise_sigma = 1.0
    #
    #     # setable values
    #     self.num_inference_steps = None
    #     self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
    #
    #     self.x_last = None
    #     self.t_last = None
    #     self.gamma = 0.5

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if not getattr(self, "_runtime_inited", False):
            # first-time initialization
            self.x_last = None
            self.x_last2 = None
            self.t_last = None
            self.gamma = 0.9
            print("gamma:", self.gamma, "type:", type(self.gamma))
            self._runtime_inited = True
            print("gamma:", self.gamma)
            self.gamma1 = 0.9
            self.gamma2 = 0.9

        # reset whenever timestep == 981
        if timestep ==981:
            self.x_last = None
            self.t_last = None
            print("gamma:", self.gamma)

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        print("prev_timestep:",prev_timestep)
        print("timestep:",timestep)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        print("timestep:",timestep)


        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        if self.x_last is None:
            print("self.x_last is None")
        else:
            print(timestep)
            a_last = self.alphas_cumprod[timestep + self.config.num_train_timesteps // self.num_inference_steps]
            print("start")
            print(timestep + self.config.num_train_timesteps // self.num_inference_steps)
            print("a_last:",a_last)
            print("end")
                                                                         
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # # additional code added by Guoqiang
        if self.x_last is not None:
            # 与调度一致的步长和索引
            step = self.config.num_train_timesteps // self.num_inference_steps
            t_int = int(timestep)
            print(f"{timestep=}")
            # t+Δ（a_last 你上面已算过；若没有，可取消注释下一行）
            a_last = self.alphas_cumprod[min(t_int + step, len(self.alphas_cumprod) - 1)]
            print(t_int + step)
            #a_last2 = self.alphas_cumprod[min(t_int + 2 * step, len(self.alphas_cumprod) - 1)]
            #print(t_int + 2* step)
            # 把 t+Δ / t+2Δ 在时刻 t 的等价表示（仅用 x0_hat 与 eps_hat）
            #x_next_to_t = a_last.sqrt() * pred_original_sample + (1. - a_last).sqrt() * model_output
            #x_next2_to_t = a_last2.sqrt() * pred_original_sample + (1. - a_last2).sqrt() * model_output

            # gamma1/gamma2（若未显式赋值，则回退到 self.gamma）
            # gamma1 = getattr(self, "gamma1", self.gamma)
            # gamma2 = getattr(self, "gamma2", self.gamma)

            # 若还没有两步历史，先用 x_{t+2Δ→t} 近似
            # self.x_last2 = getattr(self, "x_last2", x_next2_to_t)

            print("12345")
            prev_sample = (self.x_last - (1 - self.gamma) * (self.x_last - sample)
                           - self.gamma * (a_last.sqrt() * pred_original_sample
                                           + (1. - a_last).sqrt() * model_output - sample)
                           + alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction - sample
                           )
            print("gamma:", self.gamma)
            # prev_sample = (
            #         self.x_last2
            #         - (1.0 - self.gamma2) * (self.x_last2 - self.x_last)
            #         - self.gamma2 * (x_next2_to_t - x_next_to_t)
            #         - (1.0 - self.gamma1) * (self.x_last - sample)
            #         - self.gamma1 * (x_next_to_t - sample)
            #         + (alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction - sample)
            # )

        else:
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            print("54321")
            prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                if device.type == "mps":
                    # randn does not work reproducibly on mps
                    variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
                    variance_noise = variance_noise.to(device)
                else:
                    variance_noise = torch.randn(
                        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                    )
            variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * variance_noise

            prev_sample = prev_sample + variance

        print("adding")
        self.x_last = sample
        self.t_last = timestep
        # self.x_last2 = self.x_last
        # self.x_last = sample

        if not return_dict:
             return (
                 prev_sample,
                 pred_original_sample,
             )

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
