import ctypes
import torch
import os
import inspect
import random
import sys
import argparse
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, T5TokenizerFast


def seed_torch(seed=1029):
    seed=seed%4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)


def make_torch2c(tensor: torch.Tensor):
    if not tensor.is_contiguous():
        print("may error")
        tensor = tensor.contiguous()
    ptr = tensor.data_ptr()
    return ptr


int_point = ctypes.POINTER(ctypes.c_int)
int_ = ctypes.c_int
ulonglong = ctypes.c_ulonglong
cpoint = ctypes.c_void_p
vpoint = ctypes.c_void_p
spoint = ctypes.c_char_p
bool_ = ctypes.c_bool
null_ptr = ctypes.c_void_p(None)
ref = lambda x: ctypes.byref(x)


def make2_c_uint64_list(my_list):
    return (ctypes.c_uint64 * len(my_list))(*my_list)


def make2_c_int_list(my_list: list):
    return (ctypes.c_int * len(my_list))(*my_list)


def char_point_2_str(char_point: ctypes.c_char_p):
    return ctypes.string_at(char_point).decode('utf-8')


def str2char_point(string: str):
    return ctypes.c_char_p(string.encode('utf-8'))


def make2_c_point_list(my_list: list):
    return (ctypes.c_void_p * len(my_list))(*my_list)


def build_c_torch_lists(args):
    # need be torch
    return make2_c_point_list([make_torch2c(i) for i in args])


str2cpoint = str2char_point


def make2_c_string_list(my_list):
    array = [str2char_point(i) for i in my_list]
    return (ctypes.c_char_p * len(array))(*array)


class Builder:

    def __init__(self, so_path: str = "./build/libsd3.so"):
        self.so_path = so_path
        self.lib = ctypes.CDLL(self.so_path)
        self.lib_init()
        self.lib_t5()
        self.lib_flux_schnell_w4bf16()
        self.lib_clip()
        self.lib_vae()
        self.lib_flux_schnell_multi_device()
        self.lib_flux_dev_w4bf16()
        self.lib_flux_dev_multi_device()

    def lib_t5(self):
        # struct t5_encoder * t5_encoder_init(const char* filename, int device_id, const char* weight_file);
        self.lib.t5_encoder_init.argtypes = [spoint, int_, spoint]
        self.lib.t5_encoder_init.restype = vpoint
        # int t5_encoder_run(struct t5_encoder *encoder, void* data, void* output);
        self.lib.t5_encoder_run.argtypes = [vpoint, cpoint, cpoint]
        self.lib.t5_encoder_run.restype = int_
        # int t5_encoder_free(struct t5_encoder *encoder);
        self.lib.t5_encoder_free.argtypes = [vpoint]
        self.lib.t5_encoder_free.restype = int_

    def lib_clip(self):
        # struct clip_g_encoder * clip_g_encoder_init(const char* filename, int device_id);
        # int flux_schnell_run(struct flux_schnell *flux_schnell, void* input0, void* input1, void* input2, void* input3, void* rotary_emb, void* output);
        # int clip_g_free(struct clip_g_encoder *encoder);
        self.lib.clip_pooling_init.argtypes = [spoint, int_]
        self.lib.clip_pooling_init.restype = vpoint
        self.lib.clip_pooling_run.argtypes = [vpoint, cpoint, cpoint]
        self.lib.clip_pooling_run.restype = int_
        self.lib.clip_pooling_free.argtypes = [vpoint]
        self.lib.clip_pooling_free.restype = int_

    def lib_flux_schnell_w4bf16(self):
        # struct flux_schnell * flux_schnell_init(const char* filename, const char* image_rotary_emb_weight_file, int device_id);
        self.lib.flux_schnell_init.argtypes = [spoint, int_]
        self.lib.flux_schnell_init.restype = vpoint
        # inline int flux_schnell_run(struct flux_schnell *flux_schnell, void* input0, void* input1, void* input2, void* input3, void* output);
        self.lib.flux_schnell_run.argtypes = [vpoint, cpoint, cpoint, cpoint, cpoint, cpoint, cpoint, int_]
        self.lib.flux_schnell_run.restype = int_
        # int flux_schnell_free(struct flux_schnell *flux_schnell);
        self.lib.flux_schnell_free.argtypes = [vpoint]

    def lib_flux_dev_w4bf16(self):
        self.lib.flux_dev_init.argtypes = [spoint, int_]
        self.lib.flux_dev_init.restype = vpoint
        self.lib.flux_dev_run.argtypes = [vpoint, cpoint, cpoint, cpoint, cpoint, cpoint, cpoint, cpoint, int_]
        self.lib.flux_dev_run.restype = int_
        self.lib.flux_dev_free.argtypes = [vpoint]

    def lib_flux_dev_multi_device(self):
        self.lib.flux_dev_multi_device_init.argtypes = [vpoint, vpoint]
        self.lib.flux_dev_multi_device_init.restype = vpoint
        self.lib.flux_dev_multi_device_run.argtypes = [vpoint, cpoint, cpoint, cpoint, cpoint, cpoint, cpoint, cpoint,
                                                       int_]
        self.lib.flux_dev_multi_device_run.restype = int_
        self.lib.flux_dev_multi_device_free.argtypes = [vpoint]

    def lib_vae(self):
        # struct vae_decoder* vae_decoder_init(const char* filename, int device_id);
        # int vae_decoder_run(struct vae_decoder *decoder, void* latent, void* img, bool do_post_process);
        # int vae_decoder_free(struct vae_decoder *decoder);
        self.lib.vae_decoder_init.argtypes = [spoint, int_]
        self.lib.vae_decoder_init.restype = vpoint
        self.lib.vae_decoder_run.argtypes = [vpoint, cpoint, cpoint, bool_]
        self.lib.vae_decoder_run.restype = int_
        self.lib.vae_decoder_free.argtypes = [vpoint]
        self.lib.vae_decoder_free.restype = int_

    def lib_flux_schnell_multi_device(self):
        # struct flux_schnell_device_3 * flux_schnell_multi_device_init(const char** filename, int* device_ids);
        self.lib.flux_schnell_multi_device_init.argtypes = [vpoint, vpoint]
        self.lib.flux_schnell_multi_device_init.restype = vpoint
        # int flux_schnell_multi_device_run(struct flux_schnell_device_3 *flux_schnell, void* input0, void* input1, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack);
        self.lib.flux_schnell_multi_device_run.argtypes = [vpoint, cpoint, cpoint, cpoint, cpoint, cpoint, cpoint, int_]
        self.lib.flux_schnell_multi_device_run.restype = int_
        # int flux_schnell_multi_device_free(struct flux_schnell_device_3 *flux_schnell);
        self.lib.flux_schnell_multi_device_run.argtypes = [vpoint]

    def lib_init(self):
        self.lib.run_model.argtypes = [spoint, spoint, cpoint, int_, cpoint, int_, int_]


builder = Builder()


def calculate_shift(image_seq_len, base_seq_len: int = 256, max_seq_len: int = 4096, base_shift: float = 0.5,
                    max_shift: float = 1.16, ):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs, ):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class Clip_Pooling_Encoder:
    def __init__(self, path, device_id=0, batch=1, max_seq=77, hidden_size=768, return_dytpe=torch.float32,
                 builder=None):
        self.builder = builder
        self.model = builder.lib.clip_pooling_init(str2char_point(path), device_id)
        self.batch = batch
        self.max_seq = max_seq
        self.hidden_size = hidden_size
        self.dtype = return_dytpe

    def __call__(self, input_tokens, *args, **kwargs):
        input_tokens = input_tokens.float()
        pooling_embeds = torch.ones(self.batch, self.hidden_size, dtype=self.dtype, layout=torch.strided)
        self.builder.lib.clip_pooling_run(self.model,
                                          make_torch2c(input_tokens),
                                          make_torch2c(pooling_embeds))
        return pooling_embeds

    def __del__(self):
        self.builder.lib.clip_pooling_free(self.model)


class T5Bmodel:
    def __init__(self, path, device_id=0, cpu_weight_path="", batch=1, max_seq=512, hidden_size=4096,
                 return_dytpe=torch.float32, builder=None):
        self.builder = builder
        self.model = builder.lib.t5_encoder_init(str2char_point(path), device_id, str2char_point(cpu_weight_path))
        self.batch = batch
        self.max_seq = max_seq
        self.hidden_size = hidden_size
        self.dtype = return_dytpe
        self.path = path

    def __call__(self, input_tokens, *args, **kwargs):
        input_tokens = input_tokens.int()
        res = torch.ones(self.batch, self.max_seq, self.hidden_size, dtype=self.dtype, layout=torch.strided)
        self.builder.lib.t5_encoder_run(self.model,
                                        make_torch2c(input_tokens),
                                        make_torch2c(res))
        return res

    def __del__(self):
        self.builder.lib.t5_encoder_free(self.model)


class SchnellBmodel3:
    def __init__(self, paths, devices, builder=None):
        self.builder = builder
        self.paths = paths
        self.devices = devices
        assert (len(self.paths) == 3)
        assert (len(self.devices) == 3)
        self.model = builder.lib.flux_schnell_multi_device_init(make2_c_string_list(self.paths),
                                                                make2_c_int_list(self.devices))

    # TODO:guidance as input is not neccessary
    def __call__(self, latents, timestep, guidance, pooled_prompt_embeds, prompt_embeds, rotary_emb, do_unpack=0):
        latents = latents.float()
        predict_noise = torch.zeros_like(latents)
        timestep = timestep.float()
        pooled_prompt_embeds = pooled_prompt_embeds.float()
        prompt_embeds = prompt_embeds.float()
        rotary_emb = rotary_emb.float()
        self.builder.lib.flux_schnell_multi_device_run(self.model,
                                                       make_torch2c(latents),
                                                       make_torch2c(timestep),
                                                       make_torch2c(pooled_prompt_embeds),
                                                       make_torch2c(prompt_embeds),
                                                       make_torch2c(rotary_emb),
                                                       make_torch2c(predict_noise),
                                                       do_unpack)
        return predict_noise

    def __del__(self):
        self.builder.lib.flux_schnell_multi_device_free(self.model)


class SchnellBmodel:
    def __init__(self, path, device_id, return_dtype=torch.float32, builder=None):
        self.builder = builder
        self.model = builder.lib.flux_schnell_init(str2char_point(path), device_id)

    # TODO:guidance not neccessay
    def __call__(self, latents, timestep, guidance, pooled_prompt_embeds, prompt_embeds, rotary_emb, do_unpack=0):
        latents = latents.float()
        predict_noise = torch.zeros_like(latents)
        timestep = timestep.float()
        pooled_prompt_embeds = pooled_prompt_embeds.float()
        prompt_embeds = prompt_embeds.float()
        rotary_emb = rotary_emb.float()
        self.builder.lib.flux_schnell_run(self.model,
                                          make_torch2c(latents),
                                          make_torch2c(timestep),
                                          make_torch2c(pooled_prompt_embeds),
                                          make_torch2c(prompt_embeds),
                                          make_torch2c(rotary_emb),
                                          make_torch2c(predict_noise),
                                          do_unpack)
        return predict_noise

    def __del__(self):
        self.builder.lib.flux_schnell_free(self.model)


class FluxDevBmodel:
    def __init__(self, path, device_id, return_dtype=torch.float32, builder=None):
        self.builder = builder
        self.model = builder.lib.flux_dev_init(str2char_point(path), device_id)

    def __call__(self, latents, timestep, guidance, pooled_prompt_embeds, prompt_embeds, rotary_emb, do_unpack=0):
        latents = latents.float()
        predict_noise = torch.zeros_like(latents)
        timestep = timestep.float()
        guidance = guidance.float()
        pooled_prompt_embeds = pooled_prompt_embeds.float()
        prompt_embeds = prompt_embeds.float()
        rotary_emb = rotary_emb.float()
        self.builder.lib.flux_dev_run(self.model,
                                      make_torch2c(latents),
                                      make_torch2c(timestep),
                                      make_torch2c(guidance),
                                      make_torch2c(pooled_prompt_embeds),
                                      make_torch2c(prompt_embeds),
                                      make_torch2c(rotary_emb),
                                      make_torch2c(predict_noise),
                                      do_unpack)
        return predict_noise

    def __del__(self):
        self.builder.lib.flux_dev_free(self.model)


class FluxDevBmodel3:
    def __init__(self, path, device_id, return_dtype=torch.float32, builder=None):
        self.builder = builder
        assert (len(path) == 3)
        assert (len(device_id) == 3)
        self.model = builder.lib.flux_dev_multi_device_init(make2_c_string_list(path), make2_c_int_list(device_id))

    def __call__(self, latents, timestep, guidance, pooled_prompt_embeds, prompt_embeds, rotary_emb, do_unpack=0):
        latents = latents.float()
        predict_noise = torch.zeros_like(latents)
        timestep = timestep.float()
        guidance = guidance.float()
        pooled_prompt_embeds = pooled_prompt_embeds.float()
        prompt_embeds = prompt_embeds.float()
        rotary_emb = rotary_emb.float()
        self.builder.lib.flux_dev_multi_device_run(self.model,
                                                   make_torch2c(latents),
                                                   make_torch2c(timestep),
                                                   make_torch2c(guidance),
                                                   make_torch2c(pooled_prompt_embeds),
                                                   make_torch2c(prompt_embeds),
                                                   make_torch2c(rotary_emb),
                                                   make_torch2c(predict_noise),
                                                   do_unpack)
        return predict_noise

    def __del__(self):
        self.builder.lib.flux_dev_multi_device_free(self.model)


def build_schnell(paths, device_ids, builder):
    if isinstance(paths, str):
        return SchnellBmodel(paths, device_ids, builder=builder)
    elif isinstance(paths, list):
        return SchnellBmodel3(paths, device_ids, builder=builder)
    else:
        raise ValueError("Invalid paths")


def build_dev(paths, device_ids, builder):
    if isinstance(paths, str):
        return FluxDevBmodel(paths, device_ids, builder=builder)
    elif isinstance(paths, list):
        return FluxDevBmodel3(paths, device_ids, builder=builder)
    else:
        raise ValueError("Invalid paths")


class Vae_Decoder:
    # TODO:self.h and self.w
    def __init__(self, path, device_id, batch=1, h=128, w=128, c=16, oc=3, oh=1024, ow=1024, return_dtype=torch.float32,
                 builder=None):
        self.builder = builder
        self.model = builder.lib.vae_decoder_init(str2char_point(path), device_id)
        self.batch = batch
        self.h = h
        self.w = w
        self.c = c
        self.oc = oc
        self.oh = oh
        self.ow = ow
        self.dtype = return_dtype

    def __call__(self, latent, *args, **kwargs):
        latent = latent.to(self.dtype)
        img = torch.zeros(self.batch, self.oc, self.oh, self.ow, dtype=self.dtype)
        self.builder.lib.vae_decoder_run(self.model,
                                         make_torch2c(latent),
                                         make_torch2c(img),
                                         False)
        return img

    def __del__(self):
        self.builder.lib.vae_decoder_free(self.model)


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def get_timestep_and_steps(scheduler, num_inference_steps):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = 4096
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        "cpu",
        None,
        sigmas,
        mu=mu,
    )
    return timesteps, num_inference_steps


def post_process(img):
    img = (img / 2 + 0.5).clamp(0, 1).numpy()
    img = (img * 255).round().astype("uint8").transpose(0, 2, 3, 1)
    return Image.fromarray(img[0])


randn_tensor = lambda shape, dtype: torch.randn(shape, dtype=dtype)

schnell_schedule_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": False
}

dev_schedule_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True
}


class FluxSchnellPipeline:
    def __init__(self, builder,
                 clip_path,
                 t5_path,
                 transform_path,
                 vae_path,
                 tokenizer1_path,
                 tokenizer2_path,
                 rotary_emb_path,
                 device_id):
        self.builder = builder
        first_device_id = device_id
        if isinstance(device_id, int):
            pass
        else:
            first_device_id = device_id[0]
        self.clip_pooling = Clip_Pooling_Encoder(clip_path, first_device_id, builder=builder)
        self.t5 = T5Bmodel(t5_path, first_device_id, "", builder=builder)
        self.vae = Vae_Decoder(vae_path, first_device_id, builder=builder)
        self.flux_schnell = build_schnell(transform_path, device_id, builder=builder)
        self.tokenizer1 = CLIPTokenizer.from_pretrained(tokenizer1_path)
        self.tokenizer2 = T5TokenizerFast.from_pretrained(tokenizer2_path)
        self.rotary_emb = torch.load(rotary_emb_path, map_location="cpu")
        self.device_id = device_id
        self.scheduler = FlowMatchEulerDiscreteScheduler(**schnell_schedule_config)

    def __call__(self, prompt, prompt2=None, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=512,
                 generator=None):
        prompt2 = prompt2 or prompt
        text_inputs = self.tokenizer1(
            [prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        text_inputs2 = self.tokenizer2(
            [prompt2],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        clip_pooling_res = self.clip_pooling(text_inputs, 0)
        encoder_hidden_states = self.t5(text_inputs2, 0)

        latents = randn_tensor((1, 16, 1024 // 8, 1024 // 8), torch.float32)
        latents = _pack_latents(latents, 1, 16, 128, 128)
        timesteps, num_inference_steps = get_timestep_and_steps(self.scheduler, num_inference_steps)
        img_rotary_emb = self.rotary_emb
        for i in tqdm(range(num_inference_steps)):
            t = timesteps[i]
            predict = self.flux_schnell(latents, t, timesteps[i], clip_pooling_res, encoder_hidden_states,
                                        img_rotary_emb)
            latents = self.scheduler.step(predict, t, latents, return_dict=False)[0]
        latents = _unpack_latents(latents, 1024, 1024, 16)
        img = self.vae(latents)
        return post_process(img)


class FluxDevPipeline:
    def __init__(self, builder,
                 clip_path,
                 t5_path,
                 transform_path,
                 vae_path,
                 tokenizer1_path,
                 tokenizer2_path,
                 rotary_emb_path,
                 device_id):
        self.builder = builder
        first_device_id = device_id
        if isinstance(device_id, int):
            pass
        else:
            first_device_id = device_id[0]
        self.clip_pooling = Clip_Pooling_Encoder(clip_path, first_device_id, builder=builder)
        self.t5 = T5Bmodel(t5_path, first_device_id, "", builder=builder)
        self.vae = Vae_Decoder(vae_path, first_device_id, builder=builder)
        self.flux_dev = build_dev(transform_path, device_id, builder=builder)
        self.tokenizer1 = CLIPTokenizer.from_pretrained(tokenizer1_path)
        self.tokenizer2 = T5TokenizerFast.from_pretrained(tokenizer2_path)
        self.rotary_emb = torch.load(rotary_emb_path, map_location="cpu")
        self.device_id = device_id
        self.scheduler = FlowMatchEulerDiscreteScheduler(**dev_schedule_config)

    def __call__(self, prompt, prompt2=None, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=512,
                 generator=None):
        prompt2 = prompt2 or prompt
        text_inputs = self.tokenizer1(
            [prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        text_inputs2 = self.tokenizer2(
            [prompt2],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        clip_pooling_res = self.clip_pooling(text_inputs, 0)
        encoder_hidden_states = self.t5(text_inputs2, 0)

        latents = randn_tensor((1, 16, 1024 // 8, 1024 // 8), torch.float32)
        latents = _pack_latents(latents, 1, 16, 128, 128)
        timesteps, num_inference_steps = get_timestep_and_steps(self.scheduler, num_inference_steps)
        timesteps, num_inference_steps = get_timestep_and_steps(self.scheduler, num_inference_steps)
        guidance_tensor = torch.tensor(guidance_scale) * 1000
        img_rotary_emb = self.rotary_emb
        for i in tqdm(range(num_inference_steps)):
            t = timesteps[i]
            predict = self.flux_dev(latents, t, guidance_tensor, clip_pooling_res, encoder_hidden_states,
                                    img_rotary_emb)
            latents = self.scheduler.step(predict, t, latents, return_dict=False)[0]
        latents = _unpack_latents(latents, 1024, 1024, 16)
        img = self.vae(latents)
        return post_process(img)


clip_path = "./models/clip_combined.bmodel"
t5_path = "./models/t5_combined.bmodel"
schnell_transform_path = "./models/schnell_w4bf16_transform_combined.bmodel"
dev_transform_path = "./models/dev_w4bf16_transform_combined.bmodel"
vae_path = "./models/vae_decoder_F16.bmodel"
rotary_emb_path = "./models/ids_emb.pt"
tokenizer1_path = "./models/tokenizer"
tokenizer2_path = "./models/tokenizer_2"


def run(args):
    if args.seed is not None:
        seed_torch(args.seed)

    if not os.path.exists('./results'):
        os.makedirs('./results', exist_ok=True)

    count = 0
    if args.models == "schnell":
        flux = FluxSchnellPipeline(builder, clip_path, t5_path, schnell_transform_path, vae_path, tokenizer1_path, tokenizer2_path,
                                   rotary_emb_path, 0)
    elif args.models == "dev":
        flux = FluxDevPipeline(builder, clip_path, t5_path, dev_transform_path, vae_path, tokenizer1_path, tokenizer2_path,
                                   rotary_emb_path, 0)
    print("Example:\nhigh detailed texture, photograph, realistic, RAW photo of young lady, emo, dark hair, makeup, depth of field, soft split ambient window light, soft realistic shadows, perfect composition\n")
    while True:
        prompt = input("Prompt: (type \exit to exit)\n")
        if prompt == "\exit":
            break
        res = flux(
            prompt,
            prompt2=None,
            guidance_scale=args.gfc,
            num_inference_steps=args.steps,
        )
        res.save('./results/{}_result{}.jpg'.format(args.models, count))
        print("image save in results/{}_result{}.jpg".format(args.models, count))
        count += 1

if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    description = "inference full flux pipline on sophon 2300x"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-m', '--models', choices=['dev', 'schnell'], required=True, help="model choices in ['dev', 'schnell']")
    parser.add_argument('-s','--steps', default=4, type=int, help='steps')
    parser.add_argument('-g','--gfc', default=0.0, type=float, help='guidance_scale')
    parser.add_argument('-r','--seed', default=None, type=int, help='random seed')

    args = parser.parse_args()
    run(args)


