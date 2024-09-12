## FLUX.1 [schnell/dev] on SOPHON TPU

![main.jpg](./assets/main.jpg)

This repo contains minimal inference code to run text-to-image with [Flux](https://blackforestlabs.ai/) latent rectified flow transformers on [SOPHON 2300x](https://radxa.com/products/aicore/aicore-sg2300x/) TPU.

FLUX.1 dev is an open-weight, guidance-distilled model for non-commercial applications. Directly distilled from FLUX.1 [pro], FLUX.1 [dev] obtains similar quality and prompt adherence capabilities, while being more efficient than a standard model of the same size.

FLUX.1 schnell model is tailored for local development and personal use. FLUX.1 schnell is openly available under an Apache2.0 license.

FLUX.1 [schnell/dev] models by Black Forest Labs: https://blackforestlabs.ai

---

## Tested Devices
- [**radxa Fogwise® AirBox**](https://radxa.com/products/fogwise/airbox)


## TPU Setting
**Recommend TPU Memory: NPU->7615MB, VPU->2360MB, VPP->2360MB.** [How to modify ?](https://docs.radxa.com/en/sophon/airbox/local-ai-deploy/ai-tools/memory_allocate)

---
## Usage
- Clone this repository
    ```bash
    git clone https://github.com/zifeng-radxa/flux.1_tpu.git
    ```

- Download models from [ModelScope](https://modelscope.cn/models/tpu-mlir/FLUX.1_TPU) via [GLF](https://git-lfs.com/)
    ```bash
    cd flux.1_tpu
    git clone https://www.modelscope.cn/tpu-mlir/FLUX.1_TPU.git
    mv FLUX.1_TPU/ models/
    ```

- Setup environments
    ```bash
    pip3 install -r requirements.txt
    ```

- Compile flux 
    ```bash
    mkdir build && cd build
    cmake .. && make -j
    ```

- Run inference
    ```bash
    # schnell
    python3 flux_pipeline.py --models schnell
    # dev
    python3 flux_pipeline.py -m dev -s 28
    ```
    `flux_pipeline.py` parameters
    ```bash
    usage: flux_pipeline.py [-h] -m {dev,schnell} [-s STEPS] [-g GFC] [-r SEED]
    
    inference full flux pipline on sophon 2300x
    
    optional arguments:
      -h, --help            show this help message and exit
      -m {dev,schnell}, --models {dev,schnell} model choices in ['dev', 'schnell']
      -s STEPS, --steps STEPS steps
      -g GFC, --gfc GFC     guidance_scale
      -r SEED, --seed SEED  random seed
    ```
---

## TODO
- Fix o3 compile segmentation fault
- Add gradio demo
