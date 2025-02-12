"""Model loader for huggingface/diffusers JAX models."""

# pylint: disable=g-importing-member
# pylint: disable=unused-import

import os
import time

import constants
from diffusers import FlaxStableDiffusionXLPipeline
from flax.jax_utils import replicate
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
import jax.numpy as jnp
from log_config import init_logger
from utils import aot_compile

from util import image_format_converter

cc.initialize_cache(constants.CACHE_FOLDER)

NUM_DEVICES = jax.device_count()
NUM_STEPS = int(os.getenv(constants.KEY_NUM_STEPS, constants.DEFAULT_NUM_STEPS))
HEIGHT = int(os.getenv(constants.KEY_HEIGHT, constants.DEFAULT_HEIGHT))
WIDTH = int(os.getenv(constants.KEY_WIDTH, constants.DEFAULT_WIDTH))

logger = init_logger(__name__)

pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    constants.KEY_MODEL_NAME_SDXL_BASE_1_0,
    revision=constants.KEY_MODEL_REVISION_SDXL_BASE_1_0,
    split_head_dim=True,
)

scheduler_state = params.pop(constants.KEY_SCHEDULER)
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params[constants.KEY_SCHEDULER] = scheduler_state

p_params = replicate(params)

start = time.time()
logger.info("Compiling the stable diffusion pipeline ...")
p_generate = aot_compile(
    pipeline=pipeline,
    p_params=p_params,
    num_devices=NUM_DEVICES,
    prompt=constants.DEFAULT_PROMPT,
    negative_prompt=constants.DEFAULT_NEG_PROMPT,
    seed=constants.DEFAULT_SEED,
    guidance_scale=constants.DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=NUM_STEPS,
    height=HEIGHT,
    width=WIDTH,
)
logger.info("Compiled in %.2f seconds.", time.time() - start)