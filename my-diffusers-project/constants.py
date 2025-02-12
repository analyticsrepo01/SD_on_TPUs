"""Constants for JAX Stable Diffusion."""

from typing import Any

DEFAULT_PROMPT = (
    "a colorful photo of a castle in the middle of a forest with trees and"
    " bushes, shadows, high contrast, dynamic shading, hdr,"
    " detailed vegetation, digital painting, digital drawing, detailed"
    " painting, a detailed digital painting, gothic art"
)
DEFAULT_NEG_PROMPT = "fog, grainy, purple"
# The seed number to use in the pre-compile time. A Random number will be used
# in actual predictions in production.
DEFAULT_SEED = 33
DEFAULT_GUIDANCE_SCALE = 9.0
DEFAULT_NUM_STEPS = 25
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024

CACHE_FOLDER = "/tmp/cache"

KEY_MODEL_NAME_SDXL_BASE_1_0 = "stabilityai/stable-diffusion-3.5-medium"  # Changed line
KEY_MODEL_REVISION_SDXL_BASE_1_0 = "refs/pr/95"
KEY_SCHEDULER = "scheduler"
KEY_NUM_STEPS = "NUM_STEPS"
KEY_HEIGHT = "HEIGHT"
KEY_WIDTH = "WIDTH"

PyTree = Any