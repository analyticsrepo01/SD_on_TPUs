# Vertex AI Model Garden - Stable Diffusion XL

This repository contains notebooks demonstrating how to deploy and use Stable Diffusion XL models on Google Cloud Vertex AI.

## Notebooks

*   **[jax_stable_diffusion_xl_refined.ipynb](jax_stable_diffusion_xl_refined.ipynb)**: Demonstrates how to deploy the `stabilityai/stable-diffusion-xl-base-1.0` model on Vertex AI for online prediction using a TPU v5e instance. It covers deploying the model to a Vertex AI Endpoint resource and running online predictions for text-to-image tasks.

## Overview

The notebooks provide examples of:

*   Deploying Stable Diffusion XL models to Vertex AI Endpoints.
*   Running online predictions for text-to-image generation.
*   Utilizing TPU v5e instances for accelerated inference.

## Costs

Using these notebooks may incur costs for the following Google Cloud services:

*   Vertex AI
*   Cloud Storage

Refer to the [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and [Cloud Storage pricing](https://cloud.google.com/storage/pricing) documentation for more details. Use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to estimate costs based on your projected usage.

## Before you begin

Ensure you have the following:

*   A Google Cloud project with billing enabled.
*   The Vertex AI API and Compute Engine API enabled.
*   A Cloud Storage bucket for storing experiment outputs.
*   A service account with `Vertex AI User` and `Storage Object Admin` roles.

## Setup

1.  Select or create a Google Cloud project in the [Google Cloud Console](https://console.cloud.google.com/cloud-resource-manager).
2.  Enable billing for your project.
3.  Enable the Vertex AI API and Compute Engine API.
4.  Create a Cloud Storage bucket for storing experiment outputs.
5.  Create a service account with the necessary roles.

## Getting Started

Refer to the individual notebook files for detailed instructions on how to deploy and use the Stable Diffusion XL models.

## Update the container for own model:

All the steps needed to update a container :

1.  Pull (Do update the container):
    ```
    docker pull us-west1-docker.pkg.dev/my-project-0004-346516/diffusion-jax-model/my-custom-diffusers:v1.0
    ```
2.  Run (Update the container name):
    ```
    docker run -it --name temp-config-container us-west1-docker.pkg.dev/my-project-0004-346516/diffusion-jax-model/my-custom-diffusers:v1.0 /bin/bash
    ```
3.  (Inside container): `nano download_model.py` (make your changes and save)

    1.  Use the following python code and save it as `download_model.py`

    ```python
    from huggingface_hub import snapshot_download
    import os
    # --- CHOOSE YOUR MODEL HERE ---
    # Replace this with the *actual* repo_id you want.
    repo_id = "stabilityai/stable-diffusion-2-1"  # Example: SD 2.1
    # repo_id = "your-org/your-model" #  <-- Use the correct one!

    # --- Specify the cache directory (optional, but good practice) ---
    cache_dir = "/root/.cache/huggingface/hub"

    # --- Download the model snapshot ---
    try:
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
        print(f"Successfully downloaded {repo_id} to {cache_dir}")

    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
    ```
4.  (Inside container): `pip install --upgrade huggingface_hub`
5.  (Inside container): `exit`
6.  Commit:
    ```
    docker commit temp-config-container us-west1-docker.pkg.dev/my-project-0004-346516/diffusion-jax-model/my-custom-diffusers:v1.1
    ```
7.  Push:
    ```
    docker push us-west1-docker.pkg.dev/my-project-0004-346516/diffusion-jax-model/my-custom-diffusers:v1.1
    ```
8.  Remove temporary container:
    ```
    docker rm temp-config-container
    ```
