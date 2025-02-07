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
