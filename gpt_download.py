# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import time

import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if tf_ckpt_path is None:
        raise RuntimeError(
            f"No valid TensorFlow checkpoint found in '{model_dir}'. "
            "The download may be incomplete; delete the directory and try again."
        )

    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None, max_retries=3):
    def _attempt_download(download_url):
        # Resume a previous partial download when the server supports it
        existing_size = os.path.getsize(destination) if os.path.exists(destination) else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

        response = requests.get(download_url, stream=True, timeout=60, headers=headers)

        if existing_size and response.status_code == 416:
            # Nothing left to fetch: the local file already covers the remote one
            print(f"File already exists and is up-to-date: {destination}")
            return
        if existing_size and response.status_code != 206:
            # Server ignored the Range header, so start over from scratch
            existing_size = 0

        response.raise_for_status()

        content_length = int(response.headers.get("Content-Length", 0))
        file_size = existing_size + content_length if content_length else 0

        block_size = 1024 * 1024  # 1 MB
        desc = os.path.basename(download_url)
        mode = "ab" if existing_size else "wb"
        with tqdm(total=file_size, initial=existing_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, mode) as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

        if file_size and os.path.getsize(destination) != file_size:
            raise requests.exceptions.RequestException(
                f"Incomplete download for {destination}: "
                f"expected {file_size} bytes, got {os.path.getsize(destination)}"
            )

    urls = [url] + ([backup_url] if backup_url else [])
    for index, download_url in enumerate(urls):
        if index > 0:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")

        for attempt in range(1, max_retries + 1):
            try:
                _attempt_download(download_url)
                return
            except requests.exceptions.RequestException as e:
                print(
                    f"Download attempt {attempt}/{max_retries} failed for "
                    f"{download_url}: {e}"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

    if os.path.exists(destination):
        os.remove(destination)

    raise RuntimeError(
        f"Failed to download from both primary URL ({url})"
        f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
        "\nCheck your internet connection or the file availability.\n"
        "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
    )


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
