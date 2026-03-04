from datetime import datetime
from typing import Tuple
import re
import os
import sys
import tiktoken
import random as rng
from importlib.resources import files
from PIL import Image, PngImagePlugin
from io import BytesIO
from PIL import Image, PngImagePlugin
import numpy as np
import yaml

def save_images_with_metadata(
    images,
    output_path,
    filename_prefix="image",
    metadata=None,
    extension="png",
    dpi=300,
    quality=90,
    optimize=True,
    number_padding=2,
):
    output_path = os.path.abspath(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    extension = extension.lower()
    if not extension.startswith("."):
        extension = "." + extension

    existing_numbers = []
    for f in os.listdir(output_path):
        if f.startswith(filename_prefix) and f.endswith(extension):
            num_part = f[len(filename_prefix) :].split(".")[0].strip("_-")
            if num_part.isdigit():
                existing_numbers.append(int(num_part))
    counter = max(existing_numbers + [0]) + 1

    saved_files = []

    for img in images:
        if not isinstance(img, Image.Image):
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

        filename = f"{filename_prefix}_{str(counter).zfill(number_padding)}{extension}"
        filepath = os.path.join(output_path, filename)

        if extension == ".png":
            metadata = PngImagePlugin.PngInfo()
            if metadata:
                metadata.add_text("metadata", yaml.dump(metadata))
            img.save(
                filepath,
                dpi=(dpi, dpi),
                quality=quality,
                optimize=optimize,
                pnginfo=metadata,
            )
        else:
            img.save(filepath, dpi=(dpi, dpi), quality=quality, optimize=optimize)

        saved_files.append(filepath)
        counter += 1
    print(f"Saved {len(images)} images to {output_path}")

    return saved_files

