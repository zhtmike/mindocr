import os
from typing import Any, Dict, List, Union

import lmdb
import numpy as np
from tqdm import tqdm


def write_cache(env: lmdb.Environment, cache: Dict[str, Any]):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_dataset(images: List[Union[np.ndarray, bytes]], labels: List[str], output_path: str = "./lmdb_out"):
    """Create the LMDB dataset with the given img_paths and labels"""
    if len(images) != len(labels):
        raise ValueError("Lengths of the `img_paths` and `labesl` should be matched exactly.")

    os.makedirs(output_path, exist_ok=True)
    env = lmdb.Environment(output_path, map_size=1099511627776)
    cache = {}

    for i, (image, label) in tqdm(enumerate(zip(images, labels)), total=len(images), desc="Creating LMDB"):
        imageKey = "image-%09d".encode() % (i + 1)
        labelKey = "label-%09d".encode() % (i + 1)

        if isinstance(image, np.ndarray):
            image = image.tobytes()

        cache[imageKey] = image
        cache[labelKey] = label.encode()
        if i % 1000 == 0:
            write_cache(env, cache)
            cache = {}

    num_samples = len(images)
    cache["num-samples".encode()] = str(num_samples).encode()
    write_cache(env, cache)
    print(f"Created dataset with {num_samples} samples.")
