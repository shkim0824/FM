import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial

from data.transform import *


def _buildLoader(images, labels, batch_size, steps_per_epoch, rng=None, shuffle=False, transform=None):
    # Shuffle Indices
    indices = jax.random.permutation(rng, len(images)) if shuffle else jnp.arange(len(images)) # Make shuffled indices
    indices = indices[:steps_per_epoch*batch_size] # Batch size may not be divisor of length of images. We drop left ones.
    indices = indices.reshape((steps_per_epoch, batch_size,))
    for batch_idx in indices:
        batch = {'images': jnp.array(images[batch_idx]), 'labels': jnp.array(labels[batch_idx])}
        if transform is not None:
            if rng is not None:
                _, rng = jax.random.split(rng)
            sub_rng = None if rng is None else jax.random.split(rng, batch['images'].shape[0])
            batch['images'] = transform(sub_rng, batch['images'])
        yield batch
