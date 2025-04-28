def buildLoader(images, labels, batch_size, steps_per_epoch, rng=None, shuffle=False, transform=None):
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

# Hyper parameters
BATCH_SIZE = 256
TEST_SIZE = 10000
n_targets = 10
rng = random.PRNGKey(0)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

# Make validation set
train_images, val_images = train_test_split(train_set.data, test_size=000, random_state=42)
train_labels, val_labels = train_test_split(train_set.targets, test_size=000, random_state=42) # It very boost training speed

# We use img converted to jnp dtype
train_images = np.array(train_images, dtype=jnp.float32)
val_images = np.array(val_images, dtype=jnp.float32)
test_images = np.array(test_set.data, dtype=jnp.float32)

# Make labels to numpy array
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_set.targets)

# transform for CIFAR-10
transform = TransformChain([RandomHFlipTransform(0.5),
                            RandomCropTransform(size=32, padding=4),
                            ToTensorTransform()])

# Naive data loader. We should put rng later in real-time usage
trn_steps_per_epoch = len(train_images) // BATCH_SIZE
tst_steps_per_epoch = len(test_images) // TEST_SIZE
val_steps_per_epoch = len(val_images) // TEST_SIZE

trn_loader = partial(buildLoader,
                    images=train_images,
                    labels=train_labels,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=trn_steps_per_epoch,
                    shuffle=True,
                    transform=jit(vmap(ToTensorTransform())))

trn_loader_aug = partial(buildLoader,
                        images=train_images,
                        labels=train_labels,
                        batch_size=BATCH_SIZE,
                        steps_per_epoch=trn_steps_per_epoch,
                         shuffle=True,
                         transform=jit(vmap(transform)))

vl_loader = partial(buildLoader,
                    images=val_images,
                    labels=val_labels,
                    batch_size=len(val_images),
                    steps_per_epoch=1,
                    shuffle=False,
                    transform=jit(vmap(ToTensorTransform())))

tst_loader = partial(buildLoader,
                    images=test_images,
                    labels=test_labels,
                    batch_size=TEST_SIZE,
                    steps_per_epoch=tst_steps_per_epoch,
                    shuffle=False,
                    transform=jit(vmap(ToTensorTransform())))
