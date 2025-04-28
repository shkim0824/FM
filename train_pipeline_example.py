def create_res32(rng, lr_fn):
    model = ResNet32()
    variables = model.init(rng, jnp.ones((4, 32, 32, 3)), training=False)
    print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(variables)))

    # ---- SWAG Hyperparameter ----
    freq        = 351                         # snapshot frequency (step) ; once per epoch ; 45000/128 = 351
    rank        = 20                         # low-rank approximation K
    burnin_ep   = 160                         # burn-in epochs
    start_step  = burnin_ep * trn_steps_per_epoch

    tx = optax.chain(
        optax.sgd(learning_rate=lr_fn,      # 1) SGD
                    momentum=0.9, nesterov=True),
        swag(freq=freq, rank=rank,          # 2) SWAG (Calculate \Sigma_SWAG)
                start_step=start_step)
    )

    class TrainState(train_state.TrainState): # We don't use dropout in ResNet
        pass

    state = TrainState.create(
        apply_fn   = model.apply,
        params     = variables['params'],
        tx         = tx
    )
    return state
    
state = create_res32(rng, scheduler)

metrics_history = defaultdict(list)
for epoch in range(epochs):
    start_time = time.time()

    rng, *keys = random.split(rng, 4)
    train_loader = trn_loader_aug(rng=keys[0])
    val_loader = vl_loader(rng=keys[1])
    test_loader = tst_loader(rng=keys[2])

    for batch in train_loader:
        state, metrics = train_step(state, batch)
    metrics_history['train_acc'].append(metrics['acc'])
    metrics_history['train_loss'].append(metrics['loss'])

    for batch in test_loader:
        metrics = test_step(state, batch) # In test, we do not need key
        for metric, value in metrics.items(): # compute metrics
            metrics_history[f'test_{metric}'].append(value) # record metrics

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} in {epoch_time:.2f} sec")
    print(f"Train acc: {100*metrics_history['train_acc'][-1]:.2f}% Test acc: {100*metrics_history['test_acc'][-1]:.2f}%")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.set_title('Accuracy')
ax2.set_title('NLL')

ax1.plot(metrics_history['test_acc'], label='ResNet-32')
ax2.plot(metrics_history['test_loss'], label='ResNet-32')

ax1.legend()
ax2.legend()
plt.show()
plt.clf()
