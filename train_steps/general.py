from functools import partial

@jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['images'])
        one_hot = jax.nn.one_hot(batch['labels'], n_targets)
        softmax = jax.nn.softmax(logits)

        # Cross Entropy Loss
        loss_ce = evaluate_ce(softmax, one_hot)
        # L2 Regularization
        wd = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        loss = loss_ce + 0.0005 * 0.5 * wd

        metrics = {
            'acc': evaluate_acc(softmax, batch['labels']),
            'loss': loss
        }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics



@jit
def test_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['images'])
    one_hot = jax.nn.one_hot(batch['labels'], n_targets)
    softmax = jax.nn.softmax(logits)

    return {
        'acc': evaluate_acc(softmax, batch['labels']),
        'loss': evaluate_ce(softmax, one_hot)
    }

@jit
def test_step_ensemble(states, batch):
    softmax_list = [jax.nn.softmax(st.apply_fn({'params': st.params}, batch['images']))
                    for st in states]
    mean_soft = jnp.stack(softmax_list).mean(axis=0)
    one_hot = jax.nn.one_hot(batch['labels'], n_targets)
    return {
        'acc': evaluate_acc(mean_soft, batch['labels']),
        'loss': evaluate_ce(mean_soft, one_hot),
        'bs':  evaluate_bs(mean_soft, one_hot)
    }
