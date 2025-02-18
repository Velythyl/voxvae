import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnums=(1,))
def split_key(key, num_keys):
    key, *rng = jax.random.split(key, num_keys + 1)
    rng = jnp.reshape(jnp.stack(rng), (num_keys, 2))
    return key, rng

def _bool_ifelse_elementwise(cond, iftrue, iffalse):
    return iftrue * cond + iffalse * (1-cond)

@jax.jit
def bool_ifelse(cond, iftrue, iffalse):
    MAIN_SHAPE = cond.shape[0]

    if len(iffalse.shape) == 0:
        iffalse = jnp.ones_like(cond) * iffalse

    if len(iftrue.shape) == 0:
        iftrue = jnp.ones_like(cond) * iftrue

    if iffalse.shape[0] != MAIN_SHAPE:
        iffalse = jnp.repeat(iffalse[None], MAIN_SHAPE, axis=0)
    if iftrue.shape[0] != MAIN_SHAPE:
        iftrue = jnp.repeat(iftrue[None], MAIN_SHAPE, axis=0)

    cond = cond.astype(int)

    return jax.vmap(_bool_ifelse_elementwise)(cond, iftrue, iffalse)