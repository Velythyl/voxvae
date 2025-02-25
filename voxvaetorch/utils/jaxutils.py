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


def map_ternary(T, mapping):
    """
    Remaps the values 1, 2, 3 in tensor T to a, b, c respectively.
    T is assumed to be an array containing values in {0, 1, 2, 3}.
    'mapping' is a tuple (a,b,c).
    0 is treated as the identity element

    THANKS, ChatGPT!
    """
    a, b, c = mapping[0], mapping[1], mapping[2]

    # Compute the coefficients for f(x) = A*x**3 + B*x**2 + C*x
    A = (c + 3 * a - 3 * b) / 6
    B = (4 * b - 5 * a - c) / 2
    C = (18 * a - 9 * b + 2 * c) / 6

    # Apply the polynomial to every element of T.
    return A * T ** 3 + B * T ** 2 + C * T
