from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr

Array = jax.Array

@dataclass(frozen=True)
class BlockwiseUniform:
    low: Array
    high: Array

    def __init__(self, low: Array, high: Array):
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        if low.shape != high.shape:
            raise ValueError(
                f"low and high must have matching shapes, got {low.shape} and {high.shape}."
            )
        if jnp.any(high <= low):
            raise ValueError("All high values must be greater than low values.")

        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.low.shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    def sample(
        self,
        sample_shape: Optional[int | tuple[int, ...]] = (),
        *,
        seed: Array,
    ) -> Array:
        if sample_shape is None:
            sample_shape = ()
        elif isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        shape = tuple(sample_shape) + self.event_shape
        return jr.uniform(seed, shape=shape, minval=self.low, maxval=self.high)

    def log_prob(self, value: Array) -> Array:
        value = jnp.asarray(value)
        in_support = jnp.all((value >= self.low) & (value <= self.high), axis=-1)
        log_volume = jnp.sum(jnp.log(self.high - self.low))
        return jnp.where(in_support, -log_volume, -jnp.inf)

    def prob(self, value: Array) -> Array:
        return jnp.exp(self.log_prob(value))
