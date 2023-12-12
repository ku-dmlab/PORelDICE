from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params

def update_v(
    critic: Model, value: Model, batch: Batch, alpha: float, epsilon:float, discount: float, alg: str
) -> Tuple[Model, InfoDict]:
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({"params": value_params}, batch.observations)
        v_0 = value.apply({"params": value_params}, batch.initial_observations)
        q1, q2 = critic(batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
    
        if alg == "PORelDICE":
            sp_term = (q-v) / alpha
            value_loss = ((1-discount) * v_0).mean() + (alpha * 
                          jnp.where(1 + sp_term > epsilon, 
                          (0.5 * sp_term **2 + sp_term),
                          (epsilon) * (sp_term - epsilon + 1) + 0.5 * (epsilon - 1) ** 2 + epsilon - 1
                          )).mean()
        else:
            raise NotImplementedError("please choose PORelDICE")
        return value_loss, {
            "value_loss": value_loss,
            "v": v.mean(),
            "q-v": (q - v).mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info

def update_q(
    critic: Model, value: Model, batch: Batch, discount: float, alg: str
) -> Tuple[Model, InfoDict]:
    next_v = value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply(
            {"params": critic_params}, batch.observations, batch.actions
        )

        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info