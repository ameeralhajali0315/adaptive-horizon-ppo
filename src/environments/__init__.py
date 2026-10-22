
from gymnasium.envs.registration import register, registry


ENVIRONMENTS = {
    "SparseGridWorld-v0": {},
    "SparseGridWorld-v1": {
        "grid_size": 7,
        "max_steps": 40,
    },
}

for env_id, kwargs in ENVIRONMENTS.items():
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="src.environments.sparse_grid_world:SparseGridWorld",
            kwargs=kwargs,
        )
