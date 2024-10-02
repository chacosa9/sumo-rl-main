"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register
from .env import SumoEnvironment


register(
    id="sumo-rl-v0",
    entry_point="sumo_rl.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
