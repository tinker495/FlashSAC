from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray


class MjlabVectorEnv(
    VectorEnv[Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray]]
):
    """
    Gymnasium vector-env wrapper for mjlab (MuJoCo Warp) manager-based tasks.

    Mirrors ``IsaacLabVectorEnv``: mjlab exposes the same manager-based API (``actor`` /
    ``critic`` observation groups, in-simulator auto reset, ``episode_length_buf``), so the
    training loop sees the same interface and the same reset semantics. Actions are clipped to
    [-1, 1]; like IsaacLab, the auto-reset observation stands in for the terminal observation.
    The ``actor`` group plays IsaacLab's ``policy`` group. When the task defines a ``critic``
    group, the observation is ``concat(actor, critic)`` and ``actor_observation_size`` tells the
    agent where the actor slice ends (``agent.asymmetric_observation=true``), as in IsaacLab.

    Args:
        env_name (str): The task name registered in mjlab.
        num_envs (int): The number of parallel environments.
        seed (int):
        device (str):
        to_numpy (bool): If True, outputs are converted from torch tensors to numpy arrays.
    """

    def __init__(self, env_name: str, num_envs: int, seed: int, device: str, to_numpy: bool = True):
        import mjlab.tasks  # noqa: F401 - registers the built-in tasks
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        cfg = load_env_cfg(env_name)
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        cfg.auto_reset = True
        self.seed = seed
        self.device = device
        self.envs = ManagerBasedRlEnv(cfg, device=device)

        self.num_envs = self.envs.num_envs
        self.max_episode_steps = self.envs.max_episode_length
        self.to_numpy = to_numpy

        observations, _ = self.envs.reset()
        self.obs_size = tuple(observations["actor"].shape[1:])
        self.asymmetric_obs = "critic" in observations
        self.critic_obs_size = tuple(observations["critic"].shape[1:]) if self.asymmetric_obs else (0,)
        # NOTE: setting to [0, 0] since we only need the shape and dtype
        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=(self.obs_size[0] + self.critic_obs_size[0],), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_size = tuple(self.envs.single_action_space.shape)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.action_size, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[Union[torch.Tensor, F32NDArray], dict[str, Any]]:
        obs_dict, _ = self.envs.reset()
        obs = self._observation(obs_dict)
        # NOTE: decorrelate episode horizons like RSL-RL / the IsaacLab wrapper.
        if random_start_init:
            cast(Any, self.envs).episode_length_buf = torch.randint_like(
                self.envs.episode_length_buf, high=int(self.max_episode_steps)
            )
        infos: dict[str, Any] = {"actor_observation_size": self.obs_size, "asymmetric_obs": self.asymmetric_obs}
        if self.to_numpy:
            return obs.cpu().numpy(), infos
        return obs, infos

    def step(self, actions: Union[torch.Tensor, F32NDArray]) -> tuple[
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        dict[str, Any],
    ]:
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(actions).to(self.device)
        torch_actions = torch.clamp(torch_actions, -1.0, 1.0)
        obs_dict, rew, terminations, truncations, _ = self.envs.step(torch_actions)
        obs = self._observation(obs_dict)
        critic_obs = obs_dict["critic"] if self.asymmetric_obs else None
        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            critic_obs = None if critic_obs is None else critic_obs.cpu().numpy()
        infos = {"time_outs": truncations, "observations": {"critic": critic_obs}, "final_obs": obs}
        return obs, rew, terminations, truncations, infos

    def _observation(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.asymmetric_obs:
            return torch.cat((obs_dict["actor"], obs_dict["critic"]), dim=-1)
        return obs_dict["actor"]

    def close(self, **kwargs: Any) -> None:
        self.envs.close()

    def render(self) -> None:
        raise NotImplementedError("Rendering is not supported for mjlab environments")


def make_mjlab_env(env_name: str, num_envs: int, seed: int) -> MjlabVectorEnv:
    return MjlabVectorEnv(env_name=env_name, num_envs=num_envs, seed=seed, device="cuda")
