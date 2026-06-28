from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


EPSILON = 1e-6


@dataclass(frozen=True)
class TransitionData:
    obs: np.ndarray
    actions: np.ndarray
    next_obs: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode: np.ndarray
    step: np.ndarray

    @property
    def inputs(self) -> np.ndarray:
        return np.concatenate([self.obs, self.actions], axis=1).astype(np.float32)

    @property
    def targets(self) -> np.ndarray:
        delta_obs = self.next_obs - self.obs
        reward_column = self.rewards.reshape(-1, 1)
        return np.concatenate([delta_obs, reward_column], axis=1).astype(np.float32)


class MLPDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_dim, hidden_size))
            layers.append(nn.ReLU())
            previous_dim = hidden_size

        layers.append(nn.Linear(previous_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_transition_data(path: Path) -> TransitionData:
    with np.load(path, allow_pickle=False) as data:
        return TransitionData(
            obs=data["obs"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            next_obs=data["next_obs"].astype(np.float32),
            rewards=data["rewards"].astype(np.float32),
            terminated=data["terminated"].astype(np.bool_),
            truncated=data["truncated"].astype(np.bool_),
            episode=data["episode"].astype(np.int32),
            step=data["step"].astype(np.int32),
        )


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    std = values.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, EPSILON)
    return mean, std


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def unstandardize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return values * std + mean


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_ensemble_checkpoint(
    path: Path,
    *,
    models: list[MLPDynamicsModel],
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dicts": [cpu_state_dict(model) for model in models],
            "input_mean": torch.as_tensor(input_mean, dtype=torch.float32),
            "input_std": torch.as_tensor(input_std, dtype=torch.float32),
            "target_mean": torch.as_tensor(target_mean, dtype=torch.float32),
            "target_std": torch.as_tensor(target_std, dtype=torch.float32),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_sizes": hidden_sizes,
            "metadata": metadata,
        },
        path,
    )


class WorldModelEnsemble:
    def __init__(
        self,
        models: list[MLPDynamicsModel],
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.models = models
        self.input_mean = input_mean.to(device)
        self.input_std = input_std.to(device)
        self.target_mean = target_mean.to(device)
        self.target_std = target_std.to(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        for model in self.models:
            model.to(device)
            model.eval()

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> WorldModelEnsemble:
        torch_device = choose_device(device)
        checkpoint = torch.load(path, map_location=torch_device)
        hidden_sizes = tuple(checkpoint["hidden_sizes"])
        obs_dim = int(checkpoint["obs_dim"])
        action_dim = int(checkpoint["action_dim"])
        input_dim = obs_dim + action_dim
        output_dim = obs_dim + 1

        models = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = MLPDynamicsModel(input_dim, output_dim, hidden_sizes)
            model.load_state_dict(state_dict)
            models.append(model)

        return cls(
            models=models,
            input_mean=checkpoint["input_mean"],
            input_std=checkpoint["input_std"],
            target_mean=checkpoint["target_mean"],
            target_std=checkpoint["target_std"],
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=torch_device,
        )

    @torch.no_grad()
    def predict_members(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs_array = np.atleast_2d(obs).astype(np.float32)
        action_array = np.atleast_2d(actions).astype(np.float32)
        inputs = np.concatenate([obs_array, action_array], axis=1)
        inputs_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        normalized_inputs = (inputs_tensor - self.input_mean) / self.input_std

        member_outputs = []
        for model in self.models:
            normalized_target = model(normalized_inputs)
            target = unstandardize(
                normalized_target,
                self.target_mean,
                self.target_std,
            )
            member_outputs.append(target)

        targets = torch.stack(member_outputs, dim=0)
        delta_obs = targets[..., : self.obs_dim]
        rewards = targets[..., self.obs_dim]
        next_obs = obs_array[None, :, :] + delta_obs.cpu().numpy()

        return next_obs.astype(np.float32), rewards.cpu().numpy().astype(np.float32)

    def predict(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_obs_members, reward_members = self.predict_members(obs, actions)
        next_obs_mean = next_obs_members.mean(axis=0)
        reward_mean = reward_members.mean(axis=0)
        disagreement = next_obs_members.std(axis=0).mean(axis=1)
        return next_obs_mean, reward_mean, disagreement.astype(np.float32)

    @torch.no_grad()
    def rollout_members(
        self,
        initial_obs: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        action_array = np.asarray(actions, dtype=np.float32)
        initial_obs_array = np.asarray(initial_obs, dtype=np.float32).reshape(1, -1)
        member_count = len(self.models)

        member_obs = torch.as_tensor(
            np.repeat(initial_obs_array, member_count, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        member_states = [member_obs.cpu().numpy()]
        member_rewards = []

        for action in action_array:
            action_tensor = torch.as_tensor(
                np.repeat(action.reshape(1, -1), member_count, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            outputs = []

            for model_index, model in enumerate(self.models):
                model_input = torch.cat(
                    [
                        member_obs[model_index : model_index + 1],
                        action_tensor[model_index : model_index + 1],
                    ],
                    dim=1,
                )
                normalized_input = (model_input - self.input_mean) / self.input_std
                normalized_target = model(normalized_input)
                outputs.append(
                    unstandardize(
                        normalized_target,
                        self.target_mean,
                        self.target_std,
                    )
                )

            targets = torch.cat(outputs, dim=0)
            delta_obs = targets[:, : self.obs_dim]
            rewards = targets[:, self.obs_dim]
            member_obs = member_obs + delta_obs

            member_states.append(member_obs.cpu().numpy())
            member_rewards.append(rewards.cpu().numpy())

        return (
            np.stack(member_states, axis=0).astype(np.float32),
            np.stack(member_rewards, axis=0).astype(np.float32),
        )
