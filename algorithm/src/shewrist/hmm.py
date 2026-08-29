"""Small categorical HMM used only to smooth shadow-model probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_EPS = 1e-12


@dataclass
class TemporalHMM:
    """Categorical transition model with Viterbi decoding.

    The CNN probabilities are treated as emissions.  This class has no path to
    the deterministic exposure or pressure safety state machines.
    """

    start_probability: np.ndarray
    transition_probability: np.ndarray

    def __post_init__(self) -> None:
        start = np.asarray(self.start_probability, dtype=float)
        transition = np.asarray(self.transition_probability, dtype=float)
        if start.ndim != 1 or transition.shape != (len(start), len(start)):
            raise ValueError("invalid HMM probability shapes")
        if np.any(start < 0.0) or np.any(transition < 0.0):
            raise ValueError("HMM probabilities must be non-negative")
        if not np.isclose(np.sum(start), 1.0):
            raise ValueError("start probabilities must sum to one")
        if not np.allclose(np.sum(transition, axis=1), 1.0):
            raise ValueError("each transition row must sum to one")
        self.start_probability = start
        self.transition_probability = transition

    @classmethod
    def fit(
        cls,
        sequences: Iterable[np.ndarray],
        n_classes: int,
        laplace: float = 0.5,
        self_transition_prior: float = 20.0,
        background_transition_prior: float = 2.0,
        background_index: int = 0,
    ) -> "TemporalHMM":
        if n_classes < 2:
            raise ValueError("at least two classes are required")
        start = np.full(n_classes, float(laplace), dtype=float)
        transition = np.full((n_classes, n_classes), float(laplace), dtype=float)
        transition[np.diag_indices(n_classes)] += float(self_transition_prior)
        transition[background_index, :] += float(background_transition_prior)
        transition[:, background_index] += float(background_transition_prior)
        observed = 0
        for values in sequences:
            labels = np.asarray(values, dtype=int)
            labels = labels[(labels >= 0) & (labels < n_classes)]
            if len(labels) == 0:
                continue
            start[labels[0]] += 1.0
            np.add.at(transition, (labels[:-1], labels[1:]), 1.0)
            observed += len(labels)
        if observed == 0:
            raise ValueError("no valid labels supplied to HMM")
        foreground = [index for index in range(n_classes) if index != background_index]
        if foreground:
            foreground_start = float(np.mean(start[foreground]))
            start[foreground] = foreground_start
            background_to_action = float(np.mean(transition[background_index, foreground]))
            transition[background_index, foreground] = background_to_action
            action_to_background = float(np.mean(transition[foreground, background_index]))
            action_self = float(np.mean([transition[index, index] for index in foreground]))
            action_other_values = [
                transition[source, target]
                for source in foreground
                for target in foreground
                if source != target
            ]
            action_to_other = float(np.mean(action_other_values)) if action_other_values else float(laplace)
            for source in foreground:
                transition[source, background_index] = action_to_background
                transition[source, foreground] = action_to_other
                transition[source, source] = action_self
        start /= np.sum(start)
        transition /= np.sum(transition, axis=1, keepdims=True)
        return cls(start, transition)

    def decode(self, emission_probability: np.ndarray) -> np.ndarray:
        emissions = np.asarray(emission_probability, dtype=float)
        if emissions.ndim != 2 or emissions.shape[1] != len(self.start_probability):
            raise ValueError("emissions must have shape (time, n_classes)")
        if len(emissions) == 0:
            return np.empty(0, dtype=int)
        emissions = np.clip(emissions, _EPS, 1.0)
        emissions /= np.sum(emissions, axis=1, keepdims=True)
        log_start = np.log(np.clip(self.start_probability, _EPS, 1.0))
        log_transition = np.log(np.clip(self.transition_probability, _EPS, 1.0))
        score = np.empty_like(emissions)
        backpointer = np.zeros_like(emissions, dtype=int)
        score[0] = log_start + np.log(emissions[0])
        for index in range(1, len(emissions)):
            candidates = score[index - 1, :, None] + log_transition
            backpointer[index] = np.argmax(candidates, axis=0)
            score[index] = np.max(candidates, axis=0) + np.log(emissions[index])
        states = np.empty(len(emissions), dtype=int)
        states[-1] = int(np.argmax(score[-1]))
        for index in range(len(emissions) - 2, -1, -1):
            states[index] = backpointer[index + 1, states[index + 1]]
        return states

    def to_dict(self) -> dict[str, object]:
        return {
            "start_probability": self.start_probability.tolist(),
            "transition_probability": self.transition_probability.tolist(),
        }
