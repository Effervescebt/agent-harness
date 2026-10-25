"""Cheaper-model routing.

Two strategies, both measured against a single-model baseline:

- Cascade (task level): run the task on the cheap model; if it does not produce a valid answer
  (max_steps, repairs_exhausted, llm_error) or the answer fails a cheap self-check, rerun the whole
  task on the strong model. Cost = cheap attempt (+ strong attempt when escalated).
- Step escalation (step level): start on the cheap model and switch to the strong model for the rest of
  the trajectory once the cheap model has needed `escalate_after_repairs` repairs or used
  `escalate_after_steps` steps.
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMClient


@dataclass
class StepEscalation:
    cheap: LLMClient
    strong: LLMClient
    escalate_after_repairs: int = 1
    escalate_after_steps: int = 4

    def __call__(self, step: int, traj) -> LLMClient:
        repairs = sum(1 for s in traj.steps if s.repaired)
        if repairs >= self.escalate_after_repairs or step >= self.escalate_after_steps:
            return self.strong
        return self.cheap


def needs_escalation(traj, answer_check=None) -> bool:
    """Cascade rule: escalate when the cheap run did not answer, or the answer fails an optional check
    (answer_check(answer) -> bool; e.g. 'is a number' for math)."""
    if traj.status != "answered" or not traj.answer:
        return True
    return bool(answer_check and not answer_check(traj.answer))
