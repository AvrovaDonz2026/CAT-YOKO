"""Opt-in synchronized update timing for isolated resident B0 experiments.

Install this context after any candidate optimizer patch. It wraps the then
current CPUOffloadAdamW.step, so GPU candidates finish their queued work before
a sample ends. No patch or synchronization happens merely by importing it.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from cat_yoko.optim import CPUOffloadAdamW
from cat_yoko.trainer import Trainer


class UpdateTiming:
    def __init__(self, output_path: Path, warmup: int):
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        self.output_path = Path(output_path)
        self.warmup = warmup
        self.samples: list[dict] = []
        self.owner = None
        self.optimizer = None
        self.parameter_ids = None
        self.started_at = None
        self.update_tokens = 0
        self.started_updates = 0
        self.run_started = False
        self.run_finished = False
        self.error = None

    def _sync(self):
        if self.owner is not None and str(self.owner.device).startswith("cuda"):
            torch.cuda.synchronize(self.owner.device)

    def _begin_forward(self, owner, model, batch):
        if owner is not self.owner:
            return
        if self.started_at is None:
            self.parameter_ids = frozenset(id(p) for p in model.parameters() if p.requires_grad)
            self._sync()
            self.started_at = time.perf_counter()
            self.started_updates += 1
            self.update_tokens = 0
        ids = batch.get("input_ids") if isinstance(batch, dict) else None
        self.update_tokens += int(ids.numel()) if torch.is_tensor(ids) else 4096

    def _owns_optimizer(self, optimizer):
        if self.owner is None or self.started_at is None:
            return False
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if not parameters or frozenset(id(p) for p in parameters) != self.parameter_ids:
            return False
        if self.optimizer is None:
            self.optimizer = optimizer
        return optimizer is self.optimizer

    def _end_update(self):
        self._sync()
        elapsed = time.perf_counter() - self.started_at
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("synchronized update elapsed time must be finite and positive")
        self.samples.append({
            "invocation_update": len(self.samples) + 1,
            "tokens": self.update_tokens,
            "elapsed_s": elapsed,
            "completed_update_tokens_s": self.update_tokens / elapsed,
        })
        self.started_at = None
        self.update_tokens = 0

    def report(self):
        steady = self.samples[self.warmup:]
        elapsed = sum(sample["elapsed_s"] for sample in steady)
        tokens = sum(sample["tokens"] for sample in steady)
        complete = self.run_finished and self.started_at is None and bool(self.samples)
        return {
            "status": "failed" if self.error else "complete" if complete else "incomplete",
            "error": self.error,
            "warmup_updates": self.warmup,
            "started_updates": self.started_updates,
            "completed_updates": len(self.samples),
            "incomplete_update": self.started_at is not None,
            "measured_updates": len(steady),
            "discarded_updates": min(self.warmup, len(self.samples)),
            "samples": list(self.samples),
            "median_completed_update_tokens_s": (
                statistics.median(sample["completed_update_tokens_s"] for sample in steady)
                if steady else None
            ),
            "sum_steady_elapsed_s": elapsed,
            "steady_tokens": tokens,
            "aggregate_completed_update_tokens_s": tokens / elapsed if elapsed else None,
            "scope": "forward start through completed optimizer update, including GPU synchronization",
            "notes": [
                "GPU is synchronized before the first microbatch and after each complete update.",
                "Initialization, parity, initial/final evaluation, checkpoint saves and post-update logs are excluded.",
                "LR/zero_grad and the already-prefetched current batch are excluded; next-batch prefetch is included.",
                "Both variants must use this measurement; synchronization can change pipeline throughput.",
            ],
        }

    def write_report(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(self.output_path.name + ".tmp")
        temporary.write_text(json.dumps(self.report(), indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output_path)


@contextmanager
def synchronized_update_timing(output_path: Path, warmup: int = 5):
    """Measure one Trainer.run and restore every method on success or failure.

    The owner is bound only during Trainer.run. Initial held-out evaluation and
    unrelated trainers/optimizers cannot create samples. Gradient accumulation
    contributes every microbatch to one completed-update sample.
    """
    timing = UpdateTiming(output_path, warmup)
    original_run = Trainer.run
    original_forward = Trainer._forward_loss
    original_step = CPUOffloadAdamW.step

    def run(owner, *args, **kwargs):
        if timing.run_started:
            raise RuntimeError("synchronized update timing supports one Trainer.run per context")
        timing.run_started = True
        timing.owner = owner
        try:
            result = original_run(owner, *args, **kwargs)
            timing.run_finished = True
            return result
        except BaseException as error:
            timing.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            timing.owner = None

    def forward(owner, model, batch):
        timing._begin_forward(owner, model, batch)
        return original_forward(owner, model, batch)

    def step(optimizer, *args, **kwargs):
        owned = timing._owns_optimizer(optimizer)
        result = original_step(optimizer, *args, **kwargs)
        if owned:
            timing._end_update()
        return result

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(Trainer, "run", run))
            stack.enter_context(patch.object(Trainer, "_forward_loss", forward))
            stack.enter_context(patch.object(CPUOffloadAdamW, "step", step))
            yield timing
    except BaseException as error:
        if timing.error is None:
            timing.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        timing.write_report()
