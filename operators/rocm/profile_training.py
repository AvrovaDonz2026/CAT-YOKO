#!/usr/bin/env python3
"""Profile real B0 updates in an isolated run, after parity and two warmups.

This wraps model_bench (or candidate_bench) rather than changing the trainer.
The source overlay must include CPU FP32 Adam state. Setup, parity, and the
initial/final held-out evaluations stay outside the profiler. Save/eval events
inside Trainer.run receive separate labels and never count as steady compute.
All experimental updates, traces, and checkpoints go into --out.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import functools
import json
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cat_yoko.checkpoint import is_trainable_ckpt, load_checkpoint, resolve_resume_path
from cat_yoko.data import PackedBinStream
from cat_yoko.optim import CPUOffloadAdamW
from cat_yoko.trainer import Trainer
from operators.rocm import model_bench

PREFIX = "cat_profile::"
PHASES = ("forward", "ce", "backward", "clip", "optimizer", "data", "bookkeeping", "save", "eval")
STEADY_PHASES = PHASES[:-2]


def _number(event, name):
    value = getattr(event, name, 0.0)
    return float(value or 0.0)


def _event_time_range(event):
    """Return one CPU profiler clock interval, without using kernel times."""
    interval = getattr(event, "time_range", None)
    try:
        start, end = float(interval.start), float(interval.end)
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        return None
    return start, end


def _scope_time_ranges(events):
    phases, steps = [], []
    for event in events:
        if "CPU" not in str(getattr(event, "device_type", "CPU")):
            continue
        name = getattr(event, "name", "")
        interval = _event_time_range(event)
        if interval is None:
            continue
        if name == PREFIX + "train_step":
            steps.append(interval)
        elif name.startswith(PREFIX) and name[len(PREFIX):] in PHASES:
            phases.append((*interval, name[len(PREFIX):]))
    return phases, steps


def _classify_event(event, scope_ranges=None):
    """Prefer ancestry; fall back to enclosing CPU-clock record ranges."""
    current = event
    nearest = None
    inside_step = False
    while current is not None:
        name = getattr(current, "name", "")
        if name == PREFIX + "train_step":
            inside_step = True
        if name.startswith(PREFIX) and name[len(PREFIX):] in PHASES:
            phase = name[len(PREFIX):]
            if phase in ("save", "eval"):
                return phase, "parent_chain"
            if nearest is None:
                nearest = phase
        current = getattr(current, "cpu_parent", None)
    if nearest is not None or inside_step:
        return nearest or "bookkeeping", "parent_chain"
    interval = _event_time_range(event)
    if scope_ranges is None or interval is None:
        return "other", "unclassified"
    start, end = interval
    phase_ranges, step_ranges = scope_ranges
    containing = [(left, right, phase) for left, right, phase in phase_ranges
                  if left <= start and end <= right]
    # Worker-thread CE/recomputation scopes may sit inside eval/save. Those
    # exclusions take precedence even when the inner phase is more specific.
    excluded = [scope for scope in containing if scope[2] in ("save", "eval")]
    if containing:
        nearest = min(excluded or containing, key=lambda scope: (scope[1] - scope[0], -scope[0]))
        return nearest[2], "cpu_time_range"
    if any(left <= start and end <= right for left, right in step_ranges):
        return "bookkeeping", "cpu_time_range"
    return "other", "unclassified"


def event_phase(event, scope_ranges=None):
    """Classify by parent chain, then the narrowest enclosing CPU scope."""
    return _classify_event(event, scope_ranges)[0]


def summarize_events(events):
    """Sum self times once; device work is not GPU wall time under overlap."""
    phases = {name: {"self_cpu_us": 0.0, "device_work_us": 0.0, "events": 0}
              for name in (*PHASES, "other")}
    events = list(events)
    scope_ranges = _scope_time_ranges(events)
    operators, steady_operators = {}, {}
    attribution = {"parent_chain": 0, "cpu_time_range": 0, "unclassified": 0}
    transfers = {"self_cpu_us": 0.0, "device_work_us": 0.0, "events": 0}
    for event in events:
        # CPU events carry linked kernel times. Counting separate device
        # events as well would double-count that same GPU work.
        device_type = str(getattr(event, "device_type", "CPU"))
        if "CPU" not in device_type:
            continue
        phase, method = _classify_event(event, scope_ranges)
        attribution[method] += 1
        cpu = _number(event, "self_cpu_time_total")
        gpu = _number(event, "self_device_time_total")
        phases[phase]["self_cpu_us"] += cpu
        phases[phase]["device_work_us"] += gpu
        phases[phase]["events"] += 1
        name = getattr(event, "name", "unknown")
        entry = operators.setdefault(name, {"name": name, "self_cpu_us": 0.0,
                                           "device_work_us": 0.0, "events": 0})
        entry["self_cpu_us"] += cpu
        entry["device_work_us"] += gpu
        entry["events"] += 1
        if phase in STEADY_PHASES:
            steady = steady_operators.setdefault(name, {"name": name, "self_cpu_us": 0.0,
                                                       "device_work_us": 0.0, "events": 0})
            steady["self_cpu_us"] += cpu
            steady["device_work_us"] += gpu
            steady["events"] += 1
        if phase in STEADY_PHASES and any(word in name.lower() for word in
                                        ("memcpy", "copy_", "aten::to", "aten::_to_copy")):
            transfers["self_cpu_us"] += cpu
            transfers["device_work_us"] += gpu
            transfers["events"] += 1
    total_cpu = sum(phases[name]["self_cpu_us"] for name in STEADY_PHASES)
    total_gpu = sum(phases[name]["device_work_us"] for name in STEADY_PHASES)
    for name in PHASES:
        row = phases[name]
        row["steady_cpu_fraction"] = row["self_cpu_us"] / total_cpu if name in STEADY_PHASES and total_cpu else None
        row["steady_device_work_fraction"] = row["device_work_us"] / total_gpu if name in STEADY_PHASES and total_gpu else None
    return {
        "phases": phases, "steady_self_cpu_us": total_cpu,
        "steady_device_work_us": total_gpu, "transfer_overlay": transfers,
        "phase_attribution_events": attribution,
        "top_operators_by_device_work": sorted(operators.values(), key=lambda row: row["device_work_us"], reverse=True)[:40],
        "top_operators_by_self_cpu": sorted(operators.values(), key=lambda row: row["self_cpu_us"], reverse=True)[:40],
        "top_steady_operators_by_device_work": sorted(steady_operators.values(), key=lambda row: row["device_work_us"], reverse=True)[:40],
        "top_steady_operators_by_self_cpu": sorted(steady_operators.values(), key=lambda row: row["self_cpu_us"], reverse=True)[:40],
        "accounting": [
            "Self CPU times are assigned once by parent chain; CE is excluded from forward.",
            "Events without a labelled ancestor use enclosing CPU time_range scopes, choosing save/eval before the narrowest phase, then train_step bookkeeping.",
            "Device work sums linked kernel durations; it is not GPU wall time and may overlap CPU or other GPU work.",
            "Global top operator tables include save/eval/other; top_steady tables include only steady phase events.",
            "Transfer overlay includes aten::to/copy and memcpy inside steady phases; it overlaps phase totals and may include dtype conversions.",
            "Save/eval and events outside train_step are excluded from steady fractions; unlabelled work inside train_step is bookkeeping.",
            "Step spans start at _forward_loss and end after Adam.step; they include next-batch prefetch after backward, but exclude LR/zero_grad and the first active batch's earlier prefetch.",
            "Data includes packed-row assembly and H2D; CE includes vocabulary-head GEMMs, not just softmax.",
        ],
    }


class TrainingProfile:
    """Own one profiler window and restore every runtime wrapper on exit."""

    def __init__(self, output: Path, *, warmup: int = 2, active: int = 4,
                 device: str = "cuda", profile_memory: bool = False,
                 with_stack: bool = False, profiler_factory=None, metadata=None):
        self.output = Path(output)
        self.warmup, self.active = warmup, active
        self.device = device
        self.profile_memory, self.with_stack = profile_memory, with_stack
        self.profiler_factory = profiler_factory or torch.profiler.profile
        self.metadata = metadata or {}
        self.updates = 0
        self.recorded_steps = []
        self.profiler = None
        self.running = False
        self.step_scope = None
        self.step_started = None
        self.owner = None
        self.ce_depth = 0

    @contextmanager
    def region(self, name):
        if name == "ce":
            self.ce_depth += 1
        try:
            with torch.profiler.record_function(PREFIX + name) if self.running else nullcontext():
                yield
        finally:
            if name == "ce":
                self.ce_depth -= 1

    def start_profile(self):
        if self.profiler is not None:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if str(self.device).startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = self.profiler_factory(activities=activities, record_shapes=True,
                                            profile_memory=self.profile_memory,
                                            with_stack=self.with_stack)
        self.profiler.start()
        self.running = True

    def begin_forward(self):
        if self.updates < self.warmup or self.updates >= self.warmup + self.active:
            return
        if self.profiler is None:
            self.start_profile()
        if self.step_scope is None:
            self.step_scope = torch.profiler.record_function(PREFIX + "train_step")
            self.step_scope.__enter__()
            self.step_started = time.perf_counter()

    def after_update(self):
        self.updates += 1
        if self.step_scope is not None:
            elapsed = time.perf_counter() - self.step_started
            self.step_scope.__exit__(None, None, None)
            self.step_scope = None
            self.step_started = None
            self.recorded_steps.append({"invocation_update": self.updates, "forward_to_optimizer_s": elapsed})
            if "source_step" in self.metadata:
                self.recorded_steps[-1]["global_step"] = int(self.metadata["source_step"]) + self.updates

    def finish(self, error=None):
        # Stop only after Trainer.run has finished its update, logging, and
        # save epilogue. No profiler.stop sync is inserted into a step timer.
        if self.step_scope is not None:
            self.step_scope.__exit__(None, None, None)
            self.step_scope = None
        self.output.mkdir(parents=True, exist_ok=True)
        result = {**self.metadata, "warmup_updates": self.warmup,
                  "torch": torch.__version__, "hip": torch.version.hip, "device": self.device,
                  "requested_active_updates": self.active, "completed_updates": self.updates,
                  "recorded_updates": len(self.recorded_steps), "steps": self.recorded_steps,
                  "record_shapes": True, "profile_memory": self.profile_memory,
                  "with_stack": self.with_stack,
                  "status": "failed" if error else "complete",
                  "error": str(error) if error else None}
        if self.profiler is not None:
            self.profiler.stop()
            self.running = False
            trace = self.output / "trace.json"
            self.profiler.export_chrome_trace(str(trace))
            result["trace"] = trace.name
            result.update(summarize_events(self.profiler.events()))
            (self.output / "operators.txt").write_text(self.profiler.key_averages(group_by_input_shape=True).table(
                sort_by="self_device_time_total" if str(self.device).startswith("cuda") else "self_cpu_time_total", row_limit=80))
        if not error and len(self.recorded_steps) != self.active:
            result["status"] = "incomplete"
        (self.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result

    @contextmanager
    def instrument(self, trainer):
        self.owner = trainer
        with ExitStack() as stack:
            def wrap(target, name, phase, *, before=None, after=None, owned=False):
                original = getattr(target, name)

                @functools.wraps(original)
                def replacement(*args, **kwargs):
                    if owned and (not args or args[0] is not trainer):
                        return original(*args, **kwargs)
                    if before is not None:
                        before()
                    with self.region(phase):
                        value = original(*args, **kwargs)
                    if after is not None:
                        after()
                    return value
                stack.enter_context(patch.object(target, name, replacement))

            wrap(Trainer, "_forward_loss", "forward", before=self.begin_forward, owned=True)
            wrap(Trainer, "_clip", "clip", owned=True)
            wrap(Trainer, "_maybe_save", "save", owned=True)
            wrap(Trainer, "_eval_nll_stats", "eval", owned=True)
            wrap(torch.autograd, "backward", "backward")
            wrap(CPUOffloadAdamW, "step", "optimizer", after=self.after_update)
            wrap(PackedBinStream, "batch", "data")
            from cat_yoko import loss as loss_module
            wrap(loss_module, "linear_cross_entropy", "ce")
            original_linear = torch.nn.functional.linear

            @functools.wraps(original_linear)
            def linear(*args, **kwargs):
                weight = args[1] if len(args) > 1 else kwargs.get("weight")
                if self.running and self.ce_depth and torch.is_tensor(weight):
                    with self.region("vocabulary_head"):
                        return original_linear(*args, **kwargs)
                return original_linear(*args, **kwargs)
            stack.enter_context(patch.object(torch.nn.functional, "linear", linear))

            # Module regions expose MoE, attention and checkpoint recomputation
            # in the trace while the enclosing forward/backward owns accounting.
            from cat_yoko.attention import CrossAttention, WindowAttention
            from cat_yoko.moe import MoE
            wrap(MoE, "forward", "moe")
            wrap(WindowAttention, "forward", "window_attention")
            wrap(CrossAttention, "forward", "cross_attention")
            original_log = Trainer._log

            @functools.wraps(original_log)
            def logged(owner, *args, **kwargs):
                value = original_log(owner, *args, **kwargs)
                if owner is trainer:
                    if self.updates == self.warmup and self.profiler is None:
                        # The warmup update's core timer already ended. Keep
                        # profiler initialization out of the next step timer.
                        self.start_profile()
                    elif self.running:
                        self.profiler.step()
                return value
            stack.enter_context(patch.object(Trainer, "_log", logged))
            yield

    @contextmanager
    def wrap_trainer_run(self):
        original = Trainer.run

        @functools.wraps(original)
        def run(trainer, *args, **kwargs):
            error = None
            try:
                with self.instrument(trainer):
                    return original(trainer, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                self.finish(error)
        with patch.object(Trainer, "run", run):
            yield


def validate_profile_args(parser, args):
    if args.profile_warmup != 2 or not 4 <= args.profile_active <= 8:
        parser.error("profile-warmup must be 2; profile-active must be in [4,8]")
    total = args.profile_warmup + args.profile_active
    if args.run_steps is not None and args.run_steps != total:
        parser.error("run-steps must equal profile-warmup + profile-active")
    args.run_steps = total
    model_bench.validate_args(parser, args)
    if args.parity_only or args.moe_layout != "shared-storage":
        parser.error("profiling requires resident shared-storage training")
    if args.data is None or args.eval_data is None:
        parser.error("real data and held-out eval-data are required")
    if any(path.suffix not in (".bin", ".tok") for path in (args.data, args.eval_data)):
        parser.error("profiling requires packed .bin/.tok data")
    if args.eos_id != 1:
        parser.error("profiling requires explicit MiniCPM eos-id=1")
    if args.max_hours is not None:
        parser.error("use the exact update window rather than max-hours")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("out must be empty or new, separate from previous experiments")
    resume = resolve_resume_path(args.resume)
    if out == resume.parent.resolve() or out in {args.data.parent.resolve(), args.eval_data.parent.resolve()}:
        parser.error("out must be separate from source checkpoints and data")
    return resume


def check_resume(path: Path, data: Path, seq_len: int, updates: int):
    checkpoint = load_checkpoint(path)
    extra = checkpoint.get("extra") or {}
    if not is_trainable_ckpt(checkpoint) or extra.get("phase") != "B0":
        raise ValueError("source must be a B0 trainable overlay")
    optimizer = checkpoint.get("optimizer")
    if not optimizer or not optimizer.get("state"):
        raise ValueError("source must include Adam state; profiling may not restart moments")
    weights = checkpoint["trainable"]
    states = optimizer["state"]
    parameters = [parameter for group in optimizer["param_groups"] for parameter in group["params"]]
    if len(parameters) != len(set(parameters)) or set(parameters) != set(states) or len(states) != len(weights):
        raise ValueError("source Adam states must map exactly to the trainable parameters")
    counters = set()
    for state in states.values():
        step = state.get("step")
        step = int(step.item()) if torch.is_tensor(step) else int(step)
        counters.add(step)
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if not torch.is_tensor(value) or value.device.type != "cpu" or value.dtype != torch.float32:
                raise ValueError("source Adam moments must be CPU FP32")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("source Adam moments must be finite")
    if len(counters) != 1 or min(counters) < 1:
        raise ValueError("source Adam step counters must agree and be positive")
    stream = extra.get("stream") or {}
    rows = data.stat().st_size // (seq_len * 4)
    if stream.get("kind") != "packed" or int(stream.get("stride", 0)) != 1:
        raise ValueError("source must resume a single-rank packed stream")
    cursor = int(stream.get("i", -1))
    if data.stat().st_size % (seq_len * 4) or int(stream.get("nseq", 0)) != rows or cursor < 0 or cursor + updates > rows:
        raise ValueError("profile window would wrap or mismatch the source packed corpus")
    return {"source_checkpoint": str(path.resolve()), "source_step": int(extra["step"]),
            "source_adam_step": min(counters), "source_stream_i": cursor,
            "source_data": str(data.resolve()), "source_data_rows": rows,
            "optimizer_states": len(states)}


def build_parser():
    parser = model_bench.build_parser()
    parser.description = __doc__
    parser.set_defaults(moe_layout="shared-storage", run_steps=None, save_every=0,
                        eval_every=0, save_optim=True, deterministic_parity=True,
                        reference_repeat=True)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-active", type=int, default=4)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--packed-attention", action="store_true")
    parser.add_argument("--batched-grad-norm", action="store_true")
    return parser


def benchmark_arguments(args, resume):
    values = {
        "base": args.base, "resume": resume, "out": args.out, "device": args.device,
        "moe-layout": "shared-storage", "parity-seqs": args.parity_seqs,
        "seq-len": args.seq_len, "run-steps": args.run_steps, "data": args.data,
        "eval-data": args.eval_data, "eos-id": args.eos_id,
        "eval-every": args.eval_every, "eval-batches": args.eval_batches,
        "save-every": args.save_every, "keep-last": args.keep_last,
        "save-every-seconds": args.save_every_seconds,
        "cpu-threads": args.cpu_threads, "loss-atol": args.loss_atol,
        "grad-relative-l2": args.grad_relative_l2, "output-relative-l2": args.output_relative_l2,
    }
    command = [value for key, value in values.items() for value in ("--" + key, str(value))]
    command.extend(("--save-optim", "--deterministic-parity", "--reference-repeat"))
    if args.wait_gpu_idle:
        command.append("--wait-gpu-idle")
    if args.gpu_idle_max_wait is not None:
        command.extend(("--gpu-idle-max-wait", str(args.gpu_idle_max_wait)))
    for name in ("packed_attention", "batched_grad_norm"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # model_bench validates positive run_steps; establish the requested total
    # before using that shared validation.
    if args.run_steps is None:
        args.run_steps = args.profile_warmup + args.profile_active
    resume = validate_profile_args(parser, args)
    metadata = check_resume(resume, args.data, args.seq_len, args.run_steps)
    metadata.update(eval_data=str(args.eval_data.resolve()), eos_id=args.eos_id,
                    packed_attention=args.packed_attention, batched_grad_norm=args.batched_grad_norm)
    session = TrainingProfile(args.out / "profile", warmup=args.profile_warmup,
                              active=args.profile_active, device=args.device,
                              profile_memory=args.profile_memory, with_stack=args.profile_stack,
                              metadata=metadata)
    if args.packed_attention or args.batched_grad_norm:
        from operators.rocm import candidate_bench
        entry = candidate_bench.main
    else:
        entry = model_bench.main
    with session.wrap_trainer_run():
        code = entry(benchmark_arguments(args, resume))
    if code == 0 and len(session.recorded_steps) != args.profile_active:
        raise RuntimeError("training completed without the full requested profiling window")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
