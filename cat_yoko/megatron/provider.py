"""Megatron-LM pretrain hooks. Not wired until megatron-core is installed.

``megatron.training.pretrain`` expects::

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        model_type,
        forward_step,
        extra_args_provider=add_cat_yoko_args,
    )

Do **not** return a GPTModel from ``model_provider``. See
``MEGATRON_CUSTOM_SURFACE`` in mapping.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from typing import Any

from cat_yoko.config import CATYokoConfig
from cat_yoko.megatron.mapping import MEGATRON_LM, megatron_blueprint
from cat_yoko.parallel import ParallelPlan

_IMPORT_HINT = (
    f"Megatron backend is reserved but not installed.\n"
    f"  git clone {MEGATRON_LM}\n"
    f"  pip install -e Megatron-LM   # or: pip install 'cat-yoko[megatron]'\n"
    f"YOCO is not GPTModel — implement cat_yoko.megatron.provider.model_provider "
    f"as a custom MegatronModule using megatron_blueprint()."
)


class MegatronNotInstalled(ImportError):
    pass


class MegatronBackendNotReady(RuntimeError):
    """Installed Megatron-Core is not enough; the YOCO MegatronModule is still TBD."""


def import_megatron_core() -> Any:
    try:
        import megatron.core as mcore  # noqa: F401
    except ImportError as exc:
        raise MegatronNotInstalled(_IMPORT_HINT) from exc
    return mcore


def filter_transformer_kwargs(config_cls: type, payload: dict) -> dict:
    """Drop keys the installed Megatron-Core TransformerConfig does not know."""
    names = {f.name for f in fields(config_cls)}
    return {k: v for k, v in payload.items() if k in names}


def add_cat_yoko_args(parser: Any) -> Any:
    """``extra_args_provider`` for megatron.training.pretrain."""
    group = parser.add_argument_group("CAT-YOKO")
    group.add_argument("--cat-yoko-phase", default="B0")
    group.add_argument("--cat-yoko-config", choices=["12b", "tiny"], default="12b")
    return parser


def model_provider(pre_process: bool = True, post_process: bool = True) -> Any:
    import_megatron_core()
    raise MegatronBackendNotReady(
        "Megatron-Core is importable, but the YOCO dual-stack MegatronModule "
        "is not implemented yet. Use --backend torch for the reference graph, "
        "or fill in this provider using megatron_blueprint() "
        f"(see {MEGATRON_LM}). pre_process={pre_process} post_process={post_process}"
    )


def forward_step(data_iterator: Any, model: Any) -> Any:
    raise MegatronBackendNotReady("forward_step is a Megatron hook stub")


def train_valid_test_datasets_provider(train_val_test_num_samples: Any) -> Any:
    raise MegatronBackendNotReady("dataset provider is a Megatron hook stub")


def dump_mapping(cfg: CATYokoConfig, plan: ParallelPlan, phase: str, *, file=None) -> dict:
    payload = megatron_blueprint(cfg, plan, phase=phase)
    json.dump(payload, file or sys.stdout, indent=2)
    (file or sys.stdout).write("\n")
    return payload


def run_pretrain(cfg: CATYokoConfig, plan: ParallelPlan, phase: str) -> int:
    """Attempt Megatron pretrain. Mapping is ``--dump-megatron``; this only runs hooks."""
    del cfg, plan, phase
    print(f"CAT-YOKO Megatron backend: {MEGATRON_LM}", file=sys.stderr)
    print("Dump the dual-stack blueprint with --dump-megatron (no GPTModel).", file=sys.stderr)
    try:
        import_megatron_core()
    except MegatronNotInstalled as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        model_provider()
    except MegatronBackendNotReady as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 1
