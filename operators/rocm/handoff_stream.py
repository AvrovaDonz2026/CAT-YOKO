"""Fresh packed rows with an explicit monotonic absolute cursor.

This opt-in adapter leaves the ordinary PackedBinStream semantics untouched.
Its restore errors deliberately use RuntimeError: Trainer's legacy cross-kind
fallback catches ValueError and must never swallow a corpus identity mismatch.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from cat_yoko.data import PackedBinStream
from operators.rocm.fresh_corpus_handoff import (
    check_stream_state, digest, need, stream_state, validate_handoff,
)


class FreshPackedBinStream(PackedBinStream):
    def __init__(self, path, seq_len, *, handoff, eos_id=None, shard_id=0, num_shards=1):
        validate_handoff(handoff)
        need(Path(path).resolve() == Path(handoff["new_train_path"])
             and seq_len == 4096 and eos_id == 1 and shard_id == 0 and num_shards == 1,
             "fresh stream requires its exact single-GPU 4096/EOS=1 corpus")
        super().__init__(path, seq_len, eos_id=eos_id, shard_id=shard_id, num_shards=num_shards)
        need(self.nseq == handoff["new_nseq"], "fresh stream row count changed")
        self.handoff = handoff
        self._i = handoff["absolute_cursor_origin"]
        self._source_handoff_loaded = False
        self.batch_calls = 0
        self.first_batch_row = None
        self.last_batch_row = None

    def state_dict(self):
        return stream_state(self.handoff, self._i)

    def load_state_dict(self, state):
        legacy = state == {"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531}
        need(not legacy or (not self._source_handoff_loaded and self.batch_calls == 0),
             "legacy corpus handoff cannot reset an active stream")
        restored = check_stream_state(state, self.handoff, allow_source=not self._source_handoff_loaded)
        self._i = restored["i"]
        self._source_handoff_loaded = True

    def batch(self, micro_batch, device):
        need(type(micro_batch) is int and micro_batch > 0, "fresh batch size must be positive")
        absolute = self._i
        row = absolute - self.handoff["absolute_cursor_origin"]
        need(0 <= row and row + micro_batch <= self.nseq, "fresh corpus would wrap; continuation refused")
        self._i = row
        try:
            result = super().batch(micro_batch, device)
            need(self._i == row + micro_batch, "native stream advanced an unexpected number of rows")
        except BaseException:
            self._i = absolute
            raise
        self._i = absolute + micro_batch
        self.batch_calls += 1
        if self.first_batch_row is None:
            self.first_batch_row = row
        self.last_batch_row = row + micro_batch - 1
        return result


@contextmanager
def handoff_context(model_bench, handoff):
    """Scope the same adapter around parity and training, leaving eval native."""
    from cat_yoko import trainer as trainer_module
    validate_handoff(handoff, verify_files=True)
    original = model_bench.open_stream
    opened = []

    def fresh_open(path, vocab_size, seq_len, **options):
        if path is None or Path(path).resolve() != Path(handoff["new_train_path"]):
            return original(path, vocab_size, seq_len, **options)
        need(vocab_size == 130560 and options.get("rank", 0) == 0 and options.get("world", 1) == 1
             and not options.get("response_only", False) and not options.get("needle", False),
             "fresh stream cannot silently change vocabulary, shards or task")
        stream = FreshPackedBinStream(path, seq_len, handoff=handoff, eos_id=options.get("eos_id"))
        opened.append(stream)
        return stream

    with patch.object(model_bench, "open_stream", fresh_open), patch.object(trainer_module, "open_stream", fresh_open):
        yield opened
    need(digest(handoff["new_train_path"]) == handoff["new_train_sha256"], "fresh training corpus changed during use")
