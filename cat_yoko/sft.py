"""Phase F SFT rows: UltraChat / messages / alpaca → token ids + response labels.

Does not download UltraChat. ``prepare --mix phase-f`` writes jsonl; DummyStream
still masks a prompt prefix for ``--try``.
"""

from __future__ import annotations

from typing import Iterator

from cat_yoko.tokenizer import Tokenizer

_USER = {"user", "human", "prompt", "question", "system", "tool", "function"}
_ASSISTANT = {"assistant", "gpt", "bot", "model", "output", "response"}


def _role(raw: object) -> str:
    role = str(raw or "").strip().lower()
    if role in _USER:
        return "user"
    if role in _ASSISTANT:
        return "assistant"
    return ""


def _text(raw: object) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def sft_turns(row: dict) -> list[tuple[str, str]]:
    """Parse UltraChat ``data``, ShareGPT ``conversations``, Chat ``messages``."""
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        out: list[tuple[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            text = _text(m.get("content") or m.get("value") or m.get("text"))
            if text is None:
                continue
            role = _role(m.get("role") or m.get("from") or m.get("author"))
            if not role:
                role = "user" if (not out or out[-1][0] == "assistant") else "assistant"
            out.append((role, text))
        return out
    data = row.get("data")
    if isinstance(data, list) and data and all(isinstance(x, str) for x in data):
        out = []
        for i, t in enumerate(data):
            text = _text(t)
            if text is None:
                continue
            out.append(("user" if i % 2 == 0 else "assistant", text))
        return out
    prompt = _text(row.get("instruction") or row.get("prompt") or row.get("input"))
    resp = _text(row.get("output") or row.get("response") or row.get("completion"))
    if prompt and resp:
        inp = _text(row.get("input"))
        if inp and inp != prompt:
            prompt = f"{prompt}\n{inp}"
        return [("user", prompt), ("assistant", resp)]
    return []


def turns_to_text(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for role, text in turns)


def encode_sft(
    tokenizer: Tokenizer,
    turns: list[tuple[str, str]],
    seq_len: int,
) -> tuple[list[int], list[int]] | None:
    """Concatenate ``role: text`` turns. User tokens (and their EOS) get ``-100``."""
    ids: list[int] = []
    labels: list[int] = []
    for role, text in turns:
        piece = [int(t) for t in tokenizer.encode(f"{role}: {text}")]
        if not piece:
            continue
        ids.extend(piece)
        if role == "assistant":
            labels.extend(piece)
        else:
            labels.extend([-100] * len(piece))
    if not ids:
        return None
    cut = max(int(seq_len), 1)
    if len(ids) > cut:
        # Keep the response tail so a short seq_len still has trainable labels.
        ids = ids[-cut:]
        labels = labels[-cut:]
    if all(x == -100 for x in labels):
        return None
    return ids, labels


def iter_sft_pairs(
    rows: Iterator[dict],
    tokenizer: Tokenizer,
    seq_len: int,
) -> Iterator[tuple[list[int], list[int]]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        turns = sft_turns(row)
        if not turns:
            continue
        packed = encode_sft(tokenizer, turns, seq_len)
        if packed is not None:
            yield packed


def pack_sft_pairs(
    pairs: Iterator[tuple[list[int], list[int]]],
    seq_len: int,
) -> Iterator[tuple[list[int], list[int]]]:
    """Concatenate response-labeled chats until ``seq_len``.

    Cross-example CE is already ignored: user tokens (and the first token of
    the next chat) are ``labels=-100``, so the last assistant token does not
    train on the next prompt.
    """
    cut = max(int(seq_len), 1)
    buf_ids: list[int] = []
    buf_lab: list[int] = []
    for ids, labels in pairs:
        if not ids:
            continue
        if len(labels) != len(ids):
            raise ValueError("SFT labels must match token length")
        if len(ids) > cut:
            ids = ids[-cut:]
            labels = labels[-cut:]
        if all(x == -100 for x in labels):
            continue
        if buf_ids and len(buf_ids) + len(ids) > cut:
            yield buf_ids, buf_lab
            buf_ids, buf_lab = [], []
        buf_ids.extend(ids)
        buf_lab.extend(labels)
        if len(buf_ids) >= cut:
            yield buf_ids[:cut], buf_lab[:cut]
            buf_ids, buf_lab = [], []
    if buf_ids and any(x != -100 for x in buf_lab):
        yield buf_ids, buf_lab
