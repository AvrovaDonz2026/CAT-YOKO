#!/usr/bin/env bash
# Push selected CAT-YOKO artifacts to HuggingFace Hub via git SSH.
#
# Target: git@hf.co:AvrovaDonz/CAT-YOKO
# Staging default: /tmp/cat-yoko-hf  (not the GitHub checkout)
# IdentityFile: ${HF_SSH_KEY:-$HOME/.ssh/id_ed25519_hf_cat_yoko}
#
# Conservative:
#   - does not source repo env files, tokens, or keys
#   - never prints key material; never git-adds ~/.ssh/*
#   - refuses files >50GiB; warns if a file is >5GiB
#   - git-adds only the files copied this run
#
# Usage:
#   ./scripts/push_to_hf.sh --dry-run
#   ./scripts/push_to_hf.sh
#   ./scripts/push_to_hf.sh --staging /tmp/cat-yoko-hf checkpoints/b1/trainable.pt

set -euo pipefail

REMOTE="git@hf.co:AvrovaDonz/CAT-YOKO"
DEFAULT_STAGING="${HF_STAGING:-/tmp/cat-yoko-hf}"
DEFAULT_PT="checkpoints/b0/trainable.pt"
BRANCH="${HF_BRANCH:-main}"
GIB=$((1024 * 1024 * 1024))
MAX_BYTES=$((50 * GIB))
WARN_BYTES=$((5 * GIB))

DRY_RUN=0
STAGING="$DEFAULT_STAGING"
EXTRA_FILES=()

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Push selected artifacts to HuggingFace Hub (AvrovaDonz/CAT-YOKO).

  --dry-run         print plan; no clone, commit, or push
  --staging DIR     staging git dir (default: /tmp/cat-yoko-hf or $HF_STAGING)
  --branch NAME     HF branch (default: main or $HF_BRANCH)
  --help            this help

Default payload: huggingface/README.md (model card) plus huggingface/.gitattributes
when those files exist. Extra path arguments are copied as additional artifacts
(paths relative to the GitHub repo root, or absolute overlay files). Weights are
not in the GitHub tree; pass a local .pt to upload.

IdentityFile: $HF_SSH_KEY or $HOME/.ssh/id_ed25519_hf_cat_yoko
EOF
}

abspath() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$p"
  else
    python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$p"
  fi
}

human_bytes() {
  local n="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "$n"
  else
    echo "${n}B"
  fi
}

file_size() {
  stat -c '%s' "$1"
}

# Detect key *files* without embedding a private-key header in this script
# (tests forbid that substring). Never print matches. Skip large binaries.
looks_like_key_material() {
  local f="$1" sz
  [[ -f "$f" ]] || return 1
  sz="$(file_size "$f")"
  # PEM / OpenSSH keys are small; .pt graphs are not.
  (( sz > 0 && sz < 65536 )) || return 1
  LC_ALL=C grep -a -q "PRIVATE KEY" "$f"
}

under_ssh_dir() {
  local p="$1"
  local ssh_root home_resolved
  home_resolved="$(abspath "${HOME:-/root}")"
  ssh_root="$(abspath "$home_resolved/.ssh")"
  p="$(abspath "$p")"
  [[ "$p" == "$ssh_root" || "$p" == "$ssh_root"/* ]]
}

is_inside() {
  local inner="$1"
  local outer="$2"
  inner="$(abspath "$inner")"
  outer="$(abspath "$outer")"
  [[ "$inner" == "$outer" || "$inner" == "$outer"/* ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --staging)
      [[ $# -ge 2 ]] || die "--staging needs a directory"
      STAGING="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || die "--branch needs a name"
      BRANCH="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_FILES+=("$@"); break ;;
    -*) die "unknown flag: $1 (see --help)" ;;
    *) EXTRA_FILES+=("$1"); shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGING="$(abspath "$STAGING")"
ROOT="$(abspath "$ROOT")"

[[ "$STAGING" != "$ROOT" ]] || die "staging must not be the GitHub checkout ($ROOT)"
is_inside "$STAGING" "$ROOT" && die "staging must not live inside the GitHub checkout"

IDENTITY="${HF_SSH_KEY:-$HOME/.ssh/id_ed25519_hf_cat_yoko}"
if [[ "$IDENTITY" == *$'\n'* || "$IDENTITY" == *"PRIVATE KEY"* ]]; then
  die "HF_SSH_KEY must be an IdentityFile path, not key material"
fi

# Collect (src, dest-relative-to-staging) pairs.
SRCS=()
DSTS=()

add_artifact() {
  local src="$1"
  local dst="$2"
  local abs d
  [[ -e "$src" ]] || die "missing artifact: $src"
  [[ -f "$src" ]] || die "refusing non-file artifact: $src"
  abs="$(abspath "$src")"
  if under_ssh_dir "$abs"; then
    die "refusing to stage a path under ~/.ssh: $src"
  fi
  if is_inside "$abs" "$STAGING"; then
    die "artifact path is already inside staging: $src"
  fi
  if looks_like_key_material "$abs"; then
    die "refusing to stage key material: $src"
  fi
  for d in "${DSTS[@]+"${DSTS[@]}"}"; do
    if [[ "$d" == "$dst" ]]; then
      echo "skip duplicate dest $dst"
      return 0
    fi
  done
  SRCS+=("$abs")
  DSTS+=("$dst")
}

resolve_extra() {
  local raw="$1"
  local src
  if [[ "$raw" == /* ]]; then
    src="$raw"
  else
    src="$ROOT/$raw"
  fi
  [[ -e "$src" ]] || die "missing artifact: $raw"
  src="$(abspath "$src")"
  if is_inside "$src" "$ROOT"; then
    echo "${src#"$ROOT"/}"
  else
    basename "$src"
  fi
}

if [[ -f "$ROOT/$DEFAULT_PT" ]]; then
  add_artifact "$ROOT/$DEFAULT_PT" "$DEFAULT_PT"
else
  echo "note: $DEFAULT_PT is not in git (weights live on HuggingFace); pass a local overlay to upload" >&2
fi

if [[ -f "$ROOT/huggingface/README.md" ]]; then
  add_artifact "$ROOT/huggingface/README.md" "README.md"
fi
if [[ -f "$ROOT/huggingface/.gitattributes" ]]; then
  add_artifact "$ROOT/huggingface/.gitattributes" ".gitattributes"
fi
# Hub folder card for the live B0 overlay. Weights are extra args; this
# README must still ship or https://huggingface.co/.../checkpoints/b0-full
# stays on a stale 6000D pin.
if [[ -f "$ROOT/checkpoints/b0-full/README.md" ]]; then
  add_artifact "$ROOT/checkpoints/b0-full/README.md" "checkpoints/b0-full/README.md"
fi

for raw in "${EXTRA_FILES[@]+"${EXTRA_FILES[@]}"}"; do
  dest="$(resolve_extra "$raw")"
  if [[ "$raw" == /* ]]; then
    add_artifact "$raw" "$dest"
  else
    add_artifact "$ROOT/$raw" "$dest"
  fi
done

[[ ${#SRCS[@]} -gt 0 ]] || die "nothing to push"

echo "HF remote: $REMOTE"
echo "staging:   $STAGING"
echo "branch:    $BRANCH"
echo "IdentityFile path: $IDENTITY"
echo "artifacts:"

i=0
while [[ $i -lt ${#SRCS[@]} ]]; do
  src="${SRCS[$i]}"
  dst="${DSTS[$i]}"
  sz="$(file_size "$src")"
  echo "  $src -> $dst ($(human_bytes "$sz"))"
  if (( sz > MAX_BYTES )); then
    die "refuse $dst: $(human_bytes "$sz") > 50GiB"
  fi
  if (( sz > WARN_BYTES )); then
    echo "warning: $dst is $(human_bytes "$sz") > 5GiB" >&2
  fi
  i=$((i + 1))
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry-run: would git lfs install"
  echo "dry-run: would ensure remote hf $REMOTE"
  echo "dry-run: would copy artifacts into $STAGING, commit, and git push hf HEAD:$BRANCH"
  if [[ ! -f "$IDENTITY" ]]; then
    echo "warning: IdentityFile does not exist yet: $IDENTITY" >&2
  elif looks_like_key_material "$IDENTITY"; then
    : # key file present; do not print it
  fi
  echo "dry-run: no clone / commit / push"
  exit 0
fi

command -v git >/dev/null 2>&1 || die "git not found"
if ! git lfs version >/dev/null 2>&1; then
  die "git-lfs not found (needed for Hub weights)"
fi

[[ -f "$IDENTITY" ]] || die "missing IdentityFile: $IDENTITY"
[[ -r "$IDENTITY" ]] || die "IdentityFile is not readable"
if looks_like_key_material "$IDENTITY"; then
  :
else
  echo "warning: IdentityFile does not look like an OpenSSH key file" >&2
fi
if is_inside "$(abspath "$IDENTITY")" "$ROOT"; then
  die "IdentityFile must not live in the GitHub checkout"
fi
perm="$(stat -c '%a' "$IDENTITY" 2>/dev/null || echo "?")"
if [[ "$perm" != "600" && "$perm" != "400" && "$perm" != "?" ]]; then
  echo "warning: IdentityFile mode is $perm (prefer 600)" >&2
fi

# shellcheck disable=SC2027
export GIT_SSH_COMMAND="ssh -i $(printf '%q' "$IDENTITY") -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

git lfs install --skip-repo

mkdir -p "$STAGING"
if [[ -d "$STAGING/.git" ]]; then
  echo "using existing staging git repo"
else
  if [[ -n "$(ls -A "$STAGING" 2>/dev/null || true)" ]]; then
    die "staging $STAGING is not empty and is not a git repo"
  fi
  echo "cloning $REMOTE into staging"
  if ! GIT_LFS_SKIP_SMUDGE=1 git clone "$REMOTE" "$STAGING"; then
    echo "clone failed; initializing empty staging repo"
    git -C "$STAGING" init
    git -C "$STAGING" checkout -B "$BRANCH"
  fi
fi

git -C "$STAGING" lfs install --local

if git -C "$STAGING" remote get-url hf >/dev/null 2>&1; then
  have="$(git -C "$STAGING" remote get-url hf)"
  [[ "$have" == "$REMOTE" ]] || die "remote hf is '$have', expected $REMOTE"
else
  git -C "$STAGING" remote add hf "$REMOTE"
fi

if ! git -C "$STAGING" config --get user.email >/dev/null; then
  git -C "$STAGING" config user.email "cat-yoko-hf@users.noreply.huggingface.co"
fi
if ! git -C "$STAGING" config --get user.name >/dev/null; then
  git -C "$STAGING" config user.name "CAT-YOKO"
fi

i=0
ADDED=()
while [[ $i -lt ${#SRCS[@]} ]]; do
  src="${SRCS[$i]}"
  dst="${DSTS[$i]}"
  dest_path="$STAGING/$dst"
  mkdir -p "$(dirname "$dest_path")"
  cp -a "$src" "$dest_path"
  ADDED+=("$dst")
  i=$((i + 1))
done

if [[ ! -f "$STAGING/.gitattributes" ]] || ! grep -q 'filter=lfs' "$STAGING/.gitattributes" 2>/dev/null; then
  {
    [[ -f "$STAGING/.gitattributes" ]] && cat "$STAGING/.gitattributes"
    cat <<'EOF'
*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
EOF
  } >"$STAGING/.gitattributes.tmp"
  mv "$STAGING/.gitattributes.tmp" "$STAGING/.gitattributes"
  ADDED+=(".gitattributes")
fi

# git-add only copied payload paths; never ~/.ssh, never $HOME, never -A.
for dst in "${ADDED[@]}"; do
  case "$dst" in
    .ssh|.ssh/*|*"/.ssh/"*) die "internal error: refusing to git-add $dst" ;;
  esac
  git -C "$STAGING" add -- "$dst"
done

if git -C "$STAGING" diff --cached --quiet; then
  echo "nothing new to commit"
  exit 0
fi

msg="CAT-YOKO Hub: ${ADDED[*]}"
git -C "$STAGING" commit -m "$msg"
git -C "$STAGING" push hf "HEAD:$BRANCH"
echo "pushed to $REMOTE ($BRANCH)"
