"""Read-only CPU recovery check for the first new-backend checkpoint."""
import datetime
import hashlib
import json
import os
import pathlib
import random
import sys

import torch

source_dir, checkpoint, receipt = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(source_dir))
from cat_yoko.optim import CPUOffloadAdamW, _no_weight_decay_name

assert not torch.cuda.is_available(), 'audit must not allocate on GPU'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
weights, optimizer, extra = saved['trainable'], saved['optimizer'], saved['extra']
assert len(weights) == len(optimizer['state']) == 132
parameters = {name: torch.nn.Parameter(value) for name, value in weights.items()}
names = [[name for name, p in parameters.items() if p.ndim >= 2 and not _no_weight_decay_name(name)],
         [name for name, p in parameters.items() if p.ndim < 2 or _no_weight_decay_name(name)]]
groups = [{**{k: v for k, v in old.items() if k != 'params'},
           'params': [parameters[n] for n in group_names]}
          for old, group_names in zip(optimizer['param_groups'], names)]
assert len(groups) == 2 and all(len(g['params']) == len(old['params'])
                              for g, old in zip(groups, optimizer['param_groups']))
restored = CPUOffloadAdamW(groups, state_dtype=torch.float32, retain_state=True)
restored.load_state_dict(optimizer)
exported = restored.state_dict()
assert exported['param_groups'] == optimizer['param_groups']
assert list(exported['state']) == list(optimizer['state'])
for parameter_id, old in optimizer['state'].items():
    new = exported['state'][parameter_id]
    assert new['step'] == old['step'] == extra['stream']['i']
    for name in ('exp_avg', 'exp_avg_sq'):
        x, y = old[name], new[name]
        assert x.dtype == y.dtype == torch.float32
        assert x.device.type == y.device.type == 'cpu'
        assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
python_rng = random.Random()
python_rng.setstate(extra['rng_py'])
assert python_rng.getstate() == extra['rng_py']
assert extra['rng_torch'].dtype == torch.uint8 and extra['rng_torch'].device.type == 'cpu'
torch.set_rng_state(extra['rng_torch'])
assert torch.equal(torch.get_rng_state(), extra['rng_torch'])
assert isinstance(extra['rng_cuda'], list) and len(extra['rng_cuda']) == 1
assert all(t.dtype == torch.uint8 and t.device.type == 'cpu' and t.numel() > 0
           for t in extra['rng_cuda'])
sha = hashlib.sha256()
with checkpoint.open('rb') as f:
    for block in iter(lambda: f.read(8 << 20), b''):
        sha.update(block)
result = {'pass': True, 'audited_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
          'checkpoint_sha256': sha.hexdigest(), 'step': extra['step'], 'cursor': extra['stream']['i'],
          'adam_step': extra['stream']['i'], 'parameters': 132, 'moments': 264,
          'native_optimizer_load_export_bitwise': True, 'optimizer_groups_preserved': True,
          'python_rng_restored': True, 'torch_cpu_rng_restored': True,
          'serialized_cuda_rng_states': 1, 'cuda_rng_restoration_exercised_in_this_cpu_audit': False,
          'audit_gpu_allocation': False, 'checkpoint_format': saved['kind']}
receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, allow_nan=False), flush=True)
