"""Measured scGPT embeddings, adapted from official scgpt/tasks/cell_emb.py.

Input: author-distributed pancreas h5ad, already inspected on Spark.
Expression values are used as supplied, with per-cell quantile binning only.
Celltype is metadata, never an input to the encoder; no classifier is trained.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ResourceSampler:
    """Sample this Python process and CUDA allocator every 0.2 seconds."""
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            vm = psutil.virtual_memory()
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'process_rss_bytes': self.process.memory_info().rss,
                'process_cpu_percent': self.process.cpu_percent(),
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': vm.available,
            })

    def loop(self):
        while not self.stop_event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.process.cpu_percent()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--split', choices=['train', 'test'], default='test')
parser.add_argument('--n-cells', type=int, default=1000, help='0 means all cells')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--max-length', type=int, default=1200, help='includes the CLS token')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', type=Path, required=True)
opt = parser.parse_args()
if opt.batch_size < 1 or opt.max_length < 2 or opt.n_cells < 0:
    parser.error('batch-size >= 1, max-length >= 2, n-cells >= 0 required')
out = opt.out
out.mkdir(parents=True, exist_ok=True)
if (out / 'summary.json').exists():
    raise FileExistsError(f'Results already exist: {out}')
started_at = datetime.now(timezone.utc).isoformat()
torch.set_num_threads(8)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; stopping instead of measuring CPU inference')
torch.cuda.set_device(0)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
seed_all(opt.seed)

model_dir = ROOT / 'models/whole-human'
model_args = json.loads((model_dir / 'args.json').read_text())
if model_args.get('n_bins') != 51 or model_args.get('input_style') != 'binned':
    raise ValueError('This script requires the inspected 51-bin, binned-input checkpoint')
if model_args.get('input_emb_style', 'continuous') != 'continuous':
    raise ValueError('This script requires a continuous expression-value encoder')
if model_args.get('pre_norm', False):
    raise ValueError('This script implements the inspected post-norm checkpoint')
vocab = GeneVocab.from_file(model_dir / 'vocab.json')
for special in [model_args['pad_token'], '<cls>', '<eoc>']:
    if special not in vocab:
        vocab.append_token(special)
pad_id, cls_id = vocab[model_args['pad_token']], vocab['<cls>']
pad_value = model_args['pad_value']
vocab.set_default_index(pad_id)

input_path = ROOT / f'data/pancreas_demo/demo_{opt.split}.h5ad'
t0 = time.perf_counter()
adata = ad.read_h5ad(input_path)
expected = (10600, 3000) if opt.split == 'train' else (4218, 3000)
if adata.shape != expected or 'Celltype' not in adata.obs:
    raise ValueError(f'Input differs from the inspected data: {adata.shape}, {adata.obs.columns.tolist()}')
if not adata.var_names.is_unique or not adata.obs_names.is_unique:
    raise ValueError('Gene names and cell IDs must be unique')
n = adata.n_obs if opt.n_cells == 0 else opt.n_cells
if n > adata.n_obs:
    raise ValueError(f'Requested {n} cells, but input contains {adata.n_obs}')
indices = (np.arange(adata.n_obs) if n == adata.n_obs else
           np.sort(np.random.default_rng(opt.seed).choice(adata.n_obs, n, replace=False)))
gene_names = adata.var_names.astype(str).to_numpy()
gene_mask = np.array([g in vocab for g in gene_names], dtype=bool)
if int(gene_mask.sum()) != 2999:
    raise ValueError(f'Expected 2999 matched genes; got {int(gene_mask.sum())}')
selected = adata[indices, gene_mask].copy()
matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
matrix = matrix.astype(np.float32, copy=False)
if not np.isfinite(matrix).all() or (matrix < 0).any():
    raise ValueError('Selected expression matrix must be finite and nonnegative')
nnz = np.count_nonzero(matrix, axis=1)
if (nnz == 0).any():
    raise ValueError('Some selected cells have no nonzero expression after vocabulary filtering')
gene_ids = np.array(vocab(selected.var_names.tolist()), dtype=np.int64)
cell_metadata = selected.obs.copy()
cell_metadata.index.name = 'cell_id'
cell_metadata.insert(0, 'original_row_index', indices)
cell_metadata.insert(1, 'embedding_row', np.arange(n))
cell_metadata['nonzero_genes_before_sampling'] = nnz
cell_metadata['tokens_before_sampling_including_cls'] = nnz + 1
cell_metadata['tokens_used_including_cls'] = np.minimum(nnz + 1, opt.max_length)
cell_metadata['sampled_due_to_length'] = nnz + 1 > opt.max_length
cell_metadata.to_csv(out / 'selected_cells.tsv', sep='\t')
pd.DataFrame({'gene_name': selected.var_names, 'vocab_id': gene_ids}).to_csv(
    out / 'matched_genes.tsv', sep='\t', index=False)
(out / 'excluded_genes.txt').write_text('\n'.join(gene_names[~gene_mask]) + '\n')
del adata, selected
gc.collect()
input_preparation_seconds = time.perf_counter() - t0
print(f'[Input] source={opt.split}; selected matrix={matrix.shape}; matched genes=2999/3000', flush=True)
print(f'[Length] max_length={opt.max_length} includes CLS; sampled cells={int((nnz + 1 > opt.max_length).sum())}/{n}', flush=True)

# Snapshot configuration, source code and file identities for reproducibility.
for name in ['args.json', 'vocab.json']:
    shutil.copy2(model_dir / name, out / f'pretrained_{name}')
source_dir = out / 'source_snapshot'
source_dir.mkdir(exist_ok=True)
for name, obj in [('data_collator.py', DataCollator), ('model.py', TransformerModel),
                  ('gene_tokenizer.py', GeneVocab), ('preprocess.py', binning)]:
    shutil.copy2(inspect.getfile(obj), source_dir / name)
versions = {}
for pkg in ['torch', 'scgpt', 'anndata', 'scanpy', 'numpy', 'pandas', 'scipy', 'psutil']:
    versions[pkg] = metadata.version(pkg)
direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
device = torch.cuda.get_device_properties(0)
write_json(out / 'environment.json', {
    'python': sys.version, 'architecture': platform.machine(), 'packages': versions,
    'scgpt_install_source': json.loads(direct_url) if direct_url else None,
    'cuda_version': torch.version.cuda, 'gpu_name': device.name,
    'gpu_compute_capability': [device.major, device.minor],
    'cuda_reported_total_memory_gib': device.total_memory / GIB,
    'system_total_memory_gib': psutil.virtual_memory().total / GIB,
    'torch_cpu_threads': torch.get_num_threads(),
})
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
(out / 'requirements_freeze.txt').write_text(freeze.stdout)
write_json(out / 'provenance.json', {
    'input': {'path': str(input_path), 'sha256': sha256(input_path)},
    'checkpoint': {'path': str(model_dir / 'best_model.pt'), 'sha256': sha256(model_dir / 'best_model.pt')},
    'script_sha256': sha256(__file__),
    'reference_implementation': 'https://github.com/bowang-lab/scGPT/blob/cebd6fae655b9c585a4807daa3ac31bb764f06b4/scgpt/tasks/cell_emb.py',
    'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_reference_mapping.html',
})
configuration = {
    'started_at_utc': started_at, 'task': 'pretrained cell embeddings',
    'split': opt.split, 'n_cells': n, 'batch_size': opt.batch_size,
    'max_length_including_cls': opt.max_length, 'seed': opt.seed,
    'precision': 'FP32', 'autocast': False, 'tf32': False,
    'backend': 'native PyTorch Transformer; use_fast_transformer=False',
    'additional_normalize_total': False, 'additional_log1p': False,
    'expression_provenance': 'Values supplied in h5ad; no raw layer or preprocessing metadata',
    'binning': 'official DataCollator: per-cell quantiles, n_bins=51; nonzero expression only',
    'length_policy': 'keep CLS; if over length, randomly sample gene/value pairs; pad to batch maximum capped at max_length',
    'pooling': 'CLS position; L2-normalized 512-dimensional embedding',
    'labels_used_as_model_input': False, 'classifier_training': False,
    'timed_scope': 'gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording',
    'excluded_from_inference_timer': 'imports, h5ad loading, cell/gene selection, model loading, one-batch warmup, file saving, accuracy evaluation',
    'measurement_repeats': 1,
}
write_json(out / 'run_config.json', configuration)

t0 = time.perf_counter()
model = TransformerModel(
    ntoken=len(vocab), d_model=model_args['embsize'], nhead=model_args['nheads'],
    d_hid=model_args['d_hid'], nlayers=model_args['nlayers'],
    nlayers_cls=model_args['n_layers_cls'], n_cls=1, vocab=vocab,
    dropout=model_args['dropout'], pad_token=model_args['pad_token'], pad_value=pad_value,
    do_mvc=True, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
    input_emb_style='continuous', cell_emb_style='cls', explicit_zero_prob=False,
    use_fast_transformer=False, pre_norm=False,
)
state = torch.load(model_dir / 'best_model.pt', map_location='cpu', weights_only=True)
if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
    raise TypeError('Expected a plain tensor state_dict; checkpoint format requires review')
# These are the official load_pretrained native-transformer rename rules.
rules = {
    r'self_attn\._impl\.Wqkv\.': 'self_attn.in_proj_',
    r'self_attn\.Wqkv\.': 'self_attn.in_proj_',
    r'self_attn\._impl\.out_proj\.': 'self_attn.out_proj.',
}
converted = {}
for key, value in state.items():
    for pattern, replacement in rules.items():
        key = re.sub(pattern, replacement, key)
    if key in converted:
        raise ValueError(f'Duplicate checkpoint key after conversion: {key}')
    converted[key] = value
current = model.state_dict()
compatible = {k: v for k, v in converted.items() if k in current and v.shape == current[k].shape}
core_prefixes = ('encoder.', 'value_encoder.', 'transformer_encoder.')
required = [k for k in current if k.startswith(core_prefixes)]
missing_core = [k for k in required if k not in compatible]
load_report = {
    'required_embedding_keys': len(required), 'missing_embedding_keys': missing_core,
    'loaded_keys': sorted(compatible),
    'unused_or_incompatible_checkpoint_keys': sorted(set(converted) - set(compatible)),
    'model_keys_not_loaded': sorted(set(current) - set(compatible)),
    'note': 'All parameters used by _encode must load; decoder heads are unused in this embedding run.',
}
write_json(out / 'weight_loading.json', load_report)
if missing_core:
    raise RuntimeError(f'Embedding weights did not fully load: {missing_core[:10]}')
model.load_state_dict(compatible, strict=False)
parameter_count_total = sum(p.numel() for p in model.parameters())
parameter_count_encoder = sum(p.numel() for name, p in model.named_parameters() if name.startswith(core_prefixes))
del state, converted, current, compatible
gc.collect()
model.float().to('cuda').eval()
torch.cuda.synchronize()
model_loading_seconds = time.perf_counter() - t0
print(f'[Model] embedding weights loaded: {len(required)}/{len(required)}; output_dim={model_args["embsize"]}', flush=True)

collator = DataCollator(do_padding=True, pad_token_id=pad_id, pad_value=pad_value,
    do_mlm=False, do_binning=True, max_length=opt.max_length, sampling=True, keep_first_n_tokens=1)

def make_batch(start, end):
    examples = []
    for i in range(start, end):
        nz = np.flatnonzero(matrix[i])
        genes = np.concatenate([np.array([cls_id], dtype=np.int64), gene_ids[nz]])
        values = np.concatenate([np.array([pad_value], dtype=np.float32), matrix[i, nz]])
        examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
    return collator(examples)

def encode_batch(batch):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    result = torch.nn.functional.normalize(encoded[:, 0, :], p=2, dim=1)
    return result.cpu().numpy()

# Warm the CUDA execution path, then reset random seeds before the measured run.
seed_all(opt.seed)
t0 = time.perf_counter()
with torch.inference_mode():
    warm_batch = make_batch(0, min(opt.batch_size, n))
    warm_result = encode_batch(warm_batch)
torch.cuda.synchronize()
warmup_seconds = time.perf_counter() - t0
del warm_batch, warm_result
model.cur_gene_token_embs = None
gc.collect()
seed_all(opt.seed)
torch.cuda.reset_peak_memory_stats()
embeddings = np.empty((n, model_args['embsize']), dtype=np.float32)
tokens_saved = np.full((n, opt.max_length), pad_id, dtype=np.int64)
values_saved = np.full((n, opt.max_length), pad_value, dtype=np.float32)
batch_records = []
sampler = ResourceSampler()
sampler.start()
torch.cuda.synchronize()
t0 = time.perf_counter()
try:
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, n, opt.batch_size)):
            end = min(start + opt.batch_size, n)
            batch = make_batch(start, end)
            length = batch['gene'].shape[1]
            tokens_saved[start:end, :length] = batch['gene'].numpy()
            values_saved[start:end, :length] = batch['expr'].numpy()
            embeddings[start:end] = encode_batch(batch)
            batch_records.append({'batch': batch_index, 'start': start, 'end': end,
                                  'batch_size': end - start, 'padded_token_length': length})
            if batch_index == 0:
                print(f'[Model input] gene IDs={tuple(batch["gene"].shape)}, expression bins={tuple(batch["expr"].shape)}', flush=True)
            if (batch_index + 1) % 25 == 0 or end == n:
                print(f'[Inference] {end}/{n} cells', flush=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t0
finally:
    sampler.stop()
    pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)

if not np.isfinite(embeddings).all():
    raise FloatingPointError('Embeddings contain NaN/Inf')
norms = np.linalg.norm(embeddings, axis=1)
if not np.allclose(norms, 1, atol=1e-4):
    raise ValueError('Embedding L2 norms are not approximately 1')
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
t0 = time.perf_counter()
np.save(out / 'embeddings.npy', embeddings)
np.savez_compressed(out / 'tokenized_inputs.npz', gene_ids=tokens_saved, expression_bins=values_saved)
pd.DataFrame(batch_records).to_csv(out / 'batch_shapes.tsv', sep='\t', index=False)
itos = vocab.get_itos()
used_first = int(cell_metadata['tokens_used_including_cls'].iloc[0])
pd.DataFrame({
    'position': np.arange(min(20, used_first)),
    'gene_id': tokens_saved[0, :min(20, used_first)],
    'gene_name': [itos[int(i)] for i in tokens_saved[0, :min(20, used_first)]],
    'expression_bin': values_saved[0, :min(20, used_first)],
}).to_csv(out / 'first_cell_token_example.tsv', sep='\t', index=False)
output_save_seconds = time.perf_counter() - t0
summary = {
    'status': 'complete', 'task': 'pretrained cell embeddings', 'split': opt.split,
    'cells': n, 'matched_genes': int(gene_mask.sum()), 'embedding_shape': list(embeddings.shape),
    'batch_size': opt.batch_size, 'max_length_including_cls': opt.max_length,
    'precision': 'FP32', 'seed': opt.seed,
    'parameter_count_constructed_model': parameter_count_total,
    'parameter_count_embedding_encoder': parameter_count_encoder,
    'input_preparation_seconds': input_preparation_seconds,
    'model_loading_seconds': model_loading_seconds, 'warmup_seconds': warmup_seconds,
    'inference_seconds': inference_seconds, 'cells_per_second': n / inference_seconds,
    'output_save_seconds': output_save_seconds,
    'gpu_peak_allocated_gib': peak_allocated / GIB,
    'gpu_peak_reserved_gib': peak_reserved / GIB,
    'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in sampler.rows) / GIB,
    'resource_sampling_interval_seconds': 0.2,
    'length_sampling_cells': int((nnz + 1 > opt.max_length).sum()),
    'tokens_before_sampling_min_mean_max': [int((nnz + 1).min()), float((nnz + 1).mean()), int((nnz + 1).max())],
    'tokens_after_sampling_min_mean_max': [int(np.minimum(nnz + 1, opt.max_length).min()), float(np.minimum(nnz + 1, opt.max_length).mean()), int(np.minimum(nnz + 1, opt.max_length).max())],
    'embedding_l2_norm_min_max': [float(norms.min()), float(norms.max())],
    'accuracy_evaluated': False, 'weights_updated': False,
    'timing_scope': configuration['timed_scope'],
    'memory_notes': 'CUDA allocator peaks are process-local PyTorch counters, not total device usage. Reserved memory can include warmup cache. RSS is sampled, not an exact peak. Spark uses shared memory; do not sum RSS and CUDA counters as independent physical pools.',
}
write_json(out / 'summary.json', summary)
print('\n' + json.dumps({key: summary[key] for key in [
    'cells', 'embedding_shape', 'batch_size', 'max_length_including_cls', 'precision',
    'inference_seconds', 'cells_per_second', 'gpu_peak_allocated_gib',
    'gpu_peak_reserved_gib', 'python_process_rss_peak_observed_gib', 'length_sampling_cells',
]}, indent=2), flush=True)
print('SCGPT_INFERENCE_COMPLETE', flush=True)
print('Results:', out, flush=True)
