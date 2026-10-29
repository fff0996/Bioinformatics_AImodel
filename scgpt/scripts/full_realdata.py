"""scGPT pancreas classification: supervised fine-tuning and saved-bundle prediction.

Run train through scgpt_finetune.sh. Reload example (inside the same container):
python predict.py --mode predict --bundle /path/to/epoch_001 \
    --input /path/to/demo_test.h5ad --out /path/to/new_prediction_results
No package installation or test-set training is performed.
"""
import argparse
import os
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
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3
CORE = ('encoder.', 'value_encoder.', 'transformer_encoder.')
TRAINABLE = CORE + ('cls_decoder.',)
MEMORY_NOTES = ('CUDA peaks are process-local PyTorch allocator counters, not total device usage. '
                'RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: '
                'do not add RSS and CUDA as independent physical memory pools.')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


INITIAL_SWAP_BYTES = psutil.swap_memory().used

class ResourceSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        available=psutil.virtual_memory().available
        swap=psutil.swap_memory().used
        if available<8*GIB or swap>INITIAL_SWAP_BYTES+2*GIB:
            guard={'time_unix':time.time(),'available_gib':available/GIB,'swap_gib':swap/GIB,'cuda_allocated_gib':torch.cuda.memory_allocated()/GIB,'cuda_reserved_gib':torch.cuda.memory_reserved()/GIB}
            write_json(Path(os.environ['SCGPT_GUARD_DIR'])/'memory_guard.json',guard)
            print('[MEMORY_GUARD]',guard,flush=True)
            os._exit(86)
        with self.lock:
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'gpu_util_percent': float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]),
                'process_rss_bytes': self.process.memory_info().rss,
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': psutil.virtual_memory().available,
                'system_swap_used_bytes': psutil.swap_memory().used,
            })

    def loop(self):
        while not self.event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()
        self.sample()

    def metrics(self):
        return {
            'gpu_util_mean_percent': float(np.mean([r['gpu_util_percent'] for r in self.rows])),
            'gpu_util_max_percent': max(r['gpu_util_percent'] for r in self.rows),
            'gpu_util_samples': len(self.rows),
            'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
            'gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / GIB,
            'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in self.rows) / GIB,
        }


def preserve_environment(out):
    source = out / 'source_snapshot'
    source.mkdir(exist_ok=True)
    for name, obj in [('model.py', TransformerModel), ('data_collator.py', DataCollator),
                      ('preprocess.py', binning), ('gene_tokenizer.py', GeneVocab)]:
        shutil.copy2(inspect.getfile(obj), source / name)
    packages = {p: metadata.version(p) for p in
                ['torch', 'scgpt', 'numpy', 'anndata', 'scanpy', 'scipy', 'pandas', 'scikit-learn', 'psutil']}
    direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
    device = torch.cuda.get_device_properties(0)
    write_json(out / 'environment.json', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'architecture': platform.machine(), 'packages': packages,
        'scgpt_install_source': json.loads(direct_url) if direct_url else None,
        'cuda_version': torch.version.cuda, 'gpu_name': device.name,
        'gpu_compute_capability': [device.major, device.minor],
        'cuda_reported_total_memory_gib': device.total_memory / GIB,
        'system_total_memory_gib': psutil.virtual_memory().total / GIB,
        'torch_cpu_threads': torch.get_num_threads(),
    })
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (out / 'requirements_freeze.txt').write_text(freeze.stdout)


def choose_training_subset(pool, y, requested, seed):
    """Proportional subset with at least one cell from every training class."""
    if requested == 0 or requested == len(pool):
        return np.sort(pool)
    classes, counts = np.unique(y[pool], return_counts=True)
    if requested < len(classes) or requested > len(pool):
        raise ValueError(f'n-cells must be 0 (all) or {len(classes)}..{len(pool)}')
    ideal = counts * (requested / len(pool))
    quota = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while quota.sum() < requested:
        available = np.flatnonzero(quota < counts)
        k = available[np.argmax((ideal - quota)[available])]
        quota[k] += 1
    while quota.sum() > requested:
        available = np.flatnonzero(quota > 1)
        k = available[np.argmax((quota - ideal)[available])]
        quota[k] -= 1
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(pool[y[pool] == c], int(q), replace=False)
                               for c, q in zip(classes, quota)])
    return np.sort(selected)


def extract_matrix(adata, genes):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell IDs and gene names must be unique')
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f'Missing {len(missing)} training genes, examples: {sorted(missing)[:5]}')
    selected = adata[:, genes]
    matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    matrix = matrix.astype(np.float32, copy=True)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Expression values must be finite and nonnegative')
    if np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise ValueError('A cell has no nonzero expression after vocabulary filtering')
    return matrix


class Tokenizer:
    def __init__(self, matrix, gene_names, vocab, config):
        self.matrix = matrix
        self.gene_ids = np.asarray(vocab(gene_names), dtype=np.int64)
        self.pad_id = vocab[config['pad_token']]
        self.cls_id = vocab['<cls>']
        self.pad_value = config['pad_value']
        self.max_length = config['max_length_including_cls']
        self.collator = DataCollator(do_padding=True, pad_token_id=self.pad_id,
            pad_value=self.pad_value, do_mlm=False, do_binning=True,
            max_length=self.max_length, sampling=True, keep_first_n_tokens=1)

    def batch(self, rows):
        examples = []
        for i in rows:
            nz = np.flatnonzero(self.matrix[i])
            genes = np.concatenate([np.array([self.cls_id], dtype=np.int64), self.gene_ids[nz]])
            values = np.concatenate([np.array([self.pad_value], dtype=np.float32), self.matrix[i, nz]])
            examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
        return self.collator(examples)

    def empty_recording(self, n):
        return (np.full((n, self.max_length), self.pad_id, dtype=np.int64),
                np.full((n, self.max_length), self.pad_value, dtype=np.float32))

    def cache(self, rows, batch_size):
        genes, values = self.empty_recording(len(rows))
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch = self.batch(rows[start:end])
            length = batch['gene'].shape[1]
            genes[start:end, :length] = batch['gene'].numpy()
            values[start:end, :length] = batch['expr'].numpy()
        return genes, values


def cached_batch(cache, start, end, pad_id):
    genes, values = cache
    length = int(np.count_nonzero(genes[start:end] != pad_id, axis=1).max())
    return {'gene': torch.from_numpy(genes[start:end, :length]),
            'expr': torch.from_numpy(values[start:end, :length])}


def classify(model, batch, pad_id):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    # Official classification path uses raw CLS, not L2-normalized embeddings.
    return model.cls_decoder(encoded[:, 0, :])


def load_pretrained_backbone(model, checkpoint, out):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError('Expected a plain tensor state_dict')
    converted = {}
    for key, value in state.items():
        for pattern, replacement in [
            (r'self_attn\._impl\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\._impl\.out_proj\.', 'self_attn.out_proj.'),
        ]:
            key = re.sub(pattern, replacement, key)
        if key in converted:
            raise ValueError(f'Duplicate renamed checkpoint key: {key}')
        converted[key] = value
    current = model.state_dict()
    required = [k for k in current if k.startswith(CORE)]
    loaded = {k: converted[k] for k in required
              if k in converted and converted[k].shape == current[k].shape}
    missing = sorted(set(required) - set(loaded))
    write_json(out / 'weight_loading.json', {
        'required_backbone_keys': len(required), 'loaded_backbone_keys': len(loaded),
        'missing_backbone_keys': missing, 'loaded_keys': sorted(loaded),
        'classifier': 'new random initialization; pretrained classifier not used',
        'unused_checkpoint_keys': sorted(set(converted) - set(loaded)),
    })
    if missing:
        raise RuntimeError(f'Incomplete pretrained backbone: {missing[:10]}')
    model.load_state_dict(loaded, strict=False)


def evaluate(model, cache, y, cell_ids, labels, pad_id, batch_size, out):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    logits_all = np.empty((len(y), len(labels)), dtype=np.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampler = ResourceSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(y), batch_size):
                end = min(start + batch_size, len(y))
                batch = cached_batch(cache, start, end, pad_id)
                logits_all[start:end] = classify(model, batch, pad_id).cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    finally:
        sampler.stop()
        pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)
    if not np.isfinite(logits_all).all():
        raise FloatingPointError('Non-finite prediction logits')
    probabilities = torch.from_numpy(logits_all).softmax(dim=1).numpy()
    predicted = logits_all.argmax(axis=1)
    known = y >= 0
    metrics = {
        'cells_total': len(y), 'cells_evaluated_known_labels': int(known.sum()),
        'cells_without_known_training_label': int((~known).sum()),
        'forward_seconds': seconds, 'cells_per_second': len(y) / seconds,
        'timing_scope': 'cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving',
        **sampler.metrics(),
    }
    if known.any():
        true, pred = y[known], predicted[known]
        metrics.update({
            'loss': float(torch.nn.functional.cross_entropy(torch.from_numpy(logits_all[known]),
                                                           torch.from_numpy(true)).item()),
            'accuracy': float(accuracy_score(true, pred)),
            'macro_f1_all_training_classes': float(f1_score(true, pred, labels=np.arange(len(labels)), average='macro', zero_division=0)),
            'macro_f1_present_true_classes': float(f1_score(true, pred, labels=np.unique(true), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(true, pred, average='weighted', zero_division=0)),
        })
        write_json(out / 'classification_report.json', classification_report(
            true, pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0))
        pd.DataFrame(confusion_matrix(true, pred, labels=np.arange(len(labels))),
                     index=labels, columns=labels).rename_axis('true_label').to_csv(out / 'confusion_matrix.csv')
    pd.DataFrame({
        'cell_id': cell_ids, 'true_label_id': y,
        'true_label': [labels[i] if i >= 0 else '<unknown_or_unlabeled>' for i in y],
        'predicted_label_id': predicted, 'predicted_label': [labels[i] for i in predicted],
        'max_probability': probabilities.max(axis=1), 'included_in_metrics': known,
    }).to_csv(out / 'predictions.tsv', sep='\t', index=False)
    np.savez_compressed(out / 'logits_probabilities.npz', logits=logits_all, probabilities=probabilities)
    write_json(out / 'metrics.json', metrics)
    model.cur_gene_token_embs = None
    return metrics, logits_all


def rng_state():
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'algorithm': np_state[0], 'keys': np_state[1].tolist(),
                  'position': int(np_state[2]), 'has_gauss': int(np_state[3]), 'cached_gaussian': float(np_state[4])},
        'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
    }


def train(opt):
    out = opt.out
    pretrained = ROOT / 'models/whole-human'
    args = json.loads((pretrained / 'args.json').read_text())
    if (args.get('n_bins') != 51 or args.get('input_style') != 'binned'
            or args.get('input_emb_style', 'continuous') != 'continuous' or args.get('pre_norm', False)):
        raise ValueError('Requires the inspected 51-bin continuous-value post-norm checkpoint')
    vocab = GeneVocab.from_file(pretrained / 'vocab.json')
    for token in [args['pad_token'], '<cls>', '<eoc>']:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[args['pad_token']])
    input_path = ROOT / 'data/pancreas_demo/demo_train.h5ad'
    t0 = time.perf_counter()
    data = ad.read_h5ad(input_path)
    if data.shape != (10600, 3000) or 'Celltype' not in data.obs or data.obs['Celltype'].isna().any():
        raise ValueError('Input differs from the inspected 10600 x 3000 labeled training file')
    label_values = data.obs['Celltype'].astype(str).to_numpy()
    labels = sorted(set(label_values))
    if len(labels) != 13:
        raise ValueError(f'Expected 13 training classes; found {len(labels)}')
    label_to_id = {name: i for i, name in enumerate(labels)}
    y = np.asarray([label_to_id[x] for x in label_values], dtype=np.int64)
    all_rows = np.arange(data.n_obs)
    pool, validation = train_test_split(all_rows, test_size=opt.validation_fraction,
                                        random_state=opt.seed, stratify=y)
    validation = np.sort(validation)
    training = choose_training_subset(pool, y, opt.n_cells, opt.seed)
    if len(np.intersect1d(training, validation)):
        raise RuntimeError('Training and validation overlap')
    if set(y[training]) != set(range(len(labels))) or set(y[validation]) != set(range(len(labels))):
        raise ValueError('Every class must be represented in training and validation')
    genes = [str(g) for g in data.var_names if g in vocab]
    if len(genes) != 2999:
        raise ValueError(f'Expected 2999 vocabulary-matched genes, found {len(genes)}')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    roles = np.full(len(y), 'train_pool_unused', dtype=object)
    roles[validation] = 'validation'
    roles[training] = 'train_used'
    pd.DataFrame({'original_row_index': all_rows, 'cell_id': cell_ids,
                  'Celltype': label_values, 'label_id': y, 'role': roles}).to_csv(
                      out / 'split_assignments.tsv', sep='\t', index=False)
    pd.crosstab(pd.Series(label_values, name='Celltype'), pd.Series(roles, name='role')).to_csv(
        out / 'split_class_counts.csv')
    input_seconds = time.perf_counter() - t0
    del data
    preprocess = {
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'], 'n_bins': 51,
        'max_length_including_cls': opt.max_length,
        'additional_normalize_total': False, 'additional_log1p': False,
        'expression_provenance': 'Supplied h5ad values; no raw layer or preprocessing metadata',
        'tokenization': 'nonzero genes only; prepend CLS; official per-cell quantile binning; random overlength sampling; dynamic padding',
        'pooling': 'raw CLS features into official cls_decoder; no L2 normalization',
        'prediction_genes': 'require every saved training gene; reorder to training gene order; ignore additional genes',
    }
    constructor = {
        'ntoken': len(vocab), 'd_model': args['embsize'], 'nhead': args['nheads'],
        'd_hid': args['d_hid'], 'nlayers': args['nlayers'],
        'nlayers_cls': args['n_layers_cls'], 'n_cls': len(labels), 'dropout': args['dropout'],
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'],
        'do_mvc': False, 'do_dab': False, 'use_batch_labels': False,
        'domain_spec_batchnorm': False, 'input_emb_style': 'continuous', 'cell_emb_style': 'cls',
        'explicit_zero_prob': False, 'use_fast_transformer': False, 'pre_norm': False,
    }
    config = {
        'task': 'supervised cell-type classification', 'seed': opt.seed,
        'train_source_cells': len(y), 'training_pool_cells': len(pool), 'training_cells': len(training),
        'validation_cells': len(validation), 'classes': len(labels), 'epochs': opt.epochs,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'autocast': False, 'tf32': False,
        'optimizer': {'name': 'AdamW', 'lr': opt.lr, 'weight_decay': opt.weight_decay,
                      'betas': [0.9, 0.999], 'eps': 1e-8},
        'scheduler': None, 'gradient_clip_max_norm': 1.0,
        'loss': 'unweighted cross entropy', 'trainable_prefixes': list(TRAINABLE),
        'unused_expression_decoder': 'frozen and bypassed',
        'classifier': 'new official ClsDecoder; all classifier layers trainable',
        'split': 'stratified by Celltype before subsampling; no donor/batch metadata available',
        'validation_limit': 'within-file cell split; not donor-independent validation',
        'validation_tokens': 'generated once with seed+10000, saved and reused',
        'epoch_seed': 'seed+epoch; reinitializes RNG before each epoch',
        'checkpoint_selection': 'highest validation macro-F1 over all 13 training classes',
        'test_used_for_training_or_selection': False,
        'training_warmup_steps': 0,
        'training_timing_scope': 'tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization',
        'training_timer_excludes': 'imports, h5ad/model loading, validation token cache, validation, disk saving, reload verification',
        'memory_notes': MEMORY_NOTES, 'preprocessing': preprocess,
        'reproducibility': 'Seeds, exact tokens and environment saved; bitwise reproducibility across platforms is not guaranteed',
    }
    for filename, obj in [('run_config.json', config), ('preprocessing.json', preprocess),
                          ('model_config.json', constructor), ('labels.json', {'labels': labels, 'label_to_id': label_to_id}),
                          ('genes.json', genes), ('vocab.json', vocab.get_stoi())]:
        write_json(out / filename, obj)
    shutil.copy2(pretrained / 'args.json', out / 'pretrained_args.json')
    write_json(out / 'provenance.json', {
        'input': {'path': str(input_path), 'sha256': sha256(input_path)},
        'pretrained_checkpoint': {'path': str(pretrained / 'best_model.pt'), 'sha256': sha256(pretrained / 'best_model.pt')},
        'script_sha256': sha256(__file__),
        'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html',
        'dataset_source': 'TOSICA author-distributed Figshare demo_train.h5ad (file 39009503)',
    })
    print(f'[Split] train={len(training)}, validation={len(validation)}, classes={len(labels)}, matched_genes={len(genes)}', flush=True)
    print('[Input] expression values as supplied -> 51 bins; no additional normalization/log1p', flush=True)
    t0 = time.perf_counter()
    seed_all(opt.seed)
    model = TransformerModel(**constructor, vocab=vocab)
    load_pretrained_backbone(model, pretrained / 'best_model.pt', out)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    model.float().to('cuda')
    torch.cuda.synchronize()
    write_json(out/'model_loading_baseline.json', {'allocated_gib':torch.cuda.memory_allocated()/GIB,'reserved_gib':torch.cuda.memory_reserved()/GIB,'rss_gib':psutil.Process().memory_info().rss/GIB,'system_available_gib':psutil.virtual_memory().available/GIB})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=opt.lr, weight_decay=opt.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8)
    torch.cuda.synchronize()
    model_loading_seconds = time.perf_counter() - t0
    probes = ['transformer_encoder.layers.0.self_attn.in_proj_weight', 'cls_decoder.out_layer.weight']
    initial = {k: model.state_dict()[k].detach().cpu().clone() for k in probes}
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed + 10000)
    t0 = time.perf_counter()
    val_cache = tokenizer.cache(validation, opt.batch_size)
    cache_seconds = time.perf_counter() - t0
    np.savez_compressed(out / 'validation_tokenized_inputs.npz', gene_ids=val_cache[0],
                        expression_bins=val_cache[1], original_row_indices=validation, labels=y[validation])
    print('[Validation] fixed input tokens saved', flush=True)
    opt.model_out.mkdir(parents=True, exist_ok=False)
    epoch_summaries = []
    best = None
    for epoch in range(1, opt.epochs + 1):
        epoch_out = out / f'epoch_{epoch:03d}'
        epoch_out.mkdir()
        seed_all(opt.seed + epoch)
        order = np.random.default_rng(opt.seed + epoch).permutation(training)
        saved_genes, saved_values = tokenizer.empty_recording(len(order))
        steps = []
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        loss_sum, correct = 0.0, 0
        try:
            for step, start in enumerate(range(0, len(order), opt.batch_size), 1):
                end = min(start + opt.batch_size, len(order))
                rows = order[start:end]
                batch = tokenizer.batch(rows)
                length = batch['gene'].shape[1]
                saved_genes[start:end, :length] = batch['gene'].numpy()
                saved_values[start:end, :length] = batch['expr'].numpy()
                target = torch.from_numpy(y[rows]).to('cuda')
                optimizer.zero_grad(set_to_none=True)
                logits = classify(model, batch, tokenizer.pad_id)
                loss = torch.nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, step {step}')
                loss.backward()
                if epoch == 1 and step == 1:
                    missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing_grad:
                        raise RuntimeError(f'Trainable parameters without gradients: {missing_grad}')
                    print(f'[Model input] gene/value={tuple(batch["gene"].shape)}, labels={tuple(target.shape)}, logits={tuple(logits.shape)}', flush=True)
                    print('[Backward] all trainable parameters received gradients', flush=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach().item())
                batch_correct = int((logits.detach().argmax(dim=1) == target).sum().item())
                loss_sum += loss_value * len(rows)
                correct += batch_correct
                steps.append({'epoch': epoch, 'step': step, 'cells': len(rows), 'padded_length': length,
                              'loss': loss_value, 'correct': batch_correct, 'grad_norm_before_clip': float(grad_norm.item())})
                if step % 25 == 0 or end == len(order):
                    print(f'[Train {epoch}/{opt.epochs}] {end}/{len(order)} cells, online_loss={loss_sum/end:.4f}', flush=True)
                del logits, loss, target
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - t0
        finally:
            sampler.stop()
            pd.DataFrame(sampler.rows).to_csv(epoch_out / 'training_resources.csv', index=False)
        training_metrics = sampler.metrics()
        optimizer.zero_grad(set_to_none=True)
        model.cur_gene_token_embs = None
        pd.DataFrame(steps).to_csv(epoch_out / 'training_steps.csv', index=False)
        np.savez_compressed(epoch_out / 'training_tokenized_inputs.npz', gene_ids=saved_genes,
                            expression_bins=saved_values, original_row_indices=order, labels=y[order])
        del saved_genes, saved_values
        deltas = {k: float((model.state_dict()[k].detach().cpu() - initial[k]).abs().max().item()) for k in probes}
        if not all(np.isfinite(v) and v > 0 for v in deltas.values()):
            raise RuntimeError(f'Expected encoder and classifier weights to change: {deltas}')
        validation_metrics, val_logits = evaluate(model, val_cache, y[validation], cell_ids[validation],
            labels, tokenizer.pad_id, opt.batch_size, epoch_out / 'validation')
        bundle = opt.model_out / f'epoch_{epoch:03d}'
        bundle.mkdir()
        save_start = time.perf_counter()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, bundle / 'model.pt')
        torch.save({'epoch_completed': epoch, 'optimizer': optimizer.state_dict(),
                    'rng_state': rng_state(), 'run_config': config}, bundle / 'training_state.pt')
        for filename in ['model_config.json', 'preprocessing.json', 'labels.json', 'genes.json', 'vocab.json',
                         'run_config.json', 'pretrained_args.json', 'environment.json', 'requirements_freeze.txt']:
            shutil.copy2(out / filename, bundle / filename)
        shutil.copy2(__file__, bundle / 'predict.py')
        (bundle / 'README.txt').write_text(
            'Saved scGPT encoder + trained cell-type classifier. model.pt is a plain state_dict.\n'
            'Run prediction inside the scgpt-spark container using its existing environment:\n'
            'python /path/to/epoch_001/predict.py --mode predict --bundle /path/to/epoch_001 '
            '--input /workspace/scgpt/data/pancreas_demo/demo_test.h5ad --out /workspace/scgpt/results/NEW_NAME\n'
            'All input cells are retained. Unknown labels are excluded from closed-set metrics and reported separately.\n'
            'No unknown-class detector is trained; softmax confidence is not an unknown detector.\n'
            'training_state.pt stores optimizer and RNG states for recovery/development; this script has no resume CLI.\n'
            'Validation is a cell split within the supplied train file, not a donor-independent benchmark.\n')
        save_seconds = time.perf_counter() - save_start
        # Verify strict reload and identical predictions from the saved weights.
        reloaded = TransformerModel(**constructor, vocab=vocab)
        reloaded.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        reloaded.float().to('cuda').eval()
        end = min(opt.batch_size, len(validation))
        with torch.inference_mode():
            reload_logits = classify(reloaded, cached_batch(val_cache, 0, end, tokenizer.pad_id), tokenizer.pad_id).cpu().numpy()
        reload_error = float(np.max(np.abs(reload_logits - val_logits[:end])))
        if not np.allclose(reload_logits, val_logits[:end], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Saved-model reload mismatch: max error {reload_error}')
        del reloaded
        gc.collect()
        write_json(bundle / 'manifest.json', {
            'model_sha256': sha256(bundle / 'model.pt'),
            'training_state_sha256': sha256(bundle / 'training_state.pt'),
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
            'results_directory': str(out),
        })
        record = {
            'epoch': epoch, 'training_cells': len(training), 'training_seconds': training_seconds,
            'training_cells_per_second': len(training) / training_seconds,
            'online_training_loss': loss_sum / len(training), 'online_training_accuracy': correct / len(training),
            **training_metrics, 'validation': validation_metrics, 'weight_probe_max_abs_deltas': deltas,
            'model_bundle': str(bundle), 'checkpoint_save_seconds': save_seconds,
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
        }
        write_json(epoch_out / 'summary.json', record)
        epoch_summaries.append(record)
        score = validation_metrics['macro_f1_all_training_classes']
        if best is None or score > best['validation_macro_f1']:
            best = {'epoch': epoch, 'validation_macro_f1': score, 'model_bundle': str(bundle)}
            write_json(opt.model_out / 'best.json', best)
        print(f'[Validation] accuracy={validation_metrics["accuracy"]:.4f}, macro-F1={score:.4f}', flush=True)
        print(f'[Saved] {bundle}; reload check passed', flush=True)
    summary = {
        'status': 'complete', 'task': 'scGPT supervised cell-type fine-tuning',
        'training_cells': len(training), 'validation_cells': len(validation), 'classes': len(labels),
        'matched_genes': len(genes), 'epochs': opt.epochs, 'batch_size': opt.batch_size,
        'max_length_including_cls': opt.max_length, 'precision': 'FP32', 'seed': opt.seed,
        'parameter_count_constructed_model': sum(p.numel() for p in model.parameters()),
        'parameter_count_trainable': sum(p.numel() for p in trainable),
        'parameter_count_encoder': sum(p.numel() for name, p in model.named_parameters() if name.startswith(CORE)),
        'input_preparation_seconds': input_seconds, 'model_loading_seconds': model_loading_seconds,
        'validation_tokenization_seconds': cache_seconds,
        'training_seconds_total': sum(e['training_seconds'] for e in epoch_summaries),
        'training_gpu_peak_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in epoch_summaries),
        'training_gpu_peak_reserved_gib': max(e['gpu_peak_reserved_gib'] for e in epoch_summaries),
        'training_process_rss_peak_observed_gib': max(e['python_process_rss_peak_observed_gib'] for e in epoch_summaries),
        'best_checkpoint': best, 'validation': epoch_summaries[best['epoch'] - 1]['validation'],
        'weights_updated': True, 'encoder_and_classifier_updated': True, 'saved_model_reload_verified': True,
        'test_evaluated': False, 'epochs_detail': epoch_summaries,
        'timing_scope': config['training_timing_scope'], 'memory_notes': MEMORY_NOTES,
        'interpretation': 'Small-run fine-tuning and validation check; not a full-data or donor-independent accuracy benchmark',
    }
    write_json(out / 'summary.json', summary)
    print('\n' + json.dumps({k: summary[k] for k in [
        'status', 'training_cells', 'validation_cells', 'epochs', 'training_seconds_total',
        'training_gpu_peak_allocated_gib', 'training_process_rss_peak_observed_gib',
        'best_checkpoint', 'saved_model_reload_verified', 'test_evaluated']}, indent=2), flush=True)
    print('SCGPT_FINETUNE_COMPLETE', flush=True)


def predict(opt):
    if opt.bundle is None or opt.input is None:
        raise ValueError('Prediction requires --bundle and --input')
    bundle = opt.bundle
    constructor = json.loads((bundle / 'model_config.json').read_text())
    preprocess = json.loads((bundle / 'preprocessing.json').read_text())
    labels = json.loads((bundle / 'labels.json').read_text())['labels']
    label_to_id = {name: i for i, name in enumerate(labels)}
    genes = json.loads((bundle / 'genes.json').read_text())
    vocab = GeneVocab.from_file(bundle / 'vocab.json')
    vocab.set_default_index(vocab[preprocess['pad_token']])
    data = ad.read_h5ad(opt.input)
    if data.n_obs == 0:
        raise ValueError('Input has no cells')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    originals = (data.obs['Celltype'].astype('string').fillna('<unlabeled>').astype(str).to_numpy()
                 if 'Celltype' in data.obs else np.full(data.n_obs, '<unlabeled>'))
    y = np.asarray([label_to_id.get(label, -1) for label in originals], dtype=np.int64)
    pd.DataFrame({'cell_id': cell_ids, 'original_label': originals, 'known_training_label': y >= 0}).to_csv(
        opt.out / 'input_cells.tsv', sep='\t', index=False)
    write_json(opt.out / 'prediction_config.json', {
        'bundle': str(bundle), 'model_sha256': sha256(bundle / 'model.pt'),
        'input': str(opt.input), 'input_sha256': sha256(opt.input), 'seed': opt.seed,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'preprocessing': preprocess,
        'unknown_label_counts': pd.Series(originals[y < 0]).value_counts().astype(int).to_dict(),
        'unknown_policy': 'retain all predictions; evaluate known ground-truth labels only; no unknown-class detector',
    })
    model = TransformerModel(**constructor, vocab=vocab)
    model.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    model.float().to('cuda').eval()
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed)
    t0 = time.perf_counter()
    cache = tokenizer.cache(np.arange(len(y)), opt.batch_size)
    tokenization_seconds = time.perf_counter() - t0
    np.savez_compressed(opt.out / 'tokenized_inputs.npz', gene_ids=cache[0], expression_bins=cache[1], labels=y)
    metrics, _ = evaluate(model, cache, y, cell_ids, labels, tokenizer.pad_id, opt.batch_size, opt.out)
    summary = {'status': 'complete', 'task': 'fine-tuned cell-type predictions',
               'tokenization_seconds': tokenization_seconds, 'warmup_steps': 0,
               'weights_updated': False, 'model_bundle': str(bundle), 'memory_notes': MEMORY_NOTES, **metrics}
    write_json(opt.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print('SCGPT_PREDICTION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-out', type=Path)
    parser.add_argument('--n-cells', type=int, default=1000, help='training subset size; 0=all training-pool cells')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1200)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--input', type=Path)
    opt = parser.parse_args()
    if (opt.batch_size < 1 or opt.epochs < 1 or opt.n_cells < 0 or opt.max_length < 2
            or not 0 < opt.validation_fraction < 1 or not np.isfinite(opt.lr) or opt.lr <= 0
            or not np.isfinite(opt.weight_decay) or opt.weight_decay < 0 or not 0 <= opt.seed < 2**32 - 10000):
        parser.error('Invalid numeric option')
    if opt.mode == 'train' and opt.model_out is None:
        parser.error('Training requires --model-out')
    opt.out.mkdir(parents=True, exist_ok=True)
    if (opt.out / 'run_config.json').exists() or (opt.out / 'summary.json').exists() or (opt.out / 'prediction_config.json').exists():
        raise FileExistsError(f'Use a new output directory: {opt.out}')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; stopping instead of running on CPU')
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(opt.seed)
        preserve_environment(opt.out)
        train(opt) if opt.mode == 'train' else predict(opt)
    except Exception as exc:
        write_json(opt.out / 'failure.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
