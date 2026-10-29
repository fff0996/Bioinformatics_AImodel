#!/usr/bin/env bash
# scGPT Spark benchmark, step 00: download and inspect inputs; no training/inference.
# Run as your normal Spark user: bash scgpt_prepare_inputs.sh
set -euo pipefail

bench_root="$HOME/ai-benchmark"
scgpt_root="$bench_root/scgpt"
run_id="input_check_$(date +%Y%m%d_%H%M%S)_$$"
container="scgpt-spark"

# Verify that scGPT outputs will persist on the host.
sudo -v
state=$(sudo docker inspect "$container" --format '{{.State.Status}}')
if [[ "$state" != running ]]; then
    echo "ERROR: $container is not running. Start it with: sudo docker start $container" >&2
    exit 1
fi
mount_source=$(sudo docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/workspace/scgpt"}}{{.Source}}{{end}}{{end}}')
if [[ -z "$mount_source" || "$(readlink -f "$mount_source")" != "$(readlink -f "$scgpt_root")" ]]; then
    echo "ERROR: expected host mount $scgpt_root -> /workspace/scgpt; found: $mount_source" >&2
    exit 1
fi

mkdir -p "$bench_root/scripts" \
    "$scgpt_root/scripts" "$scgpt_root/results/$run_id"
saved_runner="$bench_root/scripts/00_prepare_scgpt_inputs.sh"
if [[ "$(readlink -f "$0")" != "$(readlink -f "$saved_runner")" ]]; then
    cp -- "$0" "$saved_runner"
fi

cat > "$scgpt_root/scripts/00_prepare_inputs.py" <<'PY'
"""Download author-provided pancreas data and inspect it without preprocessing.

No expression values, labels, gene names or train/test assignments are changed.
"""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import requests
from scipy import sparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

root, output = map(Path, sys.argv[1:3])
data = root / "data/pancreas_demo"
data.mkdir(parents=True, exist_ok=True)
model = root / "models/whole-human"
vocab = json.loads((model / "vocab.json").read_text())
args = json.loads((model / "args.json").read_text())
if not (model / "best_model.pt").is_file():
    raise FileNotFoundError(model / "best_model.pt")

# TOSICA authors' Figshare releases; do not claim byte identity with scGPT's Drive copy.
sources = [
    ("demo_train.h5ad", "39009503", "642c52f7556acd6e6f60e00fc52c4509", "21980552"),
    ("demo_test.h5ad", "39009500", "5083cd8fafe9d6f4c389d9cde43ce7f2", "21980549"),
]
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"], respect_retry_after_header=True)))

def digests(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()

report = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "input inspection only; no preprocessing, inference or training",
    "source": "Author-provided TOSICA pancreas demo files on Figshare",
    "files": [],
    "model_config": {key: args.get(key) for key in [
        "embsize", "nheads", "d_hid", "nlayers", "n_bins", "max_seq_len", "input_style"]},
}
labels_by_file, genes_by_file = {}, {}
for name, file_id, expected_md5, article_id in sources:
    path = data / name
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    if not path.exists():
        print("Downloading:", name, flush=True)
        temp = path.with_suffix(".h5ad.part")
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        actual_md5, _ = digests(temp)
        if actual_md5 != expected_md5:
            raise ValueError(f"Download checksum mismatch: {temp}; retry the script")
        temp.replace(path)
    actual_md5, sha256 = digests(path)
    if actual_md5 != expected_md5:
        raise ValueError(f"Existing file differs from the specified release: {path}. It was not overwritten.")

    a = ad.read_h5ad(path)
    if a.X is None or a.n_obs == 0 or a.n_vars == 0:
        raise ValueError(f"No expression matrix in {path}")
    # Small dense slices are used only for display/statistics; the original is unchanged.
    block = a.X[:5, :5]
    block = block.toarray() if sparse.issparse(block) else np.asarray(block)
    sample = a.X[:min(256, a.n_obs), :]
    sample = sample.toarray() if sparse.issparse(sample) else np.asarray(sample)
    finite = np.isfinite(sample)
    nonzero = sample[finite & (sample != 0)]
    finite_values = sample[finite]
    celltypes = a.obs["Celltype"] if "Celltype" in a.obs else None
    label_counts = {} if celltypes is None else {
        str(key): int(value) for key, value in celltypes.dropna().value_counts().items() if value > 0}
    gene_match = int(sum(str(gene) in vocab for gene in a.var_names))
    var_matches = {}
    for column in a.var.columns:
        if a.var[column].dtype.kind in "OUS" or str(a.var[column].dtype) == "category":
            var_matches[str(column)] = int(a.var[column].astype(str).isin(vocab).sum())
    labels_by_file[name] = set(label_counts)
    genes_by_file[name] = a.var_names.tolist()
    item = {
        "filename": name, "source_url": url,
        "article_url": f"https://figshare.com/articles/dataset/{path.stem}_h5ad/{article_id}",
        "size_bytes": path.stat().st_size, "md5": actual_md5, "sha256": sha256,
        "shape_cells_by_genes": list(a.shape), "x_type": type(a.X).__name__,
        "x_dtype": str(a.X.dtype), "obs_columns": a.obs.columns.tolist(),
        "var_columns": a.var.columns.tolist(), "layers": list(a.layers.keys()),
        "raw_shape": None if a.raw is None else list(a.raw.shape),
        "uns_keys": list(a.uns.keys()), "has_log1p_metadata": "log1p" in a.uns,
        "gene_names_first_10": a.var_names[:10].tolist(),
        "gene_names_unique": bool(a.var_names.is_unique),
        "cell_ids_unique": bool(a.obs_names.is_unique),
        "vocab_matches_var_names": gene_match, "vocab_matches_var_columns": var_matches,
        "label_column": None if celltypes is None else "Celltype",
        "label_counts": label_counts,
        "missing_labels": None if celltypes is None else int(celltypes.isna().sum()),
        "value_sample": {
            "shape": list(sample.shape), "nonfinite_count": int((~finite).sum()),
            "min": None if finite_values.size == 0 else float(finite_values.min()),
            "max": None if finite_values.size == 0 else float(finite_values.max()),
            "zero_fraction": float(np.mean(sample == 0)),
            "nonzero_integer_like_fraction": None if nonzero.size == 0 else float(
                np.mean(np.isclose(nonzero, np.rint(nonzero), atol=1e-6, rtol=0))),
            "note": "Sample statistics alone do not establish raw-count or normalization provenance.",
        },
    }
    report["files"].append(item)
    a.obs.head(10).to_csv(output / f"{path.stem}_obs_head.tsv", sep="\t")
    a.var.head(10).to_csv(output / f"{path.stem}_var_head.tsv", sep="\t")
    import pandas as pd
    block_frame = pd.DataFrame(block, index=a.obs_names[:5], columns=a.var_names[:5])
    block_frame.to_csv(output / f"{path.stem}_expression_5x5.tsv", sep="\t")

    print("\n[scGPT input]", name)
    print("shape (cells x genes):", a.shape, "X:", item["x_type"], item["x_dtype"])
    print("obs columns:", item["obs_columns"], "var columns:", item["var_columns"])
    print("genes first 10:", item["gene_names_first_10"])
    print("vocab matches:", f"{gene_match}/{a.n_vars}", "var columns:", var_matches)
    print("layers:", item["layers"], "raw shape:", item["raw_shape"], "uns:", item["uns_keys"])
    print("expression 5 x 5:\n" + block_frame.to_string())
    print("Celltype counts:", json.dumps(label_counts, ensure_ascii=False))
    print("missing labels:", item["missing_labels"])
    print("expression sample:", json.dumps(item["value_sample"], ensure_ascii=False))
    del a

train, test = "demo_train.h5ad", "demo_test.h5ad"
report["cross_file_checks"] = {
    "identical_gene_order": genes_by_file[train] == genes_by_file[test],
    "shared_genes": len(set(genes_by_file[train]) & set(genes_by_file[test])),
    "test_only_labels": sorted(labels_by_file[test] - labels_by_file[train]),
    "train_only_labels": sorted(labels_by_file[train] - labels_by_file[test]),
    "note": "Test-only classes require an explicit evaluation policy before classification training.",
}
print("\n[Train/test checks]", json.dumps(report["cross_file_checks"], ensure_ascii=False))
print("[Model config]", json.dumps(report["model_config"]))
(output / "input_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
versions = {}
for package in ["torch", "scgpt", "anndata", "scanpy", "numpy", "scipy", "pandas", "requests"]:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
(output / "environment.json").write_text(json.dumps({
    "python": platform.python_version(), "machine": platform.machine(), "packages": versions,
}, indent=2) + "\n")
print("report:", output / "input_report.json")
print("INPUT_CHECK_COMPLETE")
PY

# Save exactly the scripts used for this inspection alongside its outputs.
cp -- "$scgpt_root/scripts/00_prepare_inputs.py" "$scgpt_root/results/$run_id/"
cp -- "$saved_runner" "$scgpt_root/results/$run_id/"


printf '%s\n' "sudo docker exec $container python -u /workspace/scgpt/scripts/00_prepare_inputs.py /workspace/scgpt /workspace/scgpt/results/$run_id" \
    > "$scgpt_root/results/$run_id/run_command.txt"
sudo docker inspect "$container" --format 'Image={{.Config.Image}} ImageID={{.Image}}' \
    > "$scgpt_root/results/$run_id/container_image.txt"
sudo docker exec "$container" python -u /workspace/scgpt/scripts/00_prepare_inputs.py \
    /workspace/scgpt "/workspace/scgpt/results/$run_id" \
    2>&1 | tee "$scgpt_root/results/$run_id/input_check.log"

printf 'Saved scGPT input report: %s\n' "$scgpt_root/results/$run_id/input_report.json"
printf 'Saved scGPT inspection log: %s\n' "$scgpt_root/results/$run_id/input_check.log"
printf 'Input preparation finished. Share the terminal output before the next step.\n'
