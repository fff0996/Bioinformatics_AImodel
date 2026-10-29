# 공통 실행·계측·보고 도구 원문

모델별 최종 보고서에도 필요한 runtime helper를 수록했다. 이 문서는 공통 orchestration 및 보고서 생성 도구를 원문 그대로 보존한다. 독립 스크립트 파일은 Git에 포함하지 않는다.

<details>
<summary>scripts/00_prepare_scgpt_inputs.sh — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/00_prepare_scgpt_inputs.sh", "sha256": "8b07ce164fc598f2319e368bb70c855640ffaf52f8a2f1fc3f5628ef38a1d7c9", "encoding": "utf-8", "added_final_newline": false} -->
```bash
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
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/af2_feature_capture_launch.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/af2_feature_capture_launch.py", "sha256": "2d62f809b99dd4b97b04835333d35d25b3eb86c98d7ee514a7df20f422b6e6da", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import subprocess,datetime,shlex,json
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'colabfold';run='vmap_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');print(run,flush=True);cmd=['docker','run','--rm','--network','none','--ipc=host','--user','1000:1000','-e','JAX_PLATFORMS=cpu','-e','RUN_ID='+run,'-e','MPLCONFIGDIR=/work/results/'+run+'/mplcache','-v',str(R)+':/work','-v',str(R/'cache/colabfold')+':/cache:ro','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','python','-u','/work/scripts/batch_followup/capture.py'];(D/'capture_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'capture.log').open('w') as f:c=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
(D/'capture_exit.json').write_text(json.dumps({'exit_code':c}));print('CAPTURE_EXIT',c,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/af2_vmap_launch.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/af2_vmap_launch.py", "sha256": "68c706f5a2fcb6da7c57452936689f1e79ca218d52e2020548915db603480c01", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,subprocess,time,shlex,sys
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'colabfold';run=(R/'results/BATCH_LATEST').read_text().strip();D=R/'results'/run;assert json.loads((D/'capture_exit.json').read_text())['exit_code']==0
clair=ROOT/'clair3/results'/(ROOT/'clair3/results/BATCH_LATEST').read_text().strip()
while json.loads((clair/'summary.json').read_text())['status']=='running':time.sleep(10)
S={'status':'running','run_id':run,'workers':[],'failed':[],'scope':'experimental JAX vmap independent monomer batch axis on existing ColabFold model_1, no simultaneous GPU jobs','settings':{'guard_available_gib':16,'guard_swap_increase_gib':2}};swap0=psutil.swap_memory().used
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
save();print('AF2_BATCH_START',run,flush=True)
for batch in [1,2,4]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'other GPU work'
 if psutil.virtual_memory().available<16*2**30:S['stop_reason']='available_memory_guard';break
 name='af2-vmap-'+run+'-'+str(batch);cmd=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--ipc=host','--user','1000:1000','-e','RUN_ID='+run,'-e','BATCH='+str(batch),'-e','JAX_PLATFORMS=cuda,cpu','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','-e','MPLCONFIGDIR=/work/results/'+run+'/mplcache','-v',str(R)+':/work','-v',str(R/'cache/colabfold')+':/cache:ro','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','python','-u','/work/scripts/batch_followup/worker.py'];(D/f'batch{batch}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();reason=None
 with (D/f'batch{batch}.log').open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/f'batch{batch}_exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'reason':reason}));print('AF2_BATCH_EXIT',batch,code,flush=True)
 if (D/f'batch_{batch}/summary.json').exists():S['workers'].append(json.loads((D/f'batch_{batch}/summary.json').read_text()))
 if code:S['failed'].append({'batch':batch,'exit_code':code});S['stop_reason']='worker_error_or_guard_see_log';break
 save()
else:S['stop_reason']='actual_four_query_pool_complete'
S['status']='complete' if not S['failed'] and len(S['workers'])==3 else 'stopped';save();subprocess.run([sys.executable,str(ROOT/'scripts/two_batch_report.py'),'af2',str(D)],check=True);print('AF2_FINISHED',S['status'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/build_followup_overview.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/build_followup_overview.py", "sha256": "758bc4fe7956cd039e5bf1577f6fa641306bb72bbbcf8e4ae279e1d23cf26425", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json
ROOT=Path('/home/hyein/ai-benchmark');RAM=121.68751525878906
read=lambda p:json.loads(p.read_text())
rows=[]
def add(name,task,batch,peak,util,limits,report):rows.append([name,task,str(batch),f'{peak:.3f}',f'{100*peak/RAM:.2f}',f'{util:.2f}' if util is not None else 'N/A',limits,f'[보고서]({report})'])
for name,title,report in [('esm2','ESM-2','ESM2_batch_memory.md'),('cellpose','Cellpose','Cellpose_batch_memory.md')]:
 r=ROOT/name/'results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');c=s['conditions'][-1];add(title,'추론',c.get('batch_size',c.get('actual_max_tile_batch')),c['memory']['peak_allocated_gib'],c['gpu_util_avg'],'다음 예약량 예측 guard; 실제 OOM 아님',name+'/reports/'+report)
s=read(ROOT/'scgpt/results/actual_300k_resume_20261028_093701_696230/whole_dataset_comparison.json')[-1];add('scGPT','실제30만세포 임베딩 추론',s['batch'],s['allocated_peak_gib_max_segments'],s['gpu_util_mean_sample_weighted'],'같은30만세포 batch8/256 비교','scgpt/reports/scGPT_actual_300k.md')
for name in ['scvi','dnabert2','ntv2','rnafm','openspliceai']:
 r=ROOT/name/'results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');assert s['status']=='complete';c=s['conditions'][-1];add(name,'추론',c['batch_size'],c['memory']['peak_allocated_gib'],c['gpu_util_avg'],s['stop_reason'],name+'/reports/Batch_memory_followup.md')
r=ROOT/'clair3/results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');assert s['status']=='complete'
for kind in ['Clair3_P','Clair3_F']:
 c=[c for c in s['conditions'] if c['model']==kind][-1];add('Clair3 '+('pileup' if kind.endswith('_P') else 'full-alignment'),'실제 후보 network 추론',c['batch_size'],c['memory']['peak_allocated_gib'],c['gpu_util_avg'],'확보한 고유feature pool 끝; 최대한계 아님','clair3/reports/Batch_memory_followup.md')
r=ROOT/'colabfold/results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');assert s['status']=='complete';w=s['workers'][-1];c=w['conditions'][0];add('AlphaFold2 — ColabFold','실험적 FP32 JAX vmap',c['batch'],c['jax_memory_stats']['peak_bytes_in_use']/2**30,c['gpu_util_avg'],'실제4query 끝; JAX peak는컴파일/warmup포함 누적값','colabfold/reports/Batch_memory_followup.md')
order={'ESM-2':0,'AlphaFold2 — ColabFold':1,'scvi':2,'Cellpose':3,'dnabert2':4,'ntv2':5,'Clair3 pileup':6,'Clair3 full-alignment':7,'rnafm':8,'openspliceai':9,'scGPT':10};rows.sort(key=lambda r:order[r[0]])
lines=['# 10종 후속 실험 최종 결과','','10종 모두 후속 실행과 각 범위의 출력 검증·저장·보고서를 완료했다. 모델별 후속 범위와 정밀도·길이·메모리 정의가 다르며 같은 과제/조건의 모델 간 성능 순위로 해석하지 않는다. 아래 배치는 **시험한 최대 성공 조건**이며 이론적 최대 지원 배치가 아니다.','','|모델|후속 작업|최대 시험 성공batch|메모리 peak GiB|공유RAM대비%|GPU평균%|종료/해석 제한|상세|','|---|---|---|---|---|---|---|---|']
lines+=['|'+ '|'.join(r)+'|' for r in rows]
lines+=['','메모리 비율의 분모는CPU/GPU공유RAM121.688GiB다. GPU 연산사용률과 메모리 비율은 다른 지표이고, 전용VRAM점유율이 아니며RSS와CUDA를더하지않는다. PyTorch행은CUDAallocated peak다. AF2는JAX프로세스누적allocator peak로컴파일/warmup을포함하므로실행구간peak와직접비교하지않는다. 예약reserved와sampled RSS/가용량도각보고서에별도로있다.','','## scGPT 별도 학습 용량 결과','','FP32/1200열고정padding 학습probe의최대성공batch292: allocated100.750GiB,전체공유RAM대비82.79%. 이는위표의추론batch256·5.477GiB와다른실험이다. 90%는관측하지못했다. [학습용량보고서](scgpt/reports/scGPT_memory_batch.md)','','## 마지막 두 모델의 구현과 검증','','Clair3는실제HG003 BAM에서전체candidate tensor를추출해pileup/full-alignment를따로배치확대했다. 원래native pipeline의동일입력/확률대조와저장·재로드를검증했다. 후보좌표를내보내지못해생물학적고유위치수가아닌중복제거feature수로보고한다. VCF호출전체시간과candidate network시간은구분한다.','','AlphaFold2는CLI기본배치가아닌JAX vmap 별도실험이다. 단량체model_1/115AA실제4query/기존MSA/3recycle을사용했다. BF16벡터화참조검증은구조차이로실패해보존했고,FP32고정별도조건의batch1/2/4로실행했다. batch1과native참조CA정렬RMSD0.1795Å/pLDDT최대차0.1875로사전0.5Å기준통과. 출력PDB·좌표·pLDDT·query수검증을수행했고batch별수치차이도보존했다. 기존기본BF16·5모델순차예측과동일조건으로표현하지않는다.','','기존미실시/중단/실패기록은인수인계이력으로남기며현재최신상태는이최종표를우선한다. 원본run/명령/로그/출력/환경/입력선택/자원CSV는각모델results와보고서에있다. 추가GPU모델작업은자동시작하지않는다.']
(ROOT/'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines)+'\n')
for name in ['RESULTS.md','MEMORY_RESULTS.md','STATUS.md']:
 with (ROOT/name).open('a') as f:f.write('\n10종 후속최종정리완료. 이전AlphaFold2/Clair3미실시상태는새실행으로갱신됐다. 구현·정밀도·검증범위차이를포함한최신전체표: [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md). 추가GPU작업없음.\n')
print('FOLLOWUP_OVERVIEW_COMPLETE')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/build_integrated_reports.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/build_integrated_reports.py", "sha256": "dc30c5da00f18621c7fd72f6717b8b85723755e00b5b767bcd0681bb195a0e6d", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import re,shutil,json
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication');RAM=121.68751525878906
MODELS=[
 ('esm2','ESM-2','단백질 아미노산 서열 → 단백질당1280차원 임베딩 또는 fluorescence 예측. 체크포인트 esm2_t33_650M_UR50D(명목650M).',['ESM2_20261027_164300_125163.md','ESM2_batch_memory.md'],[
 ('추론',64.41565036773682,'실제900~950아미노산 단백질384개. GPU토큰 배열384×952(BOS/EOS포함),FP32.','80% 미확인; 다음배치를예상예약량guard로건너뜀'),
 ('부분 파인튜닝',3.6230783462524414,'실제237아미노산 서열128개 학습. 토큰예산4096 가변배치,FP32; 마지막Transformer block·최종LayerNorm·회귀head만학습.','학습배치확대실험 미실시; 전체모델학습값 아님')]),
 ('colabfold','AlphaFold2 — ColabFold','단백질 서열과 MSA → 3D 원자좌표·신뢰도. 단량체 PTM과 Multimer를기본보고서에서구분한다.',['AlphaFold2_ColabFold.md','Batch_memory_followup.md'],[
 ('기본 추론의 누적 관측값',2.705740213394165,'단백질247아미노산 조건까지진행. 단백질1개씩,5개weight set 순차예측,3recycle,BF16계산/FP32가중치. MSA축은부가정보이며단백질batch가아님.','JAX프로세스누적peak(이전query/컴파일포함); CUDA실행구간peak와직접비교불가'),
 ('별도 배치 추론',2.3083367347717285,'각115아미노산인실제단백질4개,model_1 고정,기존MSA·3recycle·FP32,JAX vmap.','80% 미확인; 실제4query풀종료. CLI기본batch기능이아닌별도구현'),
 ('학습·파인튜닝',None,'이번환경에서수행하지않음.','80% 조건 미조사; 미실시를학습불가능으로단정하지않음')]),
 ('scvi','scVI','raw count의 세포×유전자 행렬과 실제생물학적batch → 세포당10차원 latent. 처음부터학습한SCVI 모델.',['scVI.md','Batch_memory_followup.md'],[
 ('추론',.26955747604370117,'한번에10176세포×2000유전자,FP32. 기존저장모델의latent posterior mean.','80% 미확인; 확보한학습세포10176개전체batch까지시험'),
 ('처음부터 학습',.09743928909301758,'학습세포10176개를128세포×2000유전자씩분할,400epoch,FP32.','파인튜닝아님; 학습batch확대 미실시')]),
 ('cellpose','Cellpose','이미지 → 핵/세포별 instance mask. cpsam_v2(약304.6M),영상개수와동시GPU타일개수를구분한다.',['Cellpose.md','Cellpose_batch_memory.md','Cellpose_brain_NucMM.md'],[
 ('이미지 타일 추론',43.39664602279663,'실제520×696·1채널이미지32개→256×256타일384개동시처리. GPU입력384×1×256×256,BF16.','80% 미확인; 다음배치576은예상예약량guard로미실행'),
 ('정답 마스크 기반 파인튜닝',7.9846320152282715,'실제train16/val8/test16이미지. 학습GPU입력1×3×256×256(FP32),100epoch.','학습batch1 관측값; 학습batch확대 미실시'),
 ('뇌 조직 3D 추가 추론',1.4913725852966309,'NucMM-M micro-CT 192×192×192복셀4patch. tilebatch8·256타일,BF16,최종3D마스크후처리포함.','일반뇌MRI아님. AP50평균0.422; 기존타일network시간과범위다름')]),
 ('dnabert2','DNABERT-2','DNA 염기서열 → BPE토큰 → 임베딩/18개histone 표지 예측. 염기수와토큰수는다르다.',['DNABERT2.md','Batch_memory_followup.md'],[
 ('추론',60.0069,'실제4000bp 서열512개,FP32. input_ids/attention_mask 각각512×851(pool최대길이로padding).','80% 미확인; 다음배치예상예약량guard로중단'),
 ('기존 지도 학습',10.571216583251953,'BEND실제1000서열·18표지라벨,4000bp·batch8·FP32·1epoch. 실제697~851토큰.','학습batch확대 미실시. 새128개16step자원probe와구분')]),
 ('ntv2','Nucleotide Transformer V2','DNA서열 → 주로6-mer 토큰 → 임베딩 또는enhancer 분류. 공개500M checkpoint.',['NTv2.md','Batch_memory_followup.md'],[
 ('추론',21.592,'실제6000nt 서열128개. GPU입력128×1001토큰,FP32.','80% 미확인; 확보한고유서열128개한계'),
 ('전체 backbone 파인튜닝',9.252224445343018,'199~200nt enhancer서열,train512/val128/test400. batch8·FP32·2epoch,분류head와backbone학습.','학습batch확대 미실시')]),
 ('clair3','Clair3','long-read BAM+참조유전체 → 변이후보특징배열 → 변이확률/VCF. batch는read수가아닌후보feature 배열수.',['Clair3.md','Batch_memory_followup.md'],[
 ('pileup 후보 추론',4.1114301681518555,'실제HG003 ONT 후보feature5708개. 각후보33×18,입력5708×33×18. FP32모델,원본feature dtype은각audit에기록.','80% 미확인; 확보한고유feature pool끝'),
 ('full-alignment 후보 추론',1.499110221862793,'실제후보feature894개. 각후보89×33×8,입력894×89×33×8,FP32모델.','80% 미확인; 후보좌표를export하지못해고유위치수로주장하지않음'),
 ('학습·파인튜닝',None,'이번작업에서수행하지않음.','공식Train기능과이번실시범위를구분')]),
 ('rnafm','RNA-FM','RNA 염기서열 → 염기토큰 → 640차원 임베딩. 분류실험은본체를동결하고선형head만학습.',['RNAFM.md','Batch_memory_followup.md'],[
 ('기존 실제길이 혼합 추론',8.870131969451904,'train896개 RNA를한번에최대50개씩임베딩추출. 입력은배치별길이로padding,관측shape중최대50×1022토큰(최대1020nt). FP32.','이전체추출구간의peak; 어느개별batch에서peak인지별도분리하지않음. 80% 미확인'),
 ('고정512nt 배치 후속 추론',3.844,'512nt 실제RNA76개,입력76×514토큰(BOS/EOS포함),FP32.','고유76서열풀끝. 이값이전체실험최대라는뜻아님'),
 ('동결 임베딩의 분류head 학습',.43360376358032227,'추출된640차원벡터를batch8×640으로입력,28종RNA label,100epoch. 본체는학습하지않고GPU에남아있었음.','본체파인튜닝 미실시. 메모리에상주한동결본체도포함')]),
 ('openspliceai','OpenSpliceAI','DNA구간을4채널one-hot로변환 → 위치별splice site확률. 같은5개MANE10000 모델ensemble.',['OpenSpliceAI.md','Batch_memory_followup.md'],[
 ('5모델 ensemble 추론',1.463,'실제15000nt 구간156개. GPU입력156×4×15000,출력156×3×5000,FP32. 5모델순차계산/평균.','80% 미확인; 확보한고유구간156개한계'),
 ('단일 seed10 전체 파인튜닝',2.1602683067321777,'실제train96/val24/test36구간. 12×4×15000 입력,정답12×5000 위치라벨,FP32·10epoch.','5모델을함께학습한값아님; 학습batch확대 미실시')]),
 ('scgpt','scGPT','세포×유전자발현 행렬 → 세포별유전자ID·발현bin값 → 임베딩/세포유형. batch는세포행수. 최대1200토큰은CLS+유전자+padding자리.',['scGPT.md','scGPT_memory_batch.md','scGPT_realdata_full.md','scGPT_actual_300k.md'],[
 ('실제30만세포 추론',5.4772539138793945,'고유30만세포를256개씩처리. 세포당최대1200토큰,배치내실제최대길이padding,FP32.','80% 미확인; 30만세포를동시에GPU에올린값아님'),
 ('학습 용량 probe',100.75033235549927,'실제1000학습세포cache 중292세포×1200고정padding,FP32·AdamW·3step. 실제평균nonpad약306자리.','80% 도달 확인(82.79%). padding포함용량probe이며전체데이터파인튜닝과구분'),
 ('실제 전체 pancreas 파인튜닝',3.374143600463867,'train8480/val2120/test4218세포,학습batch8·FP32·최대1200·원래dynamic padding·1epoch.','큰batch284/256/224는메모리guard중단. 이조건에서80%성공을입증하지못함')])]
# Replace rounded placeholders with exact artifact values.
for m in MODELS:
 name=m[0]
 if name in ['dnabert2','ntv2','rnafm','openspliceai']:
  r=ROOT/name/'results';s=json.loads((r/(r/'BATCH_LATEST').read_text().strip()/'summary.json').read_text());exact=s['conditions'][-1]['memory']['peak_allocated_gib']
  index=1 if name=='rnafm' else 0;row=list(m[4][index]);row[1]=exact;m[4][index]=tuple(row)
def shifted(text):
 lines=text.splitlines();out=[];fence=False
 for line in lines:
  if line.startswith('> 최신 통합판:'):continue
  if line.lstrip().startswith(('```','~~~')):fence=not fence
  if not fence and re.match(r'^#{1,5} ',line):line='#'+line
  out.append(line)
 return '\n'.join(out)
for target in [ROOT,PUB]:
 entries=[]
 for model,title,meaning,parts,measurements in MODELS:
  R=target/model/'reports';out=[f'# {title} 최종 통합 보고서','','기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.','','## 1. 입력과 메모리 결론','',meaning,'','**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.','', '|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|','|---|---|---|---|---|---|']
  for task,peak,inp,limit in measurements:
   out.append('|'+ '|'.join([task,f'**{peak:.3f}**' if peak is not None else '미실시',f'{peak/RAM*100:.2f}%' if peak is not None else '—',inp,limit])+'|')
  out+=['','메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.','','## 2. 기본 정보·입력 검사·추론·학습·평가','']
  out.append(shifted((R/parts[0]).read_text()))
  for idx,part in enumerate(parts[1:],start=3):
   out+=['',f'## {idx}. 후속 실험 통합 — '+{'scGPT_memory_batch.md':'학습 배치 확대','scGPT_realdata_full.md':'실제 전체세포 학습','scGPT_actual_300k.md':'실제30만세포 추론','Cellpose_brain_NucMM.md':'뇌 조직 2D·3D 검증'}.get(part,'배치 확대 및 자원 검증'),'',shifted((R/part).read_text())]
  out+=['','## 기록 해석','','본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.']
  (R/'FINAL.md').write_text('\n'.join(out)+'\n');infer=[x[1] for x in measurements if ('추론' in x[0] or '관측값' in x[0]) and x[1] is not None];learn=[x[1] for x in measurements if not ('추론' in x[0] or '관측값' in x[0]) and x[1] is not None];entries.append((title,max(infer) if infer else None,max(learn) if learn else None,'학습용량probe만 확인' if model=='scgpt' else '미확인',model))
 out=['# 최종 통합 보고 — 모델별 입력·추론·학습 메모리','','**이 표에서 모델을 고른 뒤 최종 보고서 하나만 읽으면 된다. 각 문서에 기본 조사·입력 형태·추론·학습·배치 후속·평가·검증을 모두 합쳤다.**','','## 먼저 알아야 할 결론','','메모리80% 목표는allocated약97.35GiB다. 현재이조건을입증한것은scGPT의FP32/1200자리padding/학습용량probe(batch284약80.54%,batch292약82.79%)뿐이다. 일반적인전체세포파인튜닝에서같이달성했다는뜻은아니다. **나머지모델의80%조건은미확인**이다. 후속run을실행했다는것과80%목표를해결했다는것을구분한다.','','## 모델별 최종 보고서','','|모델|추론 관측최대 GiB|학습 관측최대 GiB|메모리80% 조건|읽을 파일|','|---|---|---|---|---|']
 for title,i,t,status,model in entries:out.append('|'+ '|'.join([title,f'{i:.3f}' if i is not None else '미실시',f'{t:.3f}' if t is not None else '미실시',status,f'[최종 보고서]({model}/reports/FINAL.md)'])+'|')
 out+=['','학습열에는작업방식이서로다른scVI처음부터학습·ESM부분파인튜닝·RNA-FM선형head학습·scGPT용량probe가함께있다. 숫자만으로학습메모리순위를매기지말고각최종보고서첫표에서**무엇을학습했는지·입력·배치**를확인한다. AF2메모리는JAX프로세스누적peak라PyTorch실행구간allocated와정의가다르다.','','원본증거를더확인할때만최종보고서안의run JSON/CSV링크를이용하면된다. 이전기본/후속파일은추적용으로보존하되주요읽기경로는이문서와각모델FINAL.md다.']
 (target/'FINAL_REPORT.md').write_text('\n'.join(out)+'\n')
 print('INTEGRATED',target,len(entries))
# Copy the legacy DNA evidence into publication as well.
dst=PUB/'dnabert2/results/legacy_training_4000bp';dst.mkdir(exist_ok=True)
for p in (ROOT/'dnabert2/results/legacy_training_4000bp').glob('*.json'):shutil.copy2(p,dst/p.name)
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/build_results_index.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/build_results_index.py", "sha256": "6cd748826101815906a53f5ec4775e62b1906afaf3c121d493cb767ee3b530e1", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Readable input -> GPU utilization + memory overview from actual recorded results."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[1];RAM=130660974592/2**30;out=[]
def read(p):return json.loads(p.read_text())
def add(s=''):out.append(s)
def table(headers,rows):
 add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
 for r in rows:add('| '+' | '.join(map(str,r))+' |')
 add()
def mem(x):return f'{x:.3f} GiB ({x/RAM*100:.2f}%)'
def util(a,b):return '미측정' if a is None else f'{a:.1f}% / {b:.0f}%'
add('# 모델별 실제 입력·GPU 사용률·메모리');add();add('갱신: '+datetime.datetime.now().isoformat(timespec='seconds'));add();add('**GPU 사용률(%)은 계산 활동 시간의 비율이고, 메모리(GiB)는 저장 공간이다.** 메모리 괄호 %는 프레임워크의 프로세스 할당량을 전체 공유 RAM 121.688 GiB로 나눈 참고값이며 시스템 전체 메모리 사용률이 아니다. Spark에서 CPU/GPU 메모리는 공유하므로 RSS와 GPU 할당량을 더하지 않는다.');add();add('DNABERT-2의 기존 4,000bp 파인튜닝에서는 10.571 GiB(공유 RAM 대비 약8.69%)를 관측했다. Cellpose 등의 추가 결과는 아래 표를 함께 확인한다. **메모리90%(약109.5 GiB) 도달 조건은 아직 확인되지 않았다.**');add()
add('## 모델을 입력 없이 적재했을 때');add();bl=[]
e=ROOT/'esm2/results/LOADING_LATEST'
if e.exists():
 b=read(e.parent/e.read_text().strip()/'loading.json');bl.append(['ESM-2 650M',format(b['total_parameters'],','),'FP32',f"{b['cuda_after_load_allocated_gib']:.6f}",f"{b['rss_after_load_gib']:.3f}",'새 프로세스, forward 없음'])
for preset in ['alphafold2_ptm','alphafold2_multimer_v3']:
 q=ROOT/'colabfold/results'/('LOADING_'+preset+'_LATEST')
 if q.exists():
  b=read(q.parent/q.read_text().strip()/'loading.json');bl.append([preset,format(b['parameters_per_model'][0]['parameters'],',')+' × 5 sets','FP32 weights / BF16 compute',f"{b['before_first_forward_ready_allocated_gib']:.6f}",f"{b['rss_after_load_gib']:.3f}",'5개 weight set 보관; 첫 모델 선택 뒤 ready 상태'])
q=ROOT/'scvi/results/LATEST'
if q.exists():
 bpath=q.parent/q.read_text().strip()/'loading_baselines.json'
 if bpath.exists():
  b=read(bpath)[0];bl.append(['scVI SCVI 기본 구조',format(b['total_parameters'],','),'FP32',f"{b['cuda_after_load_allocated_gib']:.6f}",f"{b['rss_after_model_load_gib']:.3f}",'학습 모델 로드; AnnData는 CPU에 먼저 읽음; optimizer 없음'])
q=ROOT/'cellpose/results/LATEST'
if q.exists():
 cp=q.parent/q.read_text().strip()/'summary.json'
 if cp.exists():
  c=read(cp);b=c['model_only_postload'];bl.append(['Cellpose-SAM cpsam_v2',format(c['parameters']['total'],','),'BF16 inference / FP32 training',f"{b['allocated_gib']:.6f}",f"{b['rss_gib']:.3f}",'fresh process, first forward before input'])
q=Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/results/20261007_015418_n100000_b8_bp300_fp32_1791338058787754457/summary.json')
if q.exists():
 b=read(q);bl.append(['DNABERT-2',format(b['parameters'],','),'FP32',f"{b['post_load_gpu_allocated_gib']:.6f}",f"{b['post_load_rss_gib']:.3f}",'기존 run post-load 기록 재사용'])
table(['모델','실행 구성 파라미터','정밀도','GPU allocated GiB','프로세스 RSS GiB','측정 범위'],bl);add('가중치 용량 이론값과 위 실측은 다르다. GPU allocator 외 드라이버·runtime과 CPU 데이터도 RAM을 쓰며 RSS에 포함될 수 있다. 로딩 기준과 아래 실행 peak가 다른 run이면 차이는 참고값이다.');add()
headers=['모델·작업','입력당 크기','실제 배치','고유 입력','GPU 평균 / 최대','GPU 할당 peak (공유 RAM 대비 참고 %)']
esm=ROOT/'esm2';base=esm/'results/20261027_164300_125163';sup=esm/'results'/(esm/'results/SUPPLEMENT_LATEST').read_text().strip();rows=[]
for r in read(sup/'summary.json')['conditions']:
 bs=read(base/(r['condition']+'_batches.json'));a=min(b['samples'] for b in bs);b=max(b['samples'] for b in bs);rows.append(['ESM-2 650M 추론',f"{r['length_min']}–{r['length_max']} 아미노산",str(a) if a==b else f'{a}–{b}',r['unique_inputs'],util(r['gpu_util_mean_percent'],r['gpu_util_max_percent']),mem(r['cuda_peak_allocated_gib'])])
t=read(base/'summary.json')['training'];rows.append(['ESM-2 부분 파인튜닝','235–237 아미노산','토큰 예산 4096',128,util(t['gpu_util_mean_percent'],t['gpu_util_max_percent']),mem(t['cuda_peak_allocated_gib'])]);add('## ESM-2');add();table(headers,rows);add('추론은 각 고유 서열 16개를 40회 반복 측정했다. FP32, 토큰 예산 4096. 최대 96%는 별도 기본 실행에서 관측됐고 위 표는 관측 구간을 늘린 보완 실행 수치다. 파인튜닝은 마지막 Transformer block·LayerNorm·회귀 head만 학습했다.');add();add('[상세 보고서](esm2/reports/ESM2_20261027_164300_125163.md)');add()
af=ROOT/'colabfold/reports/metrics.json'
if af.exists():
 rows=[]
 for r in read(af):rows.append(['AF2 Multimer v3' if 'multimer' in r['run'] else 'AF2 PTM',f"{r['original_residues']} 아미노산 (pad {r['padded_residue_axis']})",'1; 모델 5개 순차',1,util(r['gpu_util_mean_percent'],r['gpu_util_max_percent']),mem(r['jax_allocator_peak_gib_since_process_start'])])
 add('## AlphaFold2 — ColabFold');add();table(headers,rows);add('JAX 메모리는 각 프로세스 시작 이후의 **누적 allocator peak**이며 입력별로 초기화하지 않았다. 단일체 3 recycles, 복합체 20 recycles의 별도 preset이며 BF16 계산·FP32 가중치 경로다.');add();add('[상세 보고서](colabfold/reports/AlphaFold2_ColabFold.md) · [비교 CSV](colabfold/reports/comparison.csv)');add()
dna=Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/bend_length_results');rows=[]
for p in sorted(dna.glob('*training*bp*/summary.json')):
 r=read(p);rows.append(['DNABERT-2 파인튜닝',f"{r['settings']['bp']} bp → 평균 {r['tokens_mean']:.1f} tokens",r['settings']['batch'],r['settings']['n'],'미측정',mem(r['gpu_peak_allocated_gib'])])
add('## DNABERT-2 기존 기록');add();table(headers,rows);add('위 기존 길이 실험에는 GPU-Util이 없다. 메모리 비율을 GPU 사용률로 바꿔 해석하면 안 된다. 최신 GPU 사용률 보완 실험은 지정된 후속 순서에 남아 있다.');add()
scvi=ROOT/'scvi/results/LATEST'
if scvi.exists():
 o=scvi.parent/scvi.read_text().strip();s=read(o/'summary.json');rows=[]
 for r in s['training']:
  p=r['training_phase_profile'];rows.append(['scVI 처음부터 학습',f"세포당 {r['genes']} 유전자",128,r['unique_training_cells'],util(p['gpu_util_mean_percent'],p['gpu_util_max_percent']),mem(r['cuda_peak_allocated_gib'])])
 for r in s['inference']:rows.append(['scVI latent 추론',f"세포당 {r['genes']} 유전자",128,r['unique_inputs'],util(r['gpu_util_mean_percent'],r['gpu_util_max_percent']),mem(r['cuda_peak_allocated_gib'])])
 add('## scVI');add();table(headers,rows);add('실행 상태: '+s['status']+'. 학습은 400 epochs, FP32, batch 128을 고정한다. 학습 메모리는 해당 fit 전체(검증 포함)의 peak이고 GPU 평균은 training phase 표본이다. 완료 전 행은 후속 모델 재로드 검증이 남아 있을 수 있다.');add()
q=ROOT/'cellpose/results/LATEST'
if q.exists():
 cp=q.parent/q.read_text().strip()/'summary.json'
 if cp.exists():
  c=read(cp);rows=[]
  for r in c['conditions']:rows.append(['Cellpose-SAM inference',str(r['raw_shapes'][0]),'up to8 tiles, 256x256',r['unique_images'],util(r['gpu_util_avg'],r['gpu_util_max']),mem(r['memory']['peak_allocated_gib'])])
  if 'training' in c:
   t=c['training'];rows.append(['Cellpose-SAM fine-tuning','native520x696 ->256crop','1 crop',16,util(t['gpu_util_avg'],t['gpu_util_max']),mem(t['memory']['peak_allocated_gib'])])
  add('## Cellpose-SAM');add();table(headers,rows);add('Status: '+c.get('status','running')+'. BF16 inference / official FP32 fine-tuning, fixed tile policy. Unique image counts exclude measurement repeats. Training utilization includes label preprocessing and validation.');add();add('[Cellpose 상세 보고서](cellpose/reports/Cellpose.md) · [입력·정답 예시](cellpose/reports/input_example.png) · [비교 CSV](cellpose/reports/comparison.csv)');add()
q=ROOT/'dnabert2/results/LATEST'
if q.exists():
 cp=q.parent/q.read_text().strip()/'summary.json'
 if cp.exists():
  c=read(cp);rows=[]
  for r in c['conditions']:rows.append(['DNABERT-2 '+r['mode'],str(r['bp'])+'bp / mean '+str(round(r['tokens_mean'],1))+' tokens',8,r['unique_inputs'],util(r['gpu_util_avg'],r['gpu_util_max']),mem(r['memory']['peak_allocated_gib'])])
  add('## DNABERT-2 GPU 사용률 보완');add();table(headers,rows);add('기존 학습 모델은 보존했다. 새 학습 행은128개16step 자원 probe이며, 원본1000개 학습 결과에 사용률을 소급해 붙인 값이 아니다.');add();add('[DNABERT-2 보완 보고서](dnabert2/reports/DNABERT2.md) · [비교 CSV](dnabert2/reports/comparison.csv)');add()
q=ROOT/'scgpt/results/finetuned_test_20261023_034345_99523/summary.json'
if q.exists():
 c=read(q);add('## scGPT 기존 완료 결과');add();add('기본 추론·파인튜닝·저장·재로드·test·길이 비교 완료. test4,218세포 중 알려진 라벨4,213세포 기준 Accuracy '+str(round(c['accuracy'],4))+', 학습13클래스 macroF1 '+str(round(c['macro_f1_all_training_classes'],4))+'. GPU 개수/사용률 보완은 별도 후속 순서다.');add();add('[저장 모델 test summary](scgpt/results/finetuned_test_20261023_034345_99523/summary.json) · [길이 비교 CSV](scgpt/results/length_compare_20261023_040313_501423/comparison.csv)');add()
for name,title,report in [('ntv2','Nucleotide Transformer V2','NTv2'),('clair3','Clair3','Clair3'),('rnafm','RNA-FM','RNAFM'),('openspliceai','OpenSpliceAI','OpenSpliceAI')]:
 q=ROOT/name/'results/LATEST'
 if q.exists():
  cp=q.parent/q.read_text().strip()/'summary.json'
  if cp.exists():
   c=read(cp);rows=[]
   for r in c.get('conditions',[]):
    if 'memory' not in r:continue
    rows.append([title+' '+r.get('mode',''),str(r.get('bp',r.get('length','see report'))),'see report',r.get('unique_inputs','N/A'),util(r.get('gpu_util_avg'),r.get('gpu_util_max')),mem(r['memory']['peak_allocated_gib'])])
   add('## '+title);add();add('상태: '+c.get('status','running'));add();b=c.get('model_only_postload')
   if b:add('모델 적재 후/첫 forward 전 CUDA allocated '+str(round(b['allocated_gib'],6))+'GiB; reserved '+str(round(b['reserved_gib'],3))+'GiB; RSS '+str(round(b['rss_gib'],3))+'GiB.');add()
   if rows:table(headers,rows)
   if (ROOT/name/'reports'/(report+'.md')).exists():add('[상세 보고서]('+name+'/reports/'+report+'.md)');add()
q=ROOT/'scgpt/results/SUPPLEMENT_LATEST'
if q.exists():
 c=read(q.parent/q.read_text().strip()/'summary.json');rows=[]
 for r in c['conditions']:rows.append(['scGPT embedding','cap'+str(r['length'])+' / mean '+str(round(r['tokens_mean'],1))+' tokens',8,r['unique_inputs'],util(r['gpu_util_avg'],r['gpu_util_max']),mem(r['memory']['peak_allocated_gib'])])
 if 'training_resource_supplement' in c:
  t=c['training_resource_supplement'];e=t['epochs_detail'][0];rows.append(['scGPT training utilization supplement','max1200 incl CLS',8,t['training_cells'],util(e['gpu_util_mean_percent'],e['gpu_util_max_percent']),mem(e['gpu_peak_allocated_gib'])])
 add('## scGPT GPU 사용률 보완');add();table(headers,rows);b=c['model_only_postload'];add('새 프로세스 추론모델 적재 allocated '+str(round(b['allocated_gib'],6))+'GiB; reserved '+str(round(b['reserved_gib'],3))+'GiB; RSS '+str(round(b['rss_gib'],3))+'GiB. 기존기본/길이/test결과와분리해보완했다.');add();add('[scGPT 상세 보고서](scgpt/reports/scGPT.md) · [보완 비교 CSV](scgpt/reports/comparison_supplement.csv)');add()
add('## 후속 메모리 확대 실험');add();add('사용자가 batch를단계적으로늘려메모리80~90% 도달가능성과처리량을확인하는후속실험을추가했다. **후속실험은별도run으로실행중이다.** 위표는기본고정batch 실험/보완결과이며최대지원batch/최대점유한계가아니다. [후속메모리실험](MEMORY_RESULTS.md)');add()
add('[scVI 상세 보고서](scvi/reports/scVI.md) · [실제 학습 입력 예](scvi/reports/training_input_example.csv)');add();add('[전체 진행 상태](STATUS.md)');(ROOT/'RESULTS.md').write_text('\n'.join(out)+'\n');print(ROOT/'RESULTS.md')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/clair3_batch_launch.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/clair3_batch_launch.py", "sha256": "b8724c07ea44c668b75692f6f11f1c62090676709f754db4e207018948f8e0f9", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,subprocess,shlex,os,sys,time
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'clair3';run=(R/'results/BATCH_LATEST').read_text().strip();D=R/'results'/run;assert json.loads((D/'capture_exit.json').read_text())['exit_code']==0
S={'status':'running','run_id':run,'conditions':[],'failed':[],'settings':{'task':'native candidate model inference only','threads':4,'precision':'FP32','tf32':False,'guard_available_gib':16,'guard_swap_increase_gib':2,'baseline_calling_batch':1000,'note':'BAM candidate extraction completed once; actual cached candidate rows reused; no artificial repeated rows within batch'}};swap0=psutil.swap_memory().used
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
def stage(label,script,envextra=()):
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU job'
 name='clair3-'+label+'-'+run;cmd=['docker','run','--rm','--name',name,'--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','PYTHONPATH=/work/source:/work/scripts']
 for e in envextra:cmd+=['-e',e]
 cmd+=['local/clair3-spark:9d1dd3f','/work/.venv/bin/python','-u','/work/scripts/batch_followup/'+script];(D/(label+'_command.json')).write_text(json.dumps(cmd));t=time.time();reason=None
 with (D/(label+'.log')).open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/(label+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-t,'reason':reason}));print(label,code,flush=True);return code
save();assert stage('prepare','prepare_cache.py')==0;audit=json.loads((D/'input_audit.json').read_text());S['input_audit']=audit;save()
for kind in ['Clair3_P','Clair3_F']:
 pool=audit[kind]['unique_feature_rows'];last=None
 for B in sorted(set([b for b in [1,8,32,128,512,1000,2048,4096,8192,16384] if b<=pool]+[pool])):
  if last:
   base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*B/last['batch_size']*1.25+3
   if estimate>psutil.virtual_memory().available/2**30-16:S.setdefault('stop_reasons',{})[kind]='predicted_reservation_guard';break
  code=stage(kind.lower()+'-'+str(B),'worker.py',['CANDIDATE_MODEL='+kind,'BATCH='+str(B)])
  if code:S['failed'].append({'model':kind,'batch':B,'exit_code':code});break
  w=json.loads((D/f'{kind}_batch_{B}/summary.json').read_text());last=w['condition'];last['model']=kind;last['model_only_allocated_gib']=w['model_only_postload']['allocated_gib'];S['conditions'].append(last);save()
 else:S.setdefault('stop_reasons',{})[kind]='available_real_candidate_cache_limit'
S['status']='complete' if not S['failed'] else 'stopped';save();subprocess.run([sys.executable,str(ROOT/'scripts/two_batch_report.py'),'clair3',str(D)],check=True);print('CLAIR3_FINISHED',S['status'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/fetch_hf_files.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/fetch_hf_files.py", "sha256": "9b3766589dc4ac5d21f8f97cc7c18767209bd0101365a9f9d08f7e2b65709489", "encoding": "utf-8", "added_final_newline": false} -->
```python
import urllib.request,json,hashlib,concurrent.futures,time
from pathlib import Path

def fetch(url,p,size=None,sha=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  if size and size>32*1024**2:
   chunks=p.with_name(p.name+'.chunks');chunks.mkdir(exist_ok=True);step=4*1024**2
   def get(a):
    b=min(a+step,size)-1;d=chunks/str(a)
    if d.exists() and d.stat().st_size==b-a+1:return d
    for retry in range(4):
     try:
      req=urllib.request.Request(url+f'?download=true&part={a}',headers={'Range':f'bytes={a}-{b}'})
      with urllib.request.urlopen(req,timeout=120) as f:
       assert f.status==206 and f.headers['Content-Range'].startswith(f'bytes {a}-{b}/');data=f.read();assert len(data)==b-a+1;d.write_bytes(data)
      return d
     except Exception:
      if retry==3:raise
      time.sleep(1)
   ranges=range(0,size,step)
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for i,d in enumerate(pool.map(get,ranges)):
     if i%64==0:print(p.name,'chunks',i+1,'/',len(ranges),flush=True)
   tmp=p.with_name(p.name+'.part')
   with tmp.open('wb') as f:
    for a in ranges:f.write((chunks/str(a)).read_bytes())
   tmp.rename(p)
  else:
   with urllib.request.urlopen(url,timeout=120) as f:data=f.read()
   p.write_bytes(data)
 h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert size is None or p.stat().st_size==size;assert sha is None or h==sha,(p,h,sha)
 return {'path':str(p),'url':url,'sha256':h,'bytes':p.stat().st_size}
if __name__=='__main__':
 R=Path('/home/hyein/ai-benchmark/ntv2');rows=[]
 for folder,kind,repo in [('models','models','InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'),('data','datasets','InstaDeepAI/nucleotide_transformer_downstream_tasks')]:
  a=json.loads((R/folder/'repository.json').read_text());rev=a['meta']['sha']
  for f in a['tree']:
   name=f['path']
   choose=(folder=='models' and '/' not in name and name not in ['pytorch_model.bin','.gitattributes'] and f['type']=='file') or (folder=='data' and name in ['enhancers/train.parquet','enhancers/test.parquet'])
   if choose:
    url='https://huggingface.co/'+('datasets/' if kind=='datasets' else '')+repo+'/resolve/'+rev+'/'+name
    rows.append(fetch(url,R/folder/('checkpoint/'+name if folder=='models' else name),f['size'],f.get('lfs',{}).get('oid')));(R/'results/sources.json').write_text(json.dumps(rows,indent=2));print('FETCHED',name,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/gpu_support.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/gpu_support.py", "sha256": "349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2", "encoding": "utf-8", "added_final_newline": false} -->
```python
import time,json,csv,threading,subprocess,os
from pathlib import Path
import numpy as np,psutil,torch
G=2**30
class Meter:
 def __init__(self,root):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.csv_path=self.root/'resources.csv';
  if self.csv_path.exists():self.csv_path=self.root/('resources_'+str(time.time_ns())+'.csv')
  self.phase='initial';self.rows=[];self.stop=threading.Event();self.swap0=psutil.swap_memory().used;self.th=threading.Thread(target=self.monitor,daemon=True);self.th.start()
 def memory(self):
  v=psutil.virtual_memory();return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,system_available_gib=v.available/G,system_total_gib=v.total/G)
 def monitor(self):
  with self.csv_path.open('w',newline='') as f:
   keys=['query_start','query_end','phase','gpu_util','allocated_gib','reserved_gib','rss_gib','available_gib','swap_gib'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   while not self.stop.is_set():
    t=time.time();phase=self.phase
    try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().splitlines()[0])
    except Exception:u=None
    v=psutil.virtual_memory();r=dict(query_start=t,query_end=time.time(),phase=phase,gpu_util=u,allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=v.available/G,swap_gib=psutil.swap_memory().used/G);self.rows.append(r);w.writerow(r);f.flush()
    if v.available<8*G or psutil.swap_memory().used>self.swap0+2*G:(self.root/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
    self.stop.wait(.5)
 def stats(self,t,end):
  rows=[r for r in self.rows if r['query_start']>=t and r['query_end']<=end];u=[r['gpu_util'] for r in rows if r['gpu_util'] is not None];return dict(samples=len(u),gpu_util_avg=float(np.mean(u)) if u else None,gpu_util_max=max(u) if u else None,rss_sampled_peak_gib=max((r['rss_gib'] for r in rows),default=None),available_sampled_min_gib=min((r['available_gib'] for r in rows),default=None))
 def begin(self,name):
  self.phase=name;torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();return time.time(),self.memory()
 def end(self,t,base,**kw):
  torch.cuda.synchronize();end=time.time();m=self.memory();return dict(**kw,seconds=end-t,baseline=base,memory=m,peak_delta_gib=m['peak_allocated_gib']-base['allocated_gib'],**self.stats(t,end))
 def close(self):self.stop.set();self.th.join(timeout=5)
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/launch_model.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/launch_model.py", "sha256": "bfc0e857b4135402be2fd0335265b13299bf5234449a467e9fd2d6ff78f6fdeb", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,datetime,json,sys,shlex
from pathlib import Path
name=sys.argv[1];script=sys.argv[2] if len(sys.argv)>2 else 'benchmark.py';R=Path('/home/hyein/ai-benchmark')/name
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU job is running'
if script=='benchmark.py':
 run=name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n')
else:run=(R/'results/LATEST').read_text().strip();D=R/'results'/run
cmd=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v {R}:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID={run} -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/{script}'];stem=script.removesuffix('.py');(D/(stem+'_command.json')).write_text(json.dumps(cmd,indent=2));print(run,script,flush=True)
with (D/(stem+'.log')).open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/(stem+'_exit.json')).write_text(json.dumps({'exit_code':code,'finished':datetime.datetime.now().isoformat()}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/remaining_followups/launch.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/launch.py", "sha256": "84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,subprocess,time,datetime,sys,os,shlex,csv,hashlib
import psutil
ROOT=Path('/home/hyein/ai-benchmark');plan=['scvi','dnabert2','ntv2','rnafm','openspliceai'];stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');Q={'status':'running','queue_id':stamp,'order':plan,'models':{},'scope':'inference batch expansion, same existing weights; sequential GPU only','applicability_notes':{'AlphaFold2':'ColabFold query list is processed sequentially; no independent-query GPU batch axis exposed in existing official path. Complex chain count changes biological input, not batch. No artificial concurrent GPU jobs or replacement model. Existing baseline retained; dedicated batch sweep not performed.','Clair3':'Existing official calling pipeline controls batch internally (predictBatchSize and GPU multiplier), no explicit CLI minibatch argument in installed path. Do not confuse candidate count, threads/workers and batch. Dedicated source-level candidate-cache batch sweep not performed; existing measured GPU batch1000 and coverage/region results retained.'}}
qp=ROOT/'FOLLOWUP_QUEUE.json'
def queue_save():
 qp.write_text(json.dumps(Q,indent=2));lines=['# 후속 배치 실험 진행','',Q['scope'],'','|모델|상태|최대 성공batch|보고서|','|---|---|---|---|']
 for model in plan:
  s=Q['models'].get(model,{});lines.append('|'+model+'|'+s.get('status','queued')+'|'+str(s.get('largest_successful_batch',''))+'|['+model+']('+model+'/reports/Batch_memory_followup.md)|')
 lines+=['','## 배치 확대 적용 범위','']+[k+': '+v for k,v in Q['applicability_notes'].items()];(ROOT/'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines)+'\n')
queue_save()
for model in plan:
 R=ROOT/model;run='batch_followup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');records=[]
 if model=='dnabert2':records=list(csv.DictReader(Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv').open()));source='existing BEND prepared n1000, fixed4000bp, actual native tokenizer length verified worker'
 elif model=='ntv2':
  original=json.loads((R/'data/inputs.json').read_text())['long_inference'];records=[{**r,'sequence':r['sequence'][(len(r['sequence'])-6000)//2:(len(r['sequence'])-6000)//2+6000]} for r in original];source='existing real GRCh38 regions, central6000nt crops -> native ~1001tokens'
 elif model=='rnafm':
  original=json.loads((R/'data/inputs.json').read_text());records=[{**r,'sequence':r['sequence'][:512]} for split in ['train','validation','test'] for r in original[split] if len(r['sequence'])>=512];source='official tutorial RNAcentral actual unique sequences >=512nt, first512nt explicit crop; pool limit retained'
 elif model=='openspliceai':
  original=json.loads((R/'data/inputs.json').read_text());records=[{**r,'sequence':r['sequence'][2500:17500]} for split in ['train','validation','test'] for r in original[split]];source='actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble'
 else:source='saved full10176cell SCVI model and its raw-count AnnData, fixed2000genes; latent posterior mean'
 if records:
  seen=set();unique=[]
  for r in records:
   assert r['sequence'] and set(r['sequence'])<=set('ACGTUNRYKMSWBDHVX')
   if r['sequence'] not in seen:unique.append(r);seen.add(r['sequence'])
  records=unique;assert len({r['id'] for r in records})==len(records);(D/'inputs.json').write_text(json.dumps(records,indent=2));pool=len(records)
 else:pool=10176
 S={'status':'running','model':model,'run_id':run,'conditions':[],'failed':[],'input_audit':{'source':source,'unique_available_inputs':pool,'selection':'nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained','token_policy':'fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers'},'largest_successful_batch':None};swap0=psutil.swap_memory().used
 def save():(D/'summary.json').write_text(json.dumps(S,indent=2));Q['models'][model]={'status':S['status'],'run_id':run,'largest_successful_batch':S.get('largest_successful_batch')};queue_save()
 save();print('MODEL_START',model,run,pool,flush=True)
 sizes=[1,2,4,8,16,32,64,96,128,192,256,384,512,768,1024,2048,4096,8192] if model!='scvi' else [128,256,512,1024,2048,4096,8192]
 sizes=sorted(set([b for b in sizes if b<=pool]+[pool]))
 for batch in sizes:
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if active:S['stop_reason']='other_GPU_job_present';S['status']='waiting_external_gpu';save();raise RuntimeError('Do not overlap existing GPU job '+active)
  available=psutil.virtual_memory().available/2**30
  if available<16 or psutil.swap_memory().used>swap0+2*2**30:S['stop_reason']='system_memory_guard';break
  if S['conditions']:
   last=S['conditions'][-1];base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*batch/last['batch_size']*1.25+3
   if estimate>available-16:S['stop_reason']='predicted_reservation_exceeds_available_budget';S['skipped_next_batch']={'batch':batch,'estimated_reserved_gib':estimate,'available_gib':available};break
  name='followup-'+model+'-'+run.lower()+'-'+str(batch);cmd=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--ipc=host','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','MODEL_KIND='+model,'-e','BATCH='+str(batch),'-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/work/.cache/numba','-e','HF_HOME=/work/.cache/huggingface','-e','HF_MODULES_CACHE=/work/.cache/modules','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1']
  python='/work/.venv/bin/python'
  if model=='dnabert2':cmd+=['-v','/home/hyein/dnabert2-benchmark:/workspace:ro'];python='/workspace/miniforge/envs/dnabert2/bin/python'
  cmd+=['nvcr.io/nvidia/pytorch:26.06-py3',python,'-u','/support/remaining_followups/worker.py'];(D/f'batch{batch}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();reason=None
  with (D/f'batch{batch}.log').open('w') as f:
   p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
   while p.poll() is None:
    if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
    time.sleep(.5)
   code=p.wait()
  (D/f'batch{batch}_exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'reason':reason}));print('BATCH_EXIT',model,batch,code,flush=True)
  if code:S['failed'].append({'batch':batch,'exit_code':code,'reason':reason or 'worker_error_see_log'});S['stop_reason']='worker_stopped_see_log';break
  worker=json.loads((D/f'batch_{batch}/summary.json').read_text());assert worker['status']=='complete';S['conditions'].append(worker['conditions'][0]);S['largest_successful_batch']=batch;save()
 else:S['stop_reason']='available_unique_input_limit'
 S['status']='complete' if not S['failed'] else 'stopped';S['reached80']=any(c['allocated_fraction_shared_ram']>=.8 for c in S['conditions']);S['reached90']=any(c['allocated_fraction_shared_ram']>=.9 for c in S['conditions']);save();subprocess.run([sys.executable,str(ROOT/'scripts/remaining_followups/report.py'),model,str(D)],check=True);print('MODEL_FINISH',model,S['status'],S.get('stop_reason'),flush=True)
Q['status']='finished_with_unresolved_items' if any(s['status']!='complete' for s in Q['models'].values()) else 'complete_for_supported_sweeps';queue_save();print('QUEUE_FINISHED',Q['status'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/remaining_followups/report.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/report.py", "sha256": "8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,sys,csv
ROOT=Path('/home/hyein/ai-benchmark');model=sys.argv[1];D=Path(sys.argv[2]);S=json.loads((D/'summary.json').read_text());R=ROOT/model;lines=[f'# {model} 실제 입력의 배치별 추론 자원','',f"상태 **{S['status']}** · run `{D.name}` · 종료사유 `{S.get('stop_reason','running')}`",'','## 입력과 고정 조건','',json.dumps(S['input_audit'],ensure_ascii=False,indent=2),'','배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.','','FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.','','## 배치별 측정','','측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.','', '|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|','|---|---|---|---|---|---|---|---|---|---|---|---|---|'];flat=[]
for c in S['conditions']:
 m=c['memory'];b=c['model_only_allocated_gib'];r={'batch':c['batch_size'],'repeats':c['repeats'],'processed_inputs':c['processed_inputs'],'seconds':c['seconds'],'inputs_per_second':c['inputs_per_second'],'loaded_allocated_gib':b,'allocated_peak_gib':m['peak_allocated_gib'],'additional_from_same_run_load_gib':m['peak_allocated_gib']-b,'reserved_peak_gib':m['peak_reserved_gib'],'percent_shared_ram':100*c['allocated_fraction_shared_ram'],'gpu_mean':c['gpu_util_avg'],'gpu_max':c['gpu_util_max'],'rss_sampled_peak_gib':c['rss_sampled_peak_gib'],'available_sampled_min_gib':c['available_sampled_min_gib']};flat.append(r);lines.append('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
lines+=['','## 결과 해석과 검증','',f"최대시험성공batch {S.get('largest_successful_batch')}. allocated80%도달 {S.get('reached80')},90%도달 {S.get('reached90')}. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.",'','출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.','','GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.','','원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.']
sources={'scvi':'https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html','dnabert2':'https://github.com/MAGICS-LAB/DNABERT_2','ntv2':'https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species','rnafm':'https://github.com/ml4bio/RNA-FM','openspliceai':'https://github.com/Kuanhao-Chao/OpenSpliceAI'}
lines+=['','[공식 출처]('+sources[model]+') · [기존 모델 보고서 디렉터리](.)']
(R/'reports/Batch_memory_followup.md').write_text('\n'.join(lines)+'\n')
if flat:
 with (R/'reports/batch_memory_followup.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for name in ['STATUS.md','RESULTS.md','MEMORY_RESULTS.md']:
 with (ROOT/name).open('a') as f:f.write('\n'+model+' 배치후속 '+S['status']+', 최대성공batch '+str(S.get('largest_successful_batch'))+'. [보고서]('+model+'/reports/Batch_memory_followup.md). 종료사유 '+S.get('stop_reason','running')+'.\n')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/remaining_followups/worker.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/worker.py", "sha256": "e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc,argparse
import numpy as np,torch
sys.path.insert(0,'/support');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];kind=os.environ['MODEL_KIND'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'inference resource batch sweep only, no training','conditions':[]}
def save():write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='load';t=time.time();tokenizer=None
 if kind=='scvi':
  import scvi
  P=R/'models/scvi_20261027_185327_767988/train_10176';m=scvi.model.SCVI.load(str(P),accelerator='cpu');m.to_device('cuda:0');m.module.eval();nets=[m.module];ad=m.adata;assert ad.n_vars==2000 and ad.n_obs>=B;S['checkpoint']=str(P);S['input']={'unique_cells':B,'genes':ad.n_vars,'raw_counts_source':'saved adata.h5ad layers counts, actual 10x PBMC4k+8k','original_pool_cells':ad.n_obs,'batch_covariate':'sample_id, biological batch distinct from GPU batch'}
  x=ad[:B].copy()
  def forward():return m.get_latent_representation(adata=x,batch_size=B,give_mean=True)
 else:
  records=json.loads((D/'inputs.json').read_text());assert len(records)>=B
  if kind=='dnabert2':
   from transformers import AutoModel,AutoTokenizer,BertConfig
   P=Path('/workspace/dnabert2_eval/model');tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModel.from_pretrained(P,config=BertConfig.from_pretrained(P),trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='ntv2':
   from transformers import AutoModelForMaskedLM,AutoTokenizer
   P=R/'models/checkpoint';tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='rnafm':
   import fm
   torch.serialization.add_safe_globals([argparse.Namespace]);m,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');m=m.float().cuda().eval();nets=[m];converter=alphabet.get_batch_converter()
  elif kind=='openspliceai':
   from openspliceai.train_base.openspliceai import SpliceAI
   W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);nets=[]
   for seed in range(10,15):
    net=SpliceAI(32,W,AR).float().cuda();net.load_state_dict(torch.load(R/f'source/models/openspliceai-mane/10000nt/model_10000nt_rs{seed}.pt',map_location='cpu',weights_only=True));net.eval();nets.append(net)
  else:raise ValueError(kind)
 torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for net in nets for p in net.parameters());S['parameters_trained_in_this_run']=0;S['theoretical_weight_storage_gib']=sum(p.numel()*p.element_size() for net in nets for p in net.parameters())/2**30;S['parameters_used_backbone']=sum(p.numel() for p in m.esm.parameters()) if kind=='ntv2' else S['parameters_total'];S['checkpoint_reference']={'scvi':'models/scvi_20261027_185327_767988/train_10176','dnabert2':'/workspace/dnabert2_eval/model + model_revision.txt','ntv2':'models/checkpoint; existing setup/download provenance','rnafm':'models/RNA-FM_pretrained.pth; existing data/sources.json','openspliceai':'source/models/openspliceai-mane/10000nt seeds10..14'}[kind];S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'precision':'FP32','tf32':False};save()
 if kind!='scvi':
  cache=D/'token_cache.pt'
  if not cache.exists():
   if tokenizer is not None:
    encoded=tokenizer([r['sequence'] for r in records],padding=True,truncation=False,return_tensors='pt');encoded={k:encoded[k] for k in ['input_ids','attention_mask']}
   elif kind=='rnafm':
    _,_,tok=converter([(r['id'],r['sequence']) for r in records]);encoded={'input_ids':tok}
   else:
    tab=np.zeros((256,4),np.uint8)
    for j,c in enumerate('ACGT'):tab[ord(c),j]=1
    encoded={'onehot':torch.from_numpy(np.stack([tab[np.frombuffer(r['sequence'].encode(),np.uint8)].T for r in records]))}
   torch.save(encoded,cache)
  encoded=torch.load(cache,map_location='cpu',weights_only=True);cpu={k:v[:B] for k,v in encoded.items()};S['input']={'unique_sequences':B,'pool_sequences':len(records),'raw_length_min':min(len(r['sequence']) for r in records[:B]),'raw_length_max':max(len(r['sequence']) for r in records[:B]),'input_shapes':{k:list(v.shape) for k,v in cpu.items()},'tokenizer_truncation':False,'padding':'same pool-wide longest token width fixed across batches; explicit crop lengths in inputs provenance','ids':[r['id'] for r in records[:B]]}
  if kind=='ntv2':assert cpu['input_ids'].shape[1]<=2048
  if kind=='rnafm':assert cpu['input_ids'].shape[1]<=1024
  def forward():
   x={k:(v.cuda().float() if k=='onehot' else v.cuda()) for k,v in cpu.items()}
   if kind in ['dnabert2','ntv2']:
    h=m(**x)[0] if kind=='dnabert2' else m.esm(**x,return_dict=True).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1).to(h.dtype);out=(h*mask).sum(1)/mask.sum(1)
   elif kind=='rnafm':
    h=m(x['input_ids'],repr_layers=[12],return_contacts=False)['representations'][12];out=torch.stack([h[i,1:len(r['sequence'])+1].mean(0) for i,r in enumerate(records[:B])])
   else:
    out=None
    for net in nets:
     z=net(x['onehot']);out=z if out is None else out+z
    out=out/len(nets)
   return out.cpu().numpy()
 meter.phase='warmup';t=time.time()
 with torch.no_grad():out=forward()
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;assert out.shape[0]==B and np.isfinite(out).all();del out;gc.collect();t,base=meter.begin('inference');steps=0
 with torch.no_grad():
  while steps<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,batch_size=B,unique_inputs=B,processed_inputs=B*steps,repeats=steps);r['inputs_per_second']=B*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];r['model_only_allocated_gib']=S['model_only_postload']['allocated_gib'];S['conditions']=[r];np.save(C/'outputs.npy',out);assert np.array_equal(np.load(C/'outputs.npy'),out);S['output_shape']=list(out.shape);S['finite_and_exact_save_reload']=True
 first=D/'reference_first.npy'
 if not first.exists():np.save(first,out[0]);S['reference_first_created']=True
 else:
  ref=np.load(first);diff=float(np.abs(ref-out[0]).max());S['batch_invariance_max_abs']=diff;assert np.allclose(ref,out[0],atol=1e-4,rtol=1e-4),(kind,diff)
 S['status']='complete';save();print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/two_batch_report.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/two_batch_report.py", "sha256": "a4fa6195788965f1fd6a0599718a24fcc35feebdc681fa10f26bcab070c22277", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,sys,csv
ROOT=Path('/home/hyein/ai-benchmark');kind=sys.argv[1];D=Path(sys.argv[2]);S=json.loads((D/'summary.json').read_text());R=ROOT/('colabfold' if kind=='af2' else 'clair3');out=[]
def a(x=''):out.append(x)
if kind=='clair3':
 a('# Clair3 실제 후보 tensor의 배치 확대 후속 실험');a();a('상태 **'+S['status']+'** · '+D.name);a();a('기존HG003 ONT BAM/GRCh38 chr20:100000~300000/threads4/한GPUworker의공식variant calling을다시실행해실제후보feature tensor 전체를cache했다. 모델입력이나호출결과를바꾸지않는opt-in capture hook을사용했다. CLI옵션이없더라도모델의실제forward 입력을분리해batch를조절할수있음을확인한실험이다. 후보생성과VCF호출은별도단계이며후속시간은CPUcache→GPU nativeforward→CPU확률반환만포함한다.');a();a('pileup과full-alignment는다른입력형태/모델로따로측정한다. 같은feature값은중복제거했고가짜zero row/반복복제row를batch에채우지않았다. 원본candidate좌표는capturehook으로내보내지못했으므로고유생물학적위치수대신고유feature row수를보고한다. cache파일SHA256·원본shape/dtype·선택pool은input_audit.json에있다.');a();a('|모델|고유feature pool|각후보shape|dtype|');a('|---|---|---|---|')
 for k,v in S['input_audit'].items():a(f"|{k}|{v['unique_feature_rows']}|{v['feature_shape']}|{v['dtype']}|")
 a();a('|모델|batch|반복|고유후보|총처리|초|후보/초|모델적재GiB|allocated peak GiB|추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|');a('|---|---|---|---|---|---|---|---|---|---|---|---|---|');flat=[]
 for c in S['conditions']:
  m=c['memory'];base=c['model_only_allocated_gib'];r={'model':c['model'],'batch':c['batch_size'],'repeats':c['repeats'],'unique_candidates':c['unique_candidates'],'processed_candidates':c['processed_candidates'],'seconds':c['seconds'],'candidates_per_second':c['candidates_per_second'],'model_loaded_gib':base,'allocated_peak_gib':m['peak_allocated_gib'],'additional_from_load_gib':m['peak_allocated_gib']-base,'reserved_peak_gib':m['peak_reserved_gib'],'percent_shared_ram':100*c['allocated_fraction_shared_ram'],'gpu_avg':c['gpu_util_avg'],'gpu_max':c['gpu_util_max']};flat.append(r);a('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
 a();a('FP32/TF32=False/seed42, 기존공식ont가중치를고정했다. 각batch는새프로세스로실행한다. 전체파라미터/적재기준값/실제입력dtype·shape/출력shape/표본메모리는각worker summary에있다.');a('원래공식pipeline에서기록한동일후보의출력확률과batch1재추론대조도통과했다(P최대오차4.17e-6,F1.06e-5). native_pipeline_candidate_equivalence.json에저장했다.');a('전체출력finite·저장재로드정확일치·같은첫후보의batch간확률일치(atol/rtol1e-4)를검증했다. 실제BAM호출VCF도별도보존했다. 이단계는새로운variant accuracy평가나학습이아니며기존truth정확도결과를참조한다.');a();a('종료이유: '+json.dumps(S.get('stop_reasons',{}),ensure_ascii=False));a('확보한실제후보pool에서측정한최대batch이며모델절대한계가아니다. GPU사용률은메모리비율과다르다. CUDAallocated/reserved는프로세스allocator,분모는CPU/GPU공유RAM121.688GiB다. RSS와CUDA를더하지않는다. 가용16GiB·swap+2GiB guard,0.5초대기+조회표본간격. RSS/가용량sampled값과CUDApeak를구분한다.');a();a('[공식 Clair3](https://github.com/HKU-BAL/Clair3) · [기본 결과](Clair3.md)')
else:
 a('# AlphaFold2 — ColabFold JAX vmap 배치 후속 실험');a();a('상태 **'+S['status']+'** · '+D.name);a();a('공식CLI가제공하는배치기능이아닌**별도실험구현**이다. 기존ColabFold1.6.2/AlphaFold2 PTM model_1의가중치·forward를사용하고 JAX vmap으로서로독립인실제단백질query축을추가했다. 복합체사슬을batch로세거나다른GPU작업을동시에띄우지않았다. 이번비교는FP32활성값/가중치를고정했다. 앞선BF16벡터화는native대조CA RMSD1.075Å로사전0.5Å검증기준을넘어실패기록을보존했고,수치민감도를확인하기위해별도FP32조건을분리했다. 기본BF16성능수치와직접비교하지않는다. 모델수는고정1개,3recycle(초기pass포함4회),ensemble1,seed42/query,dropout비활성,template/relaxation없음이다. 기본보고서의5모델순차실행과다른조건으로구분한다.');a();a('기존실제단량체4개는각115잔기이며기존MSA를재사용한다. 원래MSA깊이·SHA256은input_audit.json,실제처리msa mask깊이와각feature shape은worker summary에있다. extra-MSA/cluster축은pool내최대shape로zero pad하고마스크도0으로유지했다. 잔기서열을복제하거나추가하지않았다. 큰모델에서vmap이동작하는지부터단일native predict와검증하는pilot이며길이900AA같은다른조건의결과로확대해해석하지않는다.');a();a('|batch/고유query|반복|초|query/초|GPU평균/최대%|프로세스누적JAX peak GiB|실행중sampled bytes-in-use peak GiB|컴파일초|');a('|---|---|---|---|---|---|---|---|');flat=[]
 for w in S['workers']:
  if w['status']!='complete':continue
  c=w['conditions'][0];m=c['jax_memory_stats'];sample=c['execution_sampled_peak_bytes_in_use'];r={'batch':c['batch'],'repeats':c['repeats'],'seconds':c['seconds'],'queries_per_second':c['queries_per_second'],'gpu_avg':c['gpu_util_avg'],'gpu_max':c['gpu_util_max'],'cumulative_process_peak_gib':m['peak_bytes_in_use']/2**30 if 'peak_bytes_in_use' in m else None,'execution_sampled_peak_gib':sample/2**30 if sample is not None else None,'compile_seconds':w['compilation_seconds']};flat.append(r);a('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
 a();a('시간'+'은동기화된4passforward·recycle의FP16반환·prev재입력을포함한다. MSA/featurecache구성·모델로딩·컴파일·warmup·PDB저장·native참조검증은별도시간이다. JAXpeak는프로세스생성후컴파일/warmup포함누적allocator값이며실행구간reset peak라고쓰지않는다. compiled.memory_analysis도별도기록한다. 실행중bytes_in_use/RSS/시스템가용량은0.5초대기+조회sampled값이다. CPU/GPU공유RAM에서RSS와JAX값을더하지않는다.');a();a('단일batch1과원래runner.predict의동일query/seed출력은CA정렬RMSD·pLDDT차이로비교했다. 다른batch도첫query를batch1과비교해벡터화의수치차이를보존했다. 모든좌표/pLDDTfinite·잔기수·PDB의CA수검증을수행했다. 신뢰도와정답구조정확도를혼동하지않으며새독립구조정확도평가는하지않았다. 실제4querypool끝에서종료하므로80/90%메모리최대가능성을검증한실험은아니다.');a();a('[공식 ColabFold](https://github.com/sokrypton/ColabFold) · [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) · [기본 결과](AlphaFold2_ColabFold.md)')
a();a('원본실행: results/'+D.name+'; 명령·소스·가중치출처·inputcache·출력·자원시계열·exit·검증값을보존한다.');(R/'reports/Batch_memory_followup.md').write_text('\n'.join(out)+'\n')
if flat:
 with (R/'reports/batch_memory_followup.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for name in ['STATUS.md','RESULTS.md','MEMORY_RESULTS.md','FOLLOWUP_RESULTS.md']:
 with (ROOT/name).open('a') as f:f.write('\n'+kind+' 전용추가후속 '+S['status']+'; [결과]('+str(R.relative_to(ROOT))+'/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.\n')

qpath=ROOT/'FOLLOWUP_QUEUE.json'
if qpath.exists():
 q=json.loads(qpath.read_text());label='AlphaFold2' if kind=='af2' else 'Clair3';q.setdefault('additional_models',{})[label]={'status':S['status'],'run_id':D.name,'report':str(R.relative_to(ROOT))+'/reports/Batch_memory_followup.md'};q['applicability_notes'][label]='후속 실제 실행 결과가 기존 미실시 기록을 갱신함: '+S['status']+'. CLI 기본 배치와 별도 구현/계측 범위는 새 보고서 참조.'
 if all(q['additional_models'].get(k,{}).get('status')=='complete' for k in ['AlphaFold2','Clair3']):q['status']='all_ten_followups_completed_with_scope_differences'
 qpath.write_text(json.dumps(q,indent=2,ensure_ascii=False))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/two_remaining_launch.py — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scripts/two_remaining_launch.py", "sha256": "a93cb7bf32db3e72a391b9a3f420c4cd040776644a5c1c25850e0ee5b2d970d2", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,subprocess,datetime,shlex,json,time
ROOT=Path('/home/hyein/ai-benchmark');run='candidate_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=ROOT/'clair3/results'/run;D.mkdir();(D.parent/'BATCH_LATEST').write_text(run+'\n');print(run,flush=True);assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(ROOT/'clair3')+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','PYTHONPATH=/work/scripts:/work/source','-e','MPLCONFIGDIR=/work/.cache/matplotlib','local/clair3-spark:9d1dd3f','/work/.venv/bin/python','-u','/work/scripts/batch_followup/capture.py'];(D/'host_capture_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'capture.log').open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
print('CAPTURE_EXIT',code,flush=True);assert code==0
```
<!-- END_RESTORE_FILE -->


</details>
