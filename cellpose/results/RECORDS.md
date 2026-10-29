# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-8ad10094229e

원래 경로: `cellpose/results/environment_packages.txt` · SHA256: `e98e9dbc88d2c40c1d60ec56dde406c5269e5f1dd6a5ce216bfb5c9f10163ab0`

<details>
<summary>environment_packages.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/results/environment_packages.txt", "sha256": "e98e9dbc88d2c40c1d60ec56dde406c5269e5f1dd6a5ce216bfb5c9f10163ab0", "encoding": "utf-8", "added_final_newline": false} -->
```txt

=============
== PyTorch ==
=============

NVIDIA Release 26.06 (build 337426144)
PyTorch Version 2.13.0a0+8145d63
Container image Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2014-2024 Facebook Inc.
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2015      Google Inc.
Copyright (c) 2015      Yangqing Jia
Copyright (c) 2013-2016 The Caffe contributors
All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

GOVERNING TERMS: The software and materials are governed by the NVIDIA Software License Agreement
(found at https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the Product-Specific Terms for NVIDIA AI Products
(found at https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
   insufficient for PyTorch.  NVIDIA recommends the use of the following flags:
   docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...

absl-py==2.4.0
aiohappyeyeballs==2.6.2
aiohttp==3.14.1
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.13.0
apex @ file:///opt/pytorch/apex/dist/apex-0.1-cp312-cp312-linux_aarch64.whl#sha256=da723b7848638dfc7e1e58b8e27bd37237b093e57700154de8ed8dc61d9d10e6
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
asttokens==3.0.1
astunparse==1.6.3
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
black==26.5.1
bleach==6.4.0
cellpose==4.2.1
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
cmake==3.31.6
comm==0.2.3
contourpy==1.3.3
cuda-bindings==13.3.1
cuda-core==1.0.1
cuda-pathfinder==1.5.5
cuda-python==13.3.1
cuda-tile==1.4.0
cycler==0.12.1
Cython==3.2.5
datasets==5.0.0
debugpy==1.8.21
decorator==5.3.1
defusedxml==0.7.1
dill==0.4.1
dllist==2.0.0
docstring_parser==0.18.0
einops==0.8.2
executing==2.2.1
fastjsonschema==2.21.2
fastremap==1.21.0
filelock==3.29.1
fill_voids==2.1.2
flash_attn @ file:///opt/transfer/flash_attn-2.7.4.post1%2Bgit5231d95fe1.54157505-cp312-cp312-linux_aarch64.whl#sha256=c582b983568ec7175eba847adef22b6b380efbe6c2ca6c4103ed778ed014613f
fonttools==4.63.0
fqdn==1.5.1
frozenlist==1.8.0
fsspec==2026.4.0
gast==0.7.0
gitdb==4.0.12
GitPython==3.1.50
grpcio==1.81.0
grpcio-tools==1.81.0
h11==0.16.0
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.18.0
idna==3.18
imagecodecs==2026.10.10
ImageIO==2.38.1
importlib_metadata==9.0.0
ipykernel==7.2.0
ipython==9.14.1
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
isort==8.0.1
jedi==0.20.0
Jinja2==3.1.6
json5==0.14.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_client==8.9.0
jupyter_core==5.9.1
jupyter_server==2.19.0
jupyter_server_terminals==0.5.4
jupyterlab==4.5.8
jupyterlab_code_formatter==3.0.3
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_tensorboard_pro @ file:///builder/jupyterlab_tensorboard_pro-4.0.0-py2.py3-none-any.whl#sha256=b639e0d57174368cba55246fb2b225bc5eb99065b2f7e04366d10fdb8281f660
jupytext==1.19.3
kiwisolver==1.5.0
lark==1.3.1
lazy-loader==0.6
llvmlite==0.46.0
makefun==1.16.0
Markdown==3.10.2
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
matplotlib-inline==0.2.2
mdit-py-plugins==0.6.1
mdurl==0.1.2
mistune==3.2.1
ml_dtypes==0.5.4
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.19
mypy_extensions==1.1.0
natsort==8.4.0
nbclient==0.11.0
nbconvert==7.17.1
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.6.1
ninja==1.13.0
notebook==7.5.7
notebook_shim==0.2.4
numba==0.64.0
numpy==2.1.0
nvdlfw_inspect==0.2.2
nvidia-cutlass-dsl==4.5.1
nvidia-cutlass-dsl-libs-base==4.5.1
nvidia-dali-cuda130==2.1.1
nvidia-libnvcomp-cu13==5.1.0.21
nvidia-ml-py==13.610.43
nvidia-modelopt @ file:///tmp/modelopt/nvidia_modelopt-0.44.0-py3-none-any.whl#sha256=9b54a853dfda161db97a0dfce4d7c24269d1d19966b5e2026094186af897f74d
nvidia-nvimgcodec-cu13==0.7.0.49
nvidia-nvjpeg==13.2.0.21
nvidia-nvjpeg2k-cu13==0.10.0.49
nvidia-nvtiff-cu13==0.7.0.79
nvidia-resiliency-ext @ file:///tmp/nvrx/nvidia_resiliency_ext-0.6.0-cp312-cp312-manylinux_2_39_aarch64.whl#sha256=a71c1a1650b32c72f7fdf7d6f73b4d275961ce5d68bd68efaff9e0d7099031cb
nvtx==0.2.15
omegaconf==2.3.0
onnx @ file:///opt/pytorch/pytorch/third_party/onnx
onnx-ir==0.2.1
onnxscript==0.7.0
opencv-python-headless==5.0.0.93
optree==0.19.1
packaging==26.0
pandas==3.0.3
pandocfilters==1.5.1
parso==0.8.7
pathspec==1.1.1
pexpect==4.9.0
pillow==12.2.0
platformdirs==4.10.0
polygraphy==0.50.3
prometheus_client==0.25.0
prompt_toolkit==3.0.52
propcache==0.5.2
protobuf==6.33.6
psutil==7.2.2
ptyprocess==0.7.0
PuLP==3.3.2
pure_eval==0.2.3
pyarrow==24.0.0
pybind11==3.0.4
pycocotools @ git+https://github.com/nvidia/cocoapi.git@08350a157b41f0755a536441b5abec89ab934c5b#subdirectory=PythonAPI
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.17.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
python-json-logger==4.1.0
python_hostlist==2.3.0
pytokens==0.4.1
PyYAML==6.0.1
pyzmq==27.1.0
referencing==0.37.0
regex==2026.5.9
requests==2.34.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
roifile==2026.9.22
rpds-py==2026.5.1
safetensors==0.8.0
scikit-image==0.25.2
scipy==1.17.1
segment-anything==1.0
Send2Trash==2.1.0
sentry-sdk==2.62.0
setuptools==81.0.0
shellingham==1.5.4
six==1.16.0
smmap==5.0.3
soupsieve==2.8.4
stack-data==0.6.3
sympy==1.14.0
tabulate==0.10.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorrt @ file:///workspace/TensorRT-11.0.0.114/python/tensorrt-11.0.0.114-cp312-none-linux_aarch64.whl#sha256=b02d6572dbd05c0aa215a5772816eee18c17d3f3324c7094d8b932d3e92f4689
terminado==0.18.1
tifffile==2026.10.15
tinycss2==1.5.1
tokenizers==0.23.1
torch @ file:///opt/transfer/torch-2.13.0a0%2B8145d630e8.nv26.6.54250401-cp312-cp312-linux_aarch64.whl#sha256=8a7efd12b08642a772ec7fd4dc2ab8ea88dc964323e0aca3a0f329a662193cfa
torch_tensorrt @ file:///opt/pytorch/torch_tensorrt/dist/torch_tensorrt-2.13.0a0-cp312-cp312-linux_aarch64.whl#sha256=2f42d582051a1d3948e74c4f3dc8447599594d6a96fae38d714ab84e9a6b73b7
torchao @ file:///opt/pytorch/ao
torchdata==0.11.0
torchtitan @ file:///opt/pytorch/torchtitan
torchvision @ file:///opt/transfer/torchvision-0.27.0a0%2B499ca510.nv26.6.54250401-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=8c9341542e4b370b041bb8a0be1722b202abcf07fba94d762b2835e114fee7a5
tornado==6.5.7
tqdm==4.68.2
traitlets==5.15.1
transformer_engine @ file:///workspace/TransformerEngine
triton @ file:///tmp/dist/triton-3.7.0%2Bgitb7fa781f.nv26.6-cp312-cp312-linux_aarch64.whl#sha256=ac76a8804e3c7883c603679839f5db8b694366d50c8d2ab929fe309c63261827
triton_kernels @ file:///tmp/dist/triton_kernels-1.0.0%2Bgitb7fa781f.nv26.6-py3-none-any.whl#sha256=517938a42b240c5ad4bf5dce7720512e29cd408b2d58e2367f9f5448bdf13737
typeguard==4.5.2
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
tyro==1.0.13
tzdata==2026.2
uri-template==1.3.0
urllib3==2.7.0
wandb==0.27.2
wcwidth==0.8.1
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
Werkzeug==3.1.8
wheel==0.47.0
xxhash==3.7.0
yarl==1.24.2
zipp==4.1.0
```
<!-- END_RESTORE_FILE -->


</details>
