# Reproducible runtime for a read-only DynamicGSG checkout.
# Build with the Dynamic-GSG-Baseline checkout as the context. The final image
# contains dependencies and compiled extensions, but no baseline algorithm source.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS python-base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git python3-dev python3-pip libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
       torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
       --index-url https://download.pytorch.org/whl/cu121

FROM python-base AS extension-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake libglm-dev ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir ninja
COPY submodules/diff-gaussian-rasterization-w-depth /tmp/diff-gaussian-rasterization-w-depth
COPY submodules/GroundingDINO /tmp/provided-GroundingDINO
RUN mkdir -p /tmp/diff-gaussian-rasterization-w-depth/third_party \
    && ln -s /usr/include /tmp/diff-gaussian-rasterization-w-depth/third_party/glm
# The packaged baseline accidentally omits GroundingDINO's model sources.  Its
# retained setup.py and inference.py byte-match this official upstream commit;
# verify that identity before compiling the complete pinned source.
RUN git clone --filter=blob:none https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO \
    && git -C /tmp/GroundingDINO checkout 856dde20aee659246248e20734ef9ba5214f5e44 \
    && cmp /tmp/provided-GroundingDINO/setup.py /tmp/GroundingDINO/setup.py \
    && cmp /tmp/provided-GroundingDINO/groundingdino/util/inference.py \
           /tmp/GroundingDINO/groundingdino/util/inference.py
RUN python3 -m pip wheel --no-cache-dir --no-build-isolation \
        /tmp/diff-gaussian-rasterization-w-depth -w /wheels \
    && python3 -m pip wheel --no-cache-dir --no-build-isolation --no-deps \
        /tmp/GroundingDINO -w /wheels

FROM python-base
COPY --from=extension-builder /wheels /wheels
RUN python3 -m pip install --no-cache-dir \
       numpy==1.24.2 scipy==1.14.1 Pillow imageio matplotlib kornia natsort pyyaml \
       wandb lpips open3d==0.16.0 torchmetrics cyclonedds pytorch-msssim \
       plyfile==0.8.1 faiss-cpu==1.8.0 openai open_clip_torch==2.26.1 \
       urllib3==2.2.3 supervision==0.22.0 httpx==0.27.0 scikit-image \
       transformers==4.51.3 ultralytics fairscale jaxtyping accelerate \
       addict yapf timm pycocotools opencv-python==4.11.0.86 \
       pydantic==2.10.6 sentencepiece gradio fastapi uvicorn \
    && python3 -m pip install --no-cache-dir --no-deps /wheels/*.whl \
    && python3 -m pip install --no-cache-dir --no-deps \
       git+https://github.com/xinyu1205/recognize-anything.git@7cb804a8609e9f4b1a50b7f31436d2df40bb9481 \
       git+https://github.com/NVlabs/describe-anything.git@153ad3d33c29324e9197f565547c6bc8500da02d \
    && rm -rf /wheels
RUN apt-get update && apt-get install -y --no-install-recommends xvfb xauth \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /baseline
