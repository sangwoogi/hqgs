# CUDA 12.8 + cuDNN + (nvcc 포함) 개발용 이미지
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# --- System deps (빌드/런타임 + OpenCV(cv2)용) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    wget \
    ca-certificates \
    # OpenCV 런타임에서 자주 필요한 것들
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# (선택) nvcc 경로 확인용. 빌드 중 문제 생기면 로그로 바로 확인 가능.
RUN nvcc --version

# --- Python deps ---
# 베이스 이미지에 이미 conda가 들어있는 경우가 많지만,
# 여기서는 가장 단순하게 "현재 파이썬 환경"에 pip로 설치하는 방식.
RUN python -m pip install --upgrade pip setuptools wheel

# 프로젝트 소스 복사
COPY . /workspace

# 일반 파이썬 패키지 (필요시 추가)
# opencv-python 대신 opencv-python-headless를 권장(서버/컨테이너에서 GUI 의존성 줄임)
RUN python -m pip install --no-cache-dir \
    plyfile \
    tqdm \
    opencv-python-headless

# --- Build CUDA extensions (editable installs) ---
RUN python -m pip install --no-cache-dir --no-build-isolation -e submodules/diff-gaussian-rasterization && \
    python -m pip install --no-cache-dir --no-build-isolation -e submodules/simple-knn

# (선택) 컨테이너 진입 시 기본 쉘
CMD ["/bin/bash"]
