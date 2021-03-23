ARG REGION=us-east-1

# FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:1.7.1-gpu-py36-cu110-ubuntu18.04
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:1.8.0-gpu-py36-cu111-ubuntu18.04

ARG CUDA_HOME=/usr/local/cuda
    
#Upgrade the OS
RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends\
    libffi-dev libssl-dev build-essential libopenblas-dev libsparsehash-dev\
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Install dependant libraries for torch-points-3d
RUN python3 -m pip install -U pip \
    && pip3 install setuptools>=41.0.0 \
    && pip3 install torch==1.7.1 torchvision==0.8.2 \
#     && pip3 install torch==1.8.0 torchvision==0.9.0 \
    && pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--force_cuda" --install-option="--cuda_home=/usr/local/cuda-11.1" \
#     && pip3 install MinkowskiEngine --install-option="--force_cuda" \
#     && pip3 install git+https://github.com/mit-han-lab/torchsparse.git@f79df704e2fb3ea912c31d57e910ea0edba03da4 -v \
#     && pip3 install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html \
#     && pip3 install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu111.html \
    && pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git \
    && pip3 install pycuda\
    && rm -rf /root/.cache 

#Install and fix issues with torch-points-3d
COPY torch-points3d/. /opt/ml/code

#RUN pip uninstall torch-scatter torch-sparse torch-cluster torch-points-kernels -y
RUN rm -rf ~/.cache/pip
RUN cd /opt/ml/code && pip3 install . 

#Sagemaker environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

#Sagemaker training script and dir
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM sagemaker_train.py
