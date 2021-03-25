# SageMaker PyTorch image
FROM ubuntu:18.04

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Setup points3d inference
RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends\
    ca-certificates \
    openjdk-8-jdk-headless \
    curl \
    vim \
    libffi-dev libssl-dev build-essential libopenblas-dev libsparsehash-dev\
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Install dependant libraries for torch-points-3d
RUN python3 -m pip install -U pip \
    && pip3 install multi-model-server sagemaker-inference plyfile retrying \
    && pip3 install setuptools>=41.0.0 \
    && pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install MinkowskiEngine \
    && pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git \
#     && pip3 install git+https://github.com/mit-han-lab/torchsparse.git@f79df704e2fb3ea912c31d57e910ea0edba03da4 \
    && rm -rf /root/.cache

COPY torch-points3d/pyproject.toml /opt/ml/code/pyproject.toml
COPY torch-points3d/torch_points3d/ /opt/ml/code/torch_points3d/
COPY torch-points3d/README.md /opt/ml/code/README.md
# COPY poetry.lock poetry.lock

RUN cd /opt/ml/code/ && pip3 install . && rm -rf /root/.cache

# COPY poetry.lock poetry.lock

# RUN pip install poetry
# RUN poetry config virtualenvs.create false
# RUN poetry install

# Copy entrypoint script to the image
COPY docker-inference/dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py
RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY docker-inference/model_handler.py /home/model-server/model_handler.py

#Sagemaker environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Define an entrypoint script for the docker image
ENTRYPOINT ["python3", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]
