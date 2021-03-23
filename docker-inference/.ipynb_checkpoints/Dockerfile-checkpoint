ARG REGION=us-east-1

# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-inference:1.7.1-cpu-py36-ubuntu18.04

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
#     && pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
#     && pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install MinkowskiEngine \
    && pip3 install git+https://github.com/mit-han-lab/torchsparse.git@f79df704e2fb3ea912c31d57e910ea0edba03da4 \
    && rm -rf /root/.cache

COPY torch-points3d/pyproject.toml /opt/ml/code/pyproject.toml
COPY torch-points3d/torch_points3d/ /opt/ml/code/torch_points3d/
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
RUN pip install multi-model-server sagemaker-inference plyfile

#Sagemaker environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]
