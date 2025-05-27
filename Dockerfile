# Use the CUDA base image with Ubuntu
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3.10 \
    python3-pip 
    
RUN rm -rf /var/lib/apt/lists/*

# Manually add Python to the PATH (if needed)
ENV PATH="/usr/bin:$PATH"

# Ensure python3 and pip are correctly linked
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install numpy
RUN pip install cython
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install scipy osqp cvxpy matplotlib
RUN pip install tqdm

# Set up work directory
WORKDIR /workspace

# Copy the C++ source code into the container
COPY . .

RUN mkdir /shared

# Build the C++ code
RUN mkdir build && cd build && cmake .. && make
RUN python3 setup.py build_ext --inplace