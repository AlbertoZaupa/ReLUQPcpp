# CUDA-ADMM Docker Instructions  

This repository contains the ReLU-QP solver proposed in "Improved GPU acceleration for embedded MPC" by Alberto Zaupa, Pedro Roque, Zachary Manchester and Mikael Johansson. The solver is implemented in C++ and CUDA, and is compared against [OSQP](https://osqp.org/) and [Relu-QP](https://github.com/RoboticExplorationLab/ReLUQP-py). The repository also contains a benchmark for the quadrotor stabilization problem as well as for several random QPs.

## Setup instructions

This setup intructions assume that the user has an CUDA-capable GPU installed in the system, along with the correct drivers and the CUDA Toolkit. The code was tested with CUDA 11.6 on machines with Ubuntu 24.04 and Ubuntu 22.04, using Intel CPUs paired with an RTX 3060M and RTX 5000 Ada Gen GPUs, respectively. The solver was also tested running natively on an Nvidia Jetson Xavier 16GB.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

After installing the NVIDIA Container Toolkit, make sure to configure the docker runtime and restart the Docker service by running:  
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Building the Docker Image  
To build the Docker image, run:  
```bash
docker build -t cuda-admm .
```

## Running the Docker Container  
To start the container with GPU support and mount the `results` directory, use:  
```bash
docker run --gpus all -it --rm -v ./examples/results:/shared cuda-admm bash
```

## Running Benchmarks Inside the Container  

Once inside the container, you can run the following benchmarks:  

### Quadrotor Stabilization Benchmark  
Run the following command:  
```bash
python3 quadrotor_stabilization.py --solver SOLVER --plot PLOT
```  

Where:
- `SOLVER` specifies the solver. Must be one of:  
  - `cppsolver`  
  - `reluqp`  
  - `reluqp_warm`  
  - `osqp`  
- `PLOT` determines whether to generate plots of the system's response:  
  - `0`: No plots  
  - Any other integer: Generate plots  

Example:
```bash
python3 quadrotor_stabilization.py --solver cppsolver --plot 1
```

### Random QPs Benchmark  
To run the random QPs benchmark, execute:  
```bash
python3 random_qps.py
```

### Results

The results of the benchmarks will be stored in the `results` directory. The quadrotor stabilization benchmark will generate plots of the system's response, while the random QPs benchmark will generate a CSV file with the data.
