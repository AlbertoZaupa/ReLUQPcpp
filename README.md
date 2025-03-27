# CUDA-ADMM Docker Instructions  

## Building the Docker Image  
To build the Docker image, run:  
```sh
docker build -t cuda-admm .
```

## Running the Docker Container  
To start the container with GPU support and mount the `results` directory, use:  
```sh
docker run --gpus all -it --rm -v ./results:/shared cuda-admm bash
```

## Running Benchmarks Inside the Container  

Once inside the container, you can run the following benchmarks:  

### Quadrotor Stabilization Benchmark  
Run the following command:  
```sh
python3 quadrotor_stabilization.py --solver SOLVER --plot PLOT
```  
- `SOLVER` specifies the solver. Must be one of:  
  - `cppsolver`  
  - `reluqp`  
  - `reluqp_warm`  
  - `osqp`  
- `PLOT` determines whether to generate plots of the system's response:  
  - `0`: No plots  
  - Any other integer: Generate plots  

### Random QPs Benchmark  
To run the random QPs benchmark, execute:  
```sh
python3 random_qps.py
```
