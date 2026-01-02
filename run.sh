#!/bin/bash
echo "=== 🚀 Building Fused-FP128 Benchmark ==="
rm -f *.o trust_bench results.csv
nvcc -arch=sm_75 -O3 -Xcompiler "-fopenmp -fPIC" -c hybrid_reconstruction.cpp -o host_part.o
nvcc -arch=sm_75 -O3 -Xcompiler "-fopenmp -fPIC" -c hybrid_benchmark.cu -o device_part.o
nvcc -arch=sm_75 host_part.o device_part.o -o trust_bench -lcublas -lquadmath -lgomp

echo "=== 🏃 Running Benchmark ==="
./trust_bench

echo "=== 📊 Generating Graph ==="
python3 plot_results.py