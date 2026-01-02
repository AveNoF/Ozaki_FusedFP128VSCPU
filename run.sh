#!/bin/bash
echo "=== 🛠️  Improved Ozaki Hybrid Build ==="
g++ -O3 -fopenmp -fPIC -c hybrid_reconstruction.cpp -o host_part.o
nvcc -arch=sm_86 -O3 -Xcompiler "-fopenmp -fPIC" -c hybrid_benchmark.cu -o device_part.o
nvcc -arch=sm_86 host_part.o device_part.o -o hybrid_bench -lcublas -lquadmath -lmpfr -lgmp -lgomp

if [ $? -eq 0 ]; then
    echo "=== 🚀 Running Benchmark (N=32 to 16384) ==="
    ./hybrid_bench
    echo "=== 📊 Generating Graph ==="
    python3 plot_results.py
else
    echo "=== ❌ Build Failed ==="
    exit 1
fi