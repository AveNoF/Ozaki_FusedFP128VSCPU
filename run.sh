#!/bin/bash
echo "=== 🛡️ Building Reliable Engine (N <= 4096) ==="
rm -f *.o trust_bench
g++ -O3 -fopenmp -fPIC -c hybrid_reconstruction.cpp -o host_part.o
nvcc -arch=sm_75 -O3 -Xcompiler "-fopenmp -fPIC" -c hybrid_benchmark.cu -o device_part.o
nvcc -arch=sm_75 host_part.o device_part.o -o trust_bench -lcublas -lquadmath -lgomp

if [ $? -eq 0 ]; then
    ./trust_bench
else
    echo "=== ❌ Build Failed ==="
fi