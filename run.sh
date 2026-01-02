#!/bin/bash
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo "=== 🛡️ Building (CUDA Path: $CUDA_PATH) ==="

rm -f *.o trust_bench
# host_part は CUDA ヘッダーに依存しないようにコンパイル
g++ -O3 -fopenmp -fPIC -c hybrid_reconstruction.cpp -o host_part.o

# device_part は nvcc で標準的な CUDA ビルド
nvcc -arch=sm_75 -O3 -Xcompiler "-fopenmp -fPIC" -I$CUDA_PATH/include -c hybrid_benchmark.cu -o device_part.o

# リンク
nvcc -arch=sm_75 host_part.o device_part.o -o trust_bench -lcublas -lquadmath -lgomp

if [ $? -eq 0 ]; then
    ./trust_bench
else
    echo "=== ❌ Build Failed ==="
fi