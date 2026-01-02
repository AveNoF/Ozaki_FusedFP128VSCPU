#!/bin/bash
# デフォルト値
N_START=${1:-128}
N_MAX=${2:-4096}
STEP=${3:-2}

echo "=== 🚀 Building (Start=$N_START, Max=$N_MAX, Step=$STEP) ==="
rm -f *.o trust_bench results.csv comprehensive_analysis.png

# RTX 2060 (sm_75) 向けにビルド
nvcc -arch=sm_75 -O3 -Xcompiler "-fopenmp -fPIC" -c hybrid_reconstruction.cpp -o host_part.o
nvcc -arch=sm_75 -O3 -Xcompiler "-fopenmp -fPIC" -c hybrid_benchmark.cu -o device_part.o
nvcc -arch=sm_75 host_part.o device_part.o -o trust_bench -lcublas -lquadmath -lgomp -lmpfr -lgmp

if [ $? -eq 0 ]; then
    # ./trust_bench [最小N] [最大N] [倍率]
    ./trust_bench $N_START $N_MAX $STEP
    python3 plot_results.py
else
    echo "=== ❌ Build Failed ==="
fi