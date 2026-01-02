import pandas as pd
import matplotlib.pyplot as plt

# CSVデータの読み込み
df = pd.read_csv('results.csv')

# 2画面構成に変更（Splitを削除）
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- 1. 速度比較 (Y軸を線形スケールに変更) ---
axes[0].plot(df['N'], df['CPU128_T'], 's-', label='CPU FP128 (Ryzen 5950X)', color='#1f77b4', linewidth=2)
axes[0].plot(df['N'], df['Hybrid_T'], '^-', label='Improved Ozaki (RTX 3090 Hybrid)', color='#ff7f0e', linewidth=2)
# cuBLASは速すぎて線形だと0に見えるため、比較対象としてあえて薄く表示
axes[0].plot(df['N'], df['cuBLAS_T'], 'o--', label='cuBLAS (Standard FP64)', color='#2ca02c', alpha=0.3)

# X軸はNが倍々なのでlog2のままが綺麗ですが、リクエストに合わせて調整可能
axes[0].set_xscale('log', base=2) 
axes[0].set_yscale('linear') # ここを線形に変更
axes[0].set_xlabel('Matrix Size N', fontsize=12)
axes[0].set_ylabel('Execution Time (ms)', fontsize=12)
axes[0].set_title('Performance Comparison (Linear Scale)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, which="both", ls="-", alpha=0.3)

# --- 2. 精度比較 (RMSE) ---
axes[1].plot(df['N'], df['cuBLAS_R'], 'o-', label='cuBLAS (Standard FP64)', color='#2ca02c')
axes[1].plot(df['N'], df['CPU128_R'], 's--', label='CPU FP128', color='#1f77b4', alpha=0.7)
axes[1].plot(df['N'], df['Hybrid_R'], 'x-', label='Improved Ozaki (Hybrid)', color='#ff7f0e', markersize=8)

axes[1].set_xscale('log', base=2)
axes[1].set_yscale('log') # 精度は桁が違いすぎるので対数推奨
axes[1].set_xlabel('Matrix Size N', fontsize=12)
axes[1].set_ylabel('RMSE vs FP256 Truth', fontsize=12)
axes[1].set_title('Precision: Hybrid vs Standard', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.savefig('linear_performance_result.png')
print("グラフを 'linear_performance_result.png' として保存しました。")