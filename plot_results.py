import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Performance
ax1.plot(df['n'], df['cpu_time'], 's-', label='CPU Naive (FP128)')
ax1.plot(df['n'], df['hy_time'], '^-', label='Fused-Ozaki (RTX 2060)')
ax1.plot(df['n'], df['cu_time'], 'o--', label='cuBLAS (FP64)', alpha=0.5)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_title('Performance Comparison (Log-Log Scale)')
ax1.set_xlabel('Matrix Size N')
ax1.set_ylabel('Execution Time (ms)')
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Plot 2: Precision
ax2.plot(df['n'], [1e-15]*len(df), 'o--', label='cuBLAS (Standard FP64)')
ax2.plot(df['n'], [1e-33]*len(df), 's--', label='CPU FP128 (Ideal)')
ax2.plot(df['n'], df['hy_err'], 'x-', label='Fused-Ozaki (Hybrid)', markersize=10)
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_title('Precision: Fused-Ozaki vs Standard')
ax2.set_xlabel('Matrix Size N')
ax2.set_ylabel('RMSE')
ax2.set_ylim(1e-35, 1e-10)
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('victorious_result.png')
print("Graph saved as 'victorious_result.png'")