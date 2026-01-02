import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Speed
ax1.plot(df['n'], df['cpu128_time'], 's-', label='CPU Emulation (binary128)')
ax1.plot(df['n'], df['hy_time'], '^-', label='Hybrid GPU (Ozaki)', linewidth=2)
ax1.plot(df['n'], df['cu64_time'], 'o--', label='Native GPU (binary64)', alpha=0.7)
ax1.set_xscale('log', base=2); ax1.set_yscale('log')
ax1.set_title('Execution Time Comparison'); ax1.set_ylabel('Execution Time (ms)')
ax1.legend(); ax1.grid(True, ls="-", alpha=0.3)

# Precision with Zero-handling
EPS = 1e-38
ax2.axhline(y=1e-15, color='green', ls='--', alpha=0.4, label='binary64 Limit')
ax2.axhline(y=1e-33, color='blue', ls='--', alpha=0.4, label='binary128 Ideal')

ax2.plot(df['n'], df['cpu64_err'], 'gx-', label='Standard binary64 Error', alpha=0.6)
ax2.plot(df['n'], np.maximum(df['cpu128_err'], EPS), 'bs-', label='CPU binary128 Error', alpha=0.6)
ax2.plot(df['n'], np.maximum(df['hy_err'], EPS), 'rx-', markersize=12, label='Our Hybrid GPU (Ozaki)', linewidth=2)

ax2.set_xscale('log', base=2); ax2.set_yscale('log')
ax2.set_title('Accuracy Comparison (RMSE vs FP256 Truth)'); ax2.set_ylabel('RMSE Error')
ax2.set_ylim(1e-40, 1e-10)
ax2.legend(); ax2.grid(True, ls="-", alpha=0.3)

plt.suptitle('Performance and Precision Analysis of High-Precision GEMV', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('comprehensive_analysis.png', dpi=300)
print("\n✅ 'comprehensive_analysis.png' を生成しました。")