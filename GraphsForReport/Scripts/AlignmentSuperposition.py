import numpy as np
import matplotlib.pyplot as plt

# Data from Script 1
p_values = [1, 2, 4, 8, 16]
alignment_data = [0.665, 0.883, 0.883, 0.812, 0.890]
alignment_err = [0.01, 0.02, 0.025, 0.023, 0.04]
superposition_data = [1.86, 1.90, 1.78, 1.72, 1.30]
superposition_err = [0.05, 0.05, 0.06, 0.04, 0.05]

# Plot with log x-axis and black color
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

# Alignment on primary axis
ax.errorbar(
    p_values,
    alignment_data,
    yerr=alignment_err,
    fmt='-o',
    capsize=5,
    color='black',
    markerfacecolor='black',
    label='Alignment'
)

# Superposition on secondary axis
ax2 = ax.twinx()
ax2.errorbar(
    p_values,
    superposition_data,
    yerr=superposition_err,
    fmt='--s',
    capsize=5,
    color='black',
    markerfacecolor='black',
    label='Superposition'
)

# Log scale for x-axis (base 2)
ax.set_xscale('log', base=2)

# Labels and Title
ax.set_xlabel('p')
ax.set_ylabel('Alignment Index')
ax2.set_ylabel('Superposition Index')
plt.title('Alignment and Superposition vs. p')

# Legend
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='best')

# X-axis ticks with infinity label
ax.set_xticks(p_values)
ax.set_xticklabels([str(p) for p in p_values[:-1]] + [r'$\infty$'])

# Grid
ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.3)

fig.tight_layout()
plt.show()