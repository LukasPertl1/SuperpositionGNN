import numpy as np
import matplotlib.pyplot as plt

# Publication-quality settings similar to other scripts
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.3
})

def plot_alignment_superposition(
    p_values,
    alignment_series,
    alignment_errors,
    superposition_series,
    superposition_errors,
    alignment_labels=None,
    superposition_labels=None,
    output_file="figure.pdf",
):
    """Plot Alignment and Superposition indices versus *p* on dual y-axes.

    Parameters
    ----------
    p_values : sequence of float
        Numeric *p* values; the final entry is labeled as infinity on the x-axis.
    alignment_series : list of sequences
        Each sequence contains Alignment Index values for one line.
    alignment_errors : list of sequences
        95% confidence interval half-widths for ``alignment_series``.
    superposition_series : list of sequences
        Each sequence contains Superposition Index values for one line.
    superposition_errors : list of sequences
        95% confidence interval half-widths for ``superposition_series``.
    alignment_labels : list of str, optional
        Labels for Alignment Index lines.
    superposition_labels : list of str, optional
        Labels for Superposition Index lines.
    output_file : str, optional
        Destination for saving the figure in PDF format.
    """
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5), dpi=300)

    # Plot Alignment Index on primary axis
    if alignment_labels is None:
        alignment_labels = [f"Alignment {i+1}" for i in range(len(alignment_series))]
    for data, err, label in zip(alignment_series, alignment_errors, alignment_labels):
        ax1.errorbar(p_values, data, yerr=err, label=label, marker="o", linewidth=1.5)
    ax1.set_xlabel("p")
    ax1.set_ylabel("Alignment Index")
    ax1.set_xscale("log", base=2)

    # Secondary axis for Superposition Index
    ax2 = ax1.twinx()
    if superposition_labels is None:
        superposition_labels = [f"Superposition {i+1}" for i in range(len(superposition_series))]
    for data, err, label in zip(superposition_series, superposition_errors, superposition_labels):
        ax2.errorbar(p_values, data, yerr=err, label=label, marker="s", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Superposition Index")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    # x-axis ticks: label last value as infinity
    ax1.set_xticks(p_values)
    xticklabels = [str(p) for p in p_values]
    if xticklabels:
        xticklabels[-1] = r"$\infty$"
    ax1.set_xticklabels(xticklabels)

    ax1.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, format="pdf")
    plt.show()

if __name__ == "__main__":
    # Example data; replace with experimental results as needed
    p = [1, 2, 4, 8, 16]

    alignment_data = [
        [0.10, 0.15, 0.20, 0.25, 0.30],
    ]
    alignment_err = [
        [0.02, 0.02, 0.03, 0.02, 0.04],
    ]

    superposition_data = [
        [0.90, 0.85, 0.80, 0.78, 0.75],
    ]
    superposition_err = [
        [0.03, 0.03, 0.02, 0.02, 0.01],
    ]

    plot_alignment_superposition(
        p,
        alignment_data,
        alignment_err,
        superposition_data,
        superposition_err,
        alignment_labels=["Alignment"],
        superposition_labels=["Superposition"],
        output_file="alignment_superposition.pdf",
    )
