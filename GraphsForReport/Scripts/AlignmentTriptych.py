import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Base directory for experiment results
ROOT = Path(__file__).resolve().parents[2]
BASE_RESULTS = ROOT / "experiment_results" / "AI" / "BCE" / "2"
GCN_MAX_PLACEHOLDER = 16  # numerical placeholder for max pooling (\infty)


def _load_alignment(file_path):
    """Return (mean, err) for alignment index in the given JSON file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: could not read {file_path}: {exc}")
        return None
    mean = data.get("alignment index")
    ci = data.get("alignment index CI")
    if mean is None or ci is None or len(ci) < 2:
        print(f"Warning: alignment data missing in {file_path}")
        return None
    err = (ci[1] - ci[0]) / 2
    return mean, err


def gather_spm(arch):
    """Collect (p, mean, err) tuples for SPM mode for a given architecture."""
    arch_dir = BASE_RESULTS / "SPM" / "specify" / arch / "mean"
    data = []
    for p_dir in sorted(arch_dir.glob("p_lp=*"), key=lambda d: int(d.name.split("=")[1])):
        p = int(p_dir.name.split("=")[1])
        files = sorted(p_dir.glob("*.json"))
        file = files[0] if files else None
        if not file:
            print(f"Warning: no results file in {p_dir}")
            continue
        result = _load_alignment(file)
        if result is None:
            continue
        mean, err = result
        data.append((p, mean, err))
    return data


def gather_gcn(arch):
    """Collect (p, mean, err) tuples for GCN mode for a given architecture."""
    arch_dir = BASE_RESULTS / "GCN" / "specify" / arch
    data = []
    for p_dir in sorted(arch_dir.iterdir()):
        name = p_dir.name
        if name == "max":
            p = GCN_MAX_PLACEHOLDER
        elif name.startswith("p_gm="):
            p = float(name.split("=")[1])
        else:
            continue
        files = sorted(p_dir.glob("*.json"))
        file = files[0] if files else None
        if not file:
            print(f"Warning: no results file in {p_dir}")
            continue
        result = _load_alignment(file)
        if result is None:
            continue
        mean, err = result
        data.append((p, mean, err))
    return data


def main():
    architectures = ["12, 12, 6", "12, 12, 12", "12, 12, 18"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120, sharey=True)

    for ax, arch in zip(axes, architectures):
        spm_data = gather_spm(arch)
        gcn_data = gather_gcn(arch)

        if not spm_data and not gcn_data:
            raise RuntimeError(f"No valid points found for architecture {arch}")

        if spm_data:
            spm_data.sort(key=lambda t: t[0])
            p, mean, err = zip(*spm_data)
            ax.errorbar(p, mean, yerr=err, fmt='-o', capsize=5, label='SPM')

        if gcn_data:
            gcn_data.sort(key=lambda t: t[0])
            p, mean, err = zip(*gcn_data)
            ax.errorbar(p, mean, yerr=err, fmt='-o', capsize=5, label='GCN')

        xticks = sorted({t[0] for t in spm_data} | {t[0] for t in gcn_data})
        xlabels = [r'$\infty$' if x == GCN_MAX_PLACEHOLDER else str(int(x)) for x in xticks]

        ax.set_xscale('log', base=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel('p')
        ax.set_title(arch)
        ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel('Alignment Index')

    fig.tight_layout()
    out_path = Path(__file__).with_name('alignment_triptych.png')
    plt.savefig(out_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(f"Error: {err}")
        sys.exit(1)
