import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_aggregate(pt_path):
    """Load the pt file, sum the logical_count along the sample dimension, and obtain [layers, experts]."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    logical_count = data["logical_count"]  # [samples, layers, experts]
    # Sum along the sample dimension
    aggregated = logical_count.sum(dim=0)  # [layers, experts]
    return aggregated.numpy()


def main():
    decode_path = "./decode/xxx.pt"
    prefill_path = "./prefill/xxx.pt"

    decode_matrix = load_and_aggregate(decode_path)  # [61, 256]
    prefill_matrix = load_and_aggregate(prefill_path)  # [61, 256]

    # Using the same color scale facilitates comparison
    vmin = min(decode_matrix.min(), prefill_matrix.min())
    vmax = max(decode_matrix.max(), prefill_matrix.max())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    im1 = ax1.imshow(prefill_matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Expert Index", fontsize=12)
    ax1.set_ylabel("Layer Index", fontsize=12)
    ax1.set_title("Expert Distribution Statistics (Prefill)", fontsize=14)
    ax1.set_xticks(np.linspace(0, 255, 9))
    ax1.set_xticklabels([str(int(x)) for x in np.linspace(0, 255, 9)])
    plt.colorbar(im1, ax=ax1, label="Routing Count")

    im2 = ax2.imshow(decode_matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax2.set_xlabel("Expert Index", fontsize=12)
    ax2.set_ylabel("Layer Index", fontsize=12)
    ax2.set_title("Expert Distribution Statistics (Decode)", fontsize=14)
    ax2.set_xticks(np.linspace(0, 255, 9))
    ax2.set_xticklabels([str(int(x)) for x in np.linspace(0, 255, 9)])
    plt.colorbar(im2, ax=ax2, label="Routing Count")

    plt.tight_layout()
    plt.savefig("expert_distribution_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
