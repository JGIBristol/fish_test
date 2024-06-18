import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot(n):
    arr = np.load(f"cropped_arrs/{n:03}.npy")

    # Choose the slices to display
    indices = np.floor(np.arange(0, 160, 10)).astype(int)
    fig, axes = plt.subplots(4, 4)
    for i, ax in zip(indices, axes.flat):
        ax.imshow(arr[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(f"{n:03}")
    fig.tight_layout()
    fig.savefig(f"imgs/{n:03}.png")
    plt.close(fig)


def main():
    for path in tqdm(os.listdir("cropped_arrs/")):
        n = int(path.split(".")[0])
        if not os.path.exists(f"imgs/{n:03}.png"):
            plot(n)

if __name__ == "__main__":
    main()