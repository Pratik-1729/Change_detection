from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import CLASS_NAMES

logger = logging.getLogger(__name__)

def normalize_confusion_matrix(confsion_matrix: np.ndarray):
    cm = confsion_matrix.astype(np.float64)

    row_sum = cm.sum(axis=1, keepdims=True)

    row_sum[row_sum == 0] = 1

    return cm/row_sum

def plot_confusion_matrix(
    confusion_matrix:np.ndarray,
    normalize: bool = False,
):
    cm = confusion_matrix
    if normalize:
        cm = normalize_confusion_matrix(cm)
    class_names = list(CLASS_NAMES.values())
    plt.figure(figsize=(8,7))
    plt.imshow(cm,cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))

    plt.xticks(ticks,class_names,rotation = 45,ha="right")

    plt.yticks(ticks,class_names)

    threshold = cm.max()/2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i,j]

            if normalize:
                text = f"{value:.2f}"
            else:
                text = f"{int(value)}"
            plt.text(
                j,i,text,ha="center",va="center",
                color="white" if value > threshold else "black",
            )
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    return plt.gcf()

def save_confusion_matrix(
        confusion_matrix:np.ndarray,
        output_dir: Path,
        normalize:bool = False,
):
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plot_confusion_matrix(
        confusion_matrix,
        normalize,
    )

    filename = (
        "confusion_matrix_normalized.png"
        if normalize
        else "confusion_matrix.png"
    )

    figure.savefig(
        output_dir / filename,
        dpi = 300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info("Saved %s", filename)