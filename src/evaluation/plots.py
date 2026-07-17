import json
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

def plot_metric(
        train_values,
        val_values,
        metric_name: str,
        output_dir: Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(1,len(train_values) + 1)
    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        train_values,
        marker = "o",
        linewidth =2,
        label="Train",
    )

    plt.plot(
        epochs,
        val_values,
        marker="s",
        linewidth =2,
        label="Validation",
    )

    plt.title(f"{metric_name} Curve")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = (
        metric_name.lower().replace(" ","_") + "_curve.png"
    )
    plt.savefig(
        output_dir/filename,
        dpi=300,
        bbox_inches ="tight",
    )

    plt.close()
    logger.info("Saved %s", filename)

def save_history(
        history:dict,
        output_dir:Path,
):
    output_dir=Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    df = pd.DataFrame(history)
    df.to_csv(
        output_dir/"history.csv",
        index=False,
    )
    with open(output_dir/"history.json",
              "w",) as f:
        json.dump(
            history,
            f,
            indent=4,
        )

    logger.info("Saved history.csv")
    logger.info("Saved history.json")

def plot_training_history(
        history:dict,
        output_dir:Path,
):
    output_dir=Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    save_history(
        history,
        output_dir,
    )

    plot_metric(
        history["train_loss"],
        history["val_loss"],
        "Loss",
        output_dir,
    )

    plot_metric(
        history["train_miou"],
        history["val_miou"],
        "mIoU",
        output_dir,
    )

    plot_metric(
        history["train_dice"],
        history["val_dice"],
        "Dice",
        output_dir,
    )

    plot_metric(
        history["train_pixel_accuracy"],
        history["val_pixel_accuracy"],
        "Pixel Accuracy",
        output_dir,
    )

    print("=" * 60)
    print("Training history exported successfully!")
    print("="*60)