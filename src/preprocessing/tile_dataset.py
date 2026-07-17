import json
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import sys

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.process_split import process_split
from src.utils.dataset_config import LOVEDA_CONFIG


def create_output_directories(root: Path) -> None:
    """  
    Create processed dataset directories.
    """

    directories = [
        root / "train" / "images",
        root / "train" / "masks",
        root / "val" / "images",
        root / "val" / "masks",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_metadata(metadata_records, output_path: Path) -> None:
    """
    Save metadata as CSV.
    """

    if len(metadata_records) == 0:
        logger.warning("No metadata to save.")
        return

    metadata = pd.DataFrame(metadata_records)

    metadata.to_csv(output_path, index=False)

    logger.info("Metadata saved to: %s", output_path)


def save_summary(
    train_summary,
    val_summary,
    output_path: Path,
) -> None:


    summary = {

        "dataset": "LoveDA",

        "created_at": datetime.now().isoformat(),

        "tile_size": LOVEDA_CONFIG["tile_size"],

        "num_classes": LOVEDA_CONFIG["num_classes"],

        "train_images": train_summary.total_images,

        "train_tiles": train_summary.total_tiles,

        "val_images": val_summary.total_images,

        "val_tiles": val_summary.total_tiles,

        "train_processing_time": train_summary.processing_time,

        "val_processing_time": val_summary.processing_time,

        "total_processing_time":

            train_summary.processing_time
            + val_summary.processing_time
    }

    with open(output_path, "w") as file:

        json.dump(summary, file, indent=4)

    logger.info("Summary saved to: %s", output_path)


def main():

    from src.utils.logging import configure_logging
    configure_logging()
    logger.info("%s", "=" * 70)
    logger.info("LoveDA Dataset Preprocessing")
    logger.info("%s", "=" * 70)

    config = LOVEDA_CONFIG

    input_root = config["root"]

    output_root = config["processed_root"]

    create_output_directories(output_root)


    train_summary, train_metadata = process_split(

        dataset_name=config["name"],

        split="Train",

        input_root=input_root,

        output_root=output_root,
    )


    val_summary, val_metadata = process_split(

        dataset_name=config["name"],

        split="Val",

        input_root=input_root,

        output_root=output_root,
    )


    metadata = train_metadata + val_metadata

    save_metadata(

        metadata,

        output_root / "metadata.csv",
    )


    save_summary(

        train_summary,

        val_summary,

        output_root / "preprocessing_summary.json",
    )

    logger.info("\n")
    logger.info("%s", "=" * 70)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    main()