import json
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch

from src.utils.constants import CLASS_NAMES

logger = logging.getLogger(__name__)


def _json_serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def save_json(
        results: dict,
        output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir/"metrics.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            default=_json_serialize,
        )
    logger.info("metrics.json saved")

def save_csv(
        results:dict,
        output_dir: Path,
):
    rows = []
    for class_id,class_name in CLASS_NAMES.items():
        rows.append({
            "Class" : class_name,
            "IoU"           :results['per_class_iou'][class_id],
            "Dice"       :results['per_class_dice'][class_id],
            "Precision"  :results['per_class_precision'][class_id],
            "Recall"      :results['per_class_recall'][class_id],
            "Support"     :results['support'][class_id],

        })

    df = pd.DataFrame(rows)

    df.to_csv(
        output_dir/"class_metrics.csv",
        index=False,
    )

    logger.info("class_metrics.csv saved")

def save_summary(
        results: dict,
        output_dir : Path,
):
    summary = {
        "Loss"                :results['loss'],
        "pixel_accuracy"      :results['pixel_accuracy'],
        "Precision"           :results['precision'],
        "Recall"              :results['recall'],
        "F1 Score"            :results['f1_score'],
        "Mean IoU"            :results['miou'],
        "Dice Score"          :results['dice'],
    }

    df = pd.DataFrame([summary])
    df.to_csv(
        output_dir/"summary.csv",
        index=False,
    )

    print("summary.csv saved")