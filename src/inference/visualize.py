from pathlib import Path
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.inference.color_map import CLASS_COLORS

logger = logging.getLogger(__name__)

def mask_to_rgb(mask: np.ndarray)-> np.ndarray:
    height, width = mask.shape

    rgb = np.zeros((height,width,3),dtype=np.uint8)
    for cls,color in CLASS_COLORS.items():
        rgb[mask==cls] = color
    return rgb

def create_overlay(
        image: np.ndarray,
        prediction: np.ndarray,
        alpha : float = 0.5,
):
    rgb_mask = mask_to_rgb(prediction)

    overlay = cv2.addWeighted(
        image,
        1-alpha,
        rgb_mask,
        alpha,
        0,
    )
    return overlay

def plot_results(
        image,
        prediction,
        confidence = None,
        ground_truth = None,
):
    rgb_prediction = mask_to_rgb(prediction)
    
    overlay = create_overlay(
        image,
        prediction,
    )

    figures = 4
    if ground_truth is not None:
        figures += 1

    if confidence is not None:
        figures += 1
    
    plt.figure(figsize=(18,8))
    index = 1

    plt.subplot(2,3,index)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    index += 1

    if ground_truth is not None:
        plt.subplot(2,3,index)
        plt.imshow(mask_to_rgb(ground_truth))
        plt.title("Ground_Truth")
        plt.axis("off")

        index  += 1

    plt.subplot(2,3,index)
    plt.imshow(rgb_prediction)
    plt.title("Prediction")
    plt.axis("off")

    index += 1

    plt.subplot(2,3,index)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    index  += 1

    if confidence is not None:
        plt.subplot(2,3,index)
        plt.imshow(confidence,cmap='viridis')
        plt.colorbar()
        plt.title("Confidence")
        plt.axis("off")
    plt.tight_layout()
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close()

def save_results(
        image,
        prediction,
        output_dir,
        image_name,
):
    
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb_prediction = mask_to_rgb(prediction)
    
    overlay = create_overlay(
        image,
        prediction,
    )

    cv2.imwrite(
        str(output_dir/f"{image_name}_prediction.png"),
        cv2.cvtColor(rgb_prediction,cv2.COLOR_RGB2BGR,
        ),
        
    )
    cv2.imwrite(
        str(output_dir/f"{image_name}_overlay.png"),
        cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR,
        ),
    )
print("results saved")