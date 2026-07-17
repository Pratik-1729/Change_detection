import torch

class SegmentationMetrics:
    def __init__(self,num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.int64,
        )
    def update(
            self,
            predictions:torch.Tensor,
            targets: torch.Tensor   
            ):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        mask = (
            (targets >=0) & (targets < self.num_classes)
        )
        predictions = predictions[mask]
        targets = targets[mask]
        indices = (
            self.num_classes * targets 
            +
            predictions
            )
        cm = torch.bincount(
            indices,
            minlength=self.num_classes**2,
        )
        cm = cm.reshape(self.num_classes,self.num_classes)
        self.confusion_matrix += cm

    def compute(self):
        cm = self.confusion_matrix.float()

        eps = 1e-7
       
        tp = torch.diag(cm)

        fp = cm.sum(dim=0) - tp

        fn = cm.sum(dim=1) - tp

        tn = cm.sum() - (tp + fp + fn)

        pixel_accuracy = tp.sum() / (cm.sum() + eps)

        precision = tp/(tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2*precision*recall) / (precision + recall + eps)

        iou = tp / (tp + fp + fn + eps)
        dice = (2 * tp ) / (2 * tp + fp + fn + eps)
        support = cm.sum(dim=1)


        return{
        "pixel_accuracy": pixel_accuracy.item(),
        "precision" : precision.mean().item(),
        "f1_score" : f1.mean().item(),
        "recall" : recall.mean().item(),
        "miou" : iou.mean().item(),
        "dice" : dice.mean().item(),

        "per_class_iou": iou.tolist(),
        "per_class_precision" : precision.tolist(),
        "per_class_recall" : recall.tolist(),
        "per_class_dice" : dice.tolist(),
        "support": support.tolist(),
        "confusion_matrix" : cm.cpu().numpy(),
    
}


def evaluate_one_batch(
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
):
    predictions = torch.argmax(logits, dim=1)
    metrics = SegmentationMetrics(num_classes)
    metrics.update(predictions, targets)
    return metrics.compute()

'''
def pixel_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,

) -> float:
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct/total

def mean_iou(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes : int,
)->float:
    ious = []
    for cls in range(num_classes):
        pred = predictions == cls
        target = targets == cls
        intersection = (pred & target).sum().float()
        union = (pred | target).sum().float()

        if union == 0:
            continue
        iou = intersection/union

        ious.append(iou)

    if len(ious)==0:
        return 0.0
    return torch.stack(ious).mean().item()

def dice_score(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes : int,
)->float:
    dice_scores = []
    for cls in range(num_classes):
        pred = predictions == cls
        target = targets == cls
        intersection = (pred & target).sum().float()
        denominator = pred.sum().float() + target.sum().float()
        if denominator == 0:
            continue

        dice = (2*intersection)/denominator
        dice_scores.append(dice)
    if len(dice_scores) == 0:
        return 0.0
    return torch.stack(dice_scores).mean().item()


def evaluate_epoch(
        all_predictions: torch.Tensor,
        all_targets: torch.Tensor,
        num_classes : int,
):
    predictions = torch.cat(
        all_predictions
    )

    targets = torch.cat(all_targets)


'''


