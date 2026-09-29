"""
Evaluation and Benchmark Engine for Unsupervised Anomaly Detection (UAD).
Implements dual-level evaluation:
1. Pixel-level metrics:
   - Pixel-AUROC (Area under Receiver Operating Characteristic)
   - Pixel-AUPRC (Area under Precision-Recall curve - critical for lesion imbalance)
   - Optimal theoretical Dice score (\\lceil Dice \\rceil) via threshold grid search
2. Slice-level metrics:
   - Slice-AUROC
   - Slice-AUPRC
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch


def compute_dice_score(pred_binary: np.ndarray, gt_binary: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Computes Dice similarity coefficient: 2 * |P & G| / (|P| + |G|)
    """
    intersection = np.sum(pred_binary * gt_binary)
    total = np.sum(pred_binary) + np.sum(gt_binary)
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((2.0 * intersection + smooth) / (total + smooth))


def compute_roc_pr_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure NumPy implementation of AUROC and AUPRC with fallback support.
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
        auroc = float(roc_auc_score(y_true, y_scores))
        auprc = float(average_precision_score(y_true, y_scores))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auroc, auprc, fpr, tpr, precision, recall
    except ImportError:
        # High-performance NumPy approximation
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]

        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5, 0.0, np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1])

        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg
        auroc = float(np.trapz(tpr, fpr))

        precision = np.cumsum(y_true_sorted) / (np.arange(len(y_true_sorted)) + 1)
        recall = tpr
        auprc = float(np.trapz(precision, recall))

        return auroc, auprc, fpr, tpr, precision, recall


def find_optimal_theoretical_dice(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    num_steps: int = 100,
) -> Tuple[float, float, List[float]]:
    """
    Performs grid search over threshold values in [0, 1] to calculate
    the theoretical maximum Dice score (\\lceil Dice \\rceil) across pathological slices.
    """
    patho_indices = np.where(gt_masks.sum(axis=(1, 2, 3)) > 0)[0]
    if len(patho_indices) == 0:
        return 0.0, 0.5, [0.0]

    maps_sub = anomaly_maps[patho_indices]
    masks_sub = gt_masks[patho_indices]

    # Normalize maps to [0, 1] per slice for consistent thresholding
    min_v = maps_sub.min(axis=(2, 3), keepdims=True)
    max_v = maps_sub.max(axis=(2, 3), keepdims=True)
    norm_maps = (maps_sub - min_v) / (max_v - min_v + 1e-8)

    thresholds = np.linspace(0.01, 0.99, num_steps)
    dice_per_thresh = []

    best_dice = -1.0
    best_threshold = 0.5

    for th in thresholds:
        bin_preds = (norm_maps >= th).astype(np.float32)
        dices = [
            compute_dice_score(bin_preds[i], masks_sub[i])
            for i in range(len(patho_indices))
        ]
        mean_d = float(np.mean(dices))
        dice_per_thresh.append(mean_d)

        if mean_d > best_dice:
            best_dice = mean_d
            best_threshold = float(th)

    return best_dice, best_threshold, dice_per_thresh


class UADEvaluator:
    """
    Comprehensive benchmark evaluator for UAD testing.
    """

    def __init__(self, scorer, device: str = "cpu"):
        self.scorer = scorer
        self.device = torch.device(device)

    def evaluate_dataset(
        self,
        dataloader,
        max_batches: int = None,
    ) -> Dict[str, Any]:
        """
        Runs inference across test dataloader and aggregates pixel- and slice-level metrics.
        """
        all_inputs = []
        all_recons = []
        all_residuals = []
        all_kl_saliencies = []
        all_anomaly_maps = []
        all_gt_masks = []
        all_slice_scores = []
        all_slice_labels = []

        batch_idx = 0
        for batch in dataloader:
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(self.device)
            masks = batch["mask"].cpu().numpy()
            labels = batch["label"].cpu().numpy()

            outputs = self.scorer.score_slice(images)

            all_inputs.append(images.detach().cpu().numpy())
            all_recons.append(outputs["reconstruction"].cpu().numpy())
            all_residuals.append(outputs["residual"].cpu().numpy())
            all_kl_saliencies.append(outputs["kl_saliency"].cpu().numpy())
            all_anomaly_maps.append(outputs["anomaly_map"].cpu().numpy())
            all_gt_masks.append(masks)
            all_slice_scores.append(outputs["slice_score"].cpu().numpy())
            all_slice_labels.append(labels)

            batch_idx += 1

        all_inputs = np.concatenate(all_inputs, axis=0)
        all_recons = np.concatenate(all_recons, axis=0)
        all_residuals = np.concatenate(all_residuals, axis=0)
        all_kl_saliencies = np.concatenate(all_kl_saliencies, axis=0)
        all_anomaly_maps = np.concatenate(all_anomaly_maps, axis=0)
        all_gt_masks = np.concatenate(all_gt_masks, axis=0)
        all_slice_scores = np.concatenate(all_slice_scores, axis=0)
        all_slice_labels = np.concatenate(all_slice_labels, axis=0)

        # 1. Pixel-level Metrics
        pixel_y_true = all_gt_masks.flatten().astype(int)
        pixel_y_scores = all_anomaly_maps.flatten()

        pix_auroc, pix_auprc, pix_fpr, pix_tpr, pix_prec, pix_rec = compute_roc_pr_metrics(
            pixel_y_true, pixel_y_scores
        )

        # Optimal theoretical Dice
        optimal_dice, opt_thresh, dice_curve = find_optimal_theoretical_dice(
            all_anomaly_maps, all_gt_masks, num_steps=100
        )

        # 2. Slice-level Metrics
        slice_auroc, slice_auprc, sl_fpr, sl_tpr, sl_prec, sl_rec = compute_roc_pr_metrics(
            all_slice_labels, all_slice_scores
        )

        return {
            "pixel_auroc": pix_auroc,
            "pixel_auprc": pix_auprc,
            "optimal_dice": optimal_dice,
            "optimal_threshold": opt_thresh,
            "slice_auroc": slice_auroc,
            "slice_auprc": slice_auprc,
            "inputs": all_inputs,
            "reconstructions": all_recons,
            "residuals": all_residuals,
            "kl_saliencies": all_kl_saliencies,
            "anomaly_maps": all_anomaly_maps,
            "gt_masks": all_gt_masks,
            "slice_scores": all_slice_scores,
            "slice_labels": all_slice_labels,
        }

    # Alias for method name consistency
    evaluate = evaluate_dataset
