from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, float, float, float]:
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
    mae = float(mean_absolute_error(y_true_np, y_pred_np))
    r2 = float(r2_score(y_true_np, y_pred_np))
    denominator = np.sum(np.abs(y_true_np))
    wmape = float(np.sum(np.abs(y_true_np - y_pred_np)) / denominator) if denominator != 0 else 0.0
    return rmse, mae, r2, wmape


def compute_slope(sequence: List[float] | np.ndarray) -> float:
    sequence = np.asarray(sequence, dtype=float)
    x = np.arange(len(sequence))
    return float(np.polyfit(x, sequence, 1)[0])


def batch_compute_slope(sequence_tensor: torch.Tensor) -> torch.Tensor:
    sequence_length = sequence_tensor.shape[1]
    x = torch.arange(sequence_length, dtype=torch.float32, device=sequence_tensor.device)
    x_mean = x.mean()
    y_mean = sequence_tensor.mean(dim=1)
    covariance = ((x - x_mean) * (sequence_tensor - y_mean.unsqueeze(1))).sum(dim=1)
    variance = ((x - x_mean) ** 2).sum()
    return covariance / variance


def slope_to_class(slope: float, threshold: float = 0.3) -> int:
    if slope > threshold:
        return 1
    if slope < -threshold:
        return 2
    return 0





class ShortSequenceDTW:
    name = "Short Sequence DTW for Periodicity Detection"

    def dtw_distance(self, first: np.ndarray, second: np.ndarray) -> tuple[float, np.ndarray]:
        n, m = len(first), len(second)
        matrix = np.full((n + 1, m + 1), np.inf)
        matrix[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(first[i - 1] - second[j - 1])
                matrix[i, j] = cost + min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            best_previous = min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
            if matrix[i - 1, j - 1] == best_previous:
                i, j = i - 1, j - 1
            elif matrix[i - 1, j] == best_previous:
                i -= 1
            else:
                j -= 1
        path.reverse()
        return float(matrix[n, m]), np.asarray(path)

    def compare_sequences(self, first: List[float] | np.ndarray, second: List[float] | np.ndarray) -> Dict[str, float]:
        first_arr, second_arr = np.asarray(first), np.asarray(second)
        distance, path = self.dtw_distance(first_arr, second_arr)
        value_range = max(np.max(first_arr), np.max(second_arr)) - min(np.min(first_arr), np.min(second_arr))
        value_range = value_range if value_range != 0 else 1.0
        normalized_distance = distance / (len(path) * value_range) if len(path) > 0 else 0.0
        return {"dtw_distance": distance, "similarity_score": float(1.0 / (1.0 + normalized_distance))}


def sequence_shape_distances(y_true: np.ndarray, y_pred: np.ndarray, n_cepstral: int = 3) -> tuple[float, float]:
    cepstral_distances, dtw_similarities = [], []
    detector = ShortSequenceDTW()
    for true_sequence, pred_sequence in zip(y_true, y_pred):
        cep_true = compute_cepstral(true_sequence, n_coeffs=n_cepstral)
        cep_pred = compute_cepstral(pred_sequence, n_coeffs=n_cepstral)
        cepstral_distances.append(euclidean(cep_true, cep_pred))
        dtw_similarities.append(detector.compare_sequences(np.ravel(true_sequence), np.ravel(pred_sequence))["similarity_score"])
    return float(np.mean(cepstral_distances)), float(np.mean(dtw_similarities))


def compute_cepstral(seq, n_coeffs=3):
    seq = np.asarray(seq, dtype=float).reshape(-1)

    spec = np.fft.fft(seq)
    magnitude = np.abs(spec)
    log_magnitude = np.log(magnitude + 1e-10)
    cepstrum = np.fft.ifft(log_magnitude).real

    if len(cepstrum) > n_coeffs:
        return list(cepstrum[1:n_coeffs + 1])

    return list(
        np.pad(
            cepstrum[1:],
            (0, max(0, n_coeffs - (len(cepstrum) - 1))),
            mode="constant",
        )
    )