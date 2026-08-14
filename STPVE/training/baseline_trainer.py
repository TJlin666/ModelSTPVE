from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:  # Optional dependency. If unavailable, sklearn fallbacks keep the code runnable.
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - depends on user environment.
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    HAS_LIGHTGBM = False

from STPVE.config import ExperimentConfig, PRODUCT_FEATURES
from STPVE.data.dataset import ForecastingDataset, parse_list_column
from STPVE.baseline_models.baseline_models import NeuralBaselineModel, extract_product_feature_batch, \
    extract_product_feature_vector
from STPVE.training.metrics import (
    compute_regression_metrics,
    sequence_shape_distances,
)
from STPVE.utils import count_parameters, worker_init_fn, count_traditional_model_parameters

NEURAL_BASELINES = {"LSTM", "GRU", "RNN", "Transformer", "MLP", "Bert+Transformer"}
TRADITIONAL_BASELINES = {"Linear Regression", "LightGBM"}
ALL_BASELINES = ["Linear Regression", "LightGBM","LSTM", "GRU", "RNN", "Transformer", "MLP", "Bert+Transformer"]


class BaselineTrainer:
    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device

    def train(
            self,
            baseline_type: str,
            data: pd.DataFrame,
            graph_dict: Dict,
            vocabulary: Dict[str, int],
            model_name: str | None = None,
            save_results: bool = True,
    ) -> List[Dict]:
        baseline_type = baseline_type.replace("-", "_")
        data = self._ensure_required_columns(data)
        if baseline_type in NEURAL_BASELINES:
            results = self._train_neural_baseline(
                baseline_type, data, graph_dict, vocabulary, model_name
            )
        elif baseline_type in TRADITIONAL_BASELINES:
            results = self._train_traditional_baseline(
                baseline_type, data, graph_dict, model_name
            )
        else:
            raise ValueError(
                f"Unsupported baseline_type={baseline_type}. Available baselines: {ALL_BASELINES}"
            )
        # Full runs write results once at the end of each fold.
        # Demo results are collected and written by the caller.
        if save_results and self.config.training.demo_mode == 1:
            self.save_results(results)
        return results

    def _ensure_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if "topic_indices" not in data.columns:
            data["topic_indices"] = [[0] * self.config.data.lda_top_topics for _ in range(len(data))]
        if "comment_text" not in data.columns:
            data["comment_text"] = ""
        return data

    def _build_loader(self, dataframe: pd.DataFrame, shuffle: bool) -> DataLoader:
        kwargs = {
            "batch_size": self.config.training.batch_size,
            "shuffle": shuffle,
            "pin_memory": self.device.type == "cuda",
            "num_workers": 2,
            "worker_init_fn": worker_init_fn,
        }
        return DataLoader(ForecastingDataset(dataframe, use_graph=True), **kwargs)

    def _build_neural_model(self, baseline_type: str, vocab_size: int) -> NeuralBaselineModel:
        data_config = self.config.data
        model_config = self.config.model
        return NeuralBaselineModel(
            baseline_hidden_size=model_config.baseline_hidden_size,
            baseline_input_dim=model_config.baseline_input_dim,
            baseline_num_layers=model_config.baseline_num_layers,
            baseline_type=baseline_type,
            bert_model_name=model_config.bert_model_name,
            channels=data_config.channels,
            data_width=data_config.data_width,
            history_length=data_config.history_length,
            prediction_length=data_config.prediction_length,
            vocab_size=vocab_size
        ).to(self.device)

    def _train_neural_baseline(self, baseline_type: str, data: pd.DataFrame, graph_dict: Dict,
                               vocabulary: Dict[str, int], model_name: str) -> List[Dict]:
        training_config = self.config.training

        gkf = GroupKFold(n_splits=training_config.n_splits, shuffle=True, random_state=training_config.random_state)
        groups = data["live_room_id"].values
        results: List[Dict] = []
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(data, groups=groups), start=1):
            print(f"\n========== Baseline {model_name} | Fold {fold_idx}/{training_config.n_splits} ==========")
            fold_start = len(results)
            train_df = data.iloc[train_idx].reset_index(drop=True)
            test_df = data.iloc[test_idx].reset_index(drop=True)
            train_loader = self._build_loader(train_df, shuffle=True)
            test_loader = self._build_loader(test_df, shuffle=False)
            model = self._build_neural_model(baseline_type, len(vocabulary))
            num_parameters = count_parameters(model)
            optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
            sequence_loss_fn = torch.nn.MSELoss()
            class_loss_fn = torch.nn.CrossEntropyLoss()

            for epoch_idx in range(training_config.num_epochs):
                loss = self._train_neural_epoch(model, train_loader, graph_dict, optimizer, sequence_loss_fn,
                                                class_loss_fn, epoch_idx)
                print(f"Baseline={model_name}, Epoch {epoch_idx + 1}/{training_config.num_epochs}, Loss={loss:.4f}")
                evaluation = self._evaluate_neural(model, test_loader, graph_dict, fold_idx, epoch_idx, model_name,
                                                   baseline_type, num_parameters)
                results.append(evaluation)
            if training_config.demo_mode == 0:
                # Write all epoch rows for the fold once.
                current_fold_results = results[fold_start:]
                self._append_frame(
                    pd.DataFrame(current_fold_results),
                    self.config.paths.baseline_epoch_output_path,
                )
                self._upsert_frame(
                    self.best_r2_rows(current_fold_results),
                    self.config.paths.baseline_best_epoch_output_path,
                    keys=["model_name", "fold"],
                )
            del model, train_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if training_config.demo_mode == 1:
                break
        return results

    def _train_neural_epoch(self, model, loader, graph_dict, optimizer, sequence_loss_fn, class_loss_fn,
                            epoch_idx: int) -> float:
        model.train()
        total_loss = 0.
        optimizer.zero_grad()
        loss_weights = self.config.training.loss_weights
        progress = tqdm(enumerate(loader), total=len(loader),
                        desc=f"Baseline epoch {epoch_idx + 1}/{self.config.training.num_epochs}")
        for batch_idx, batch in progress:
            behavior = batch["behavior"].to(self.device, non_blocking=True)
            comment_indices = batch["comment_indices"].to(self.device, non_blocking=True)
            sequence_labels = batch["sequence_label"].to(self.device, non_blocking=True)
            class_labels = batch["class_label"].to(self.device, non_blocking=True)
            product_features = extract_product_feature_batch(graph_dict, list(batch["graph_key"]), self.device)
            comment_texts = list(batch.get("comment_text", [])) if isinstance(batch,
                                                                              dict) and "comment_text" in batch else None
            sequence_outputs, class_outputs = model(behavior, comment_indices, product_features,
                                                    comment_texts=comment_texts)
            sequence_loss = sequence_loss_fn(sequence_outputs, sequence_labels)
            class_loss = class_loss_fn(class_outputs, class_labels)
            loss = loss_weights.get("sequence", 1.0) * sequence_loss + loss_weights.get("class", 1) * class_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate_neural(self, model, loader, graph_dict, fold_idx: int, epoch_idx: int, model_name: str,
                         baseline_type: str, num_parameters: int) -> Dict:
        model.eval()
        true_sequence_list, pred_sequence_list = [], []
        true_class_list, pred_class_list, pred_class_probs_list = [], [], []
        with torch.no_grad():
            for batch in loader:
                behavior = batch["behavior"].to(self.device, non_blocking=True)
                comment_indices = batch["comment_indices"].to(self.device, non_blocking=True)
                sequence_labels = batch["sequence_label"].to(self.device, non_blocking=True)
                class_labels = batch["class_label"].to(self.device, non_blocking=True)
                product_features = extract_product_feature_batch(graph_dict, list(batch["graph_key"]), self.device)
                comment_texts = list(batch.get("comment_text", [])) if isinstance(batch,
                                                                                  dict) and "comment_text" in batch else None
                sequence_outputs, class_outputs = model(behavior, comment_indices, product_features,
                                                        comment_texts=comment_texts)
                true_sequence_list.append(sequence_labels)
                pred_sequence_list.append(sequence_outputs)
                true_class_list.append(class_labels)
                pred_class_list.append(class_outputs.argmax(dim=1))
                pred_class_probs_list.append(F.softmax(class_outputs, dim=1).cpu())
        y_true_sequence = torch.cat(true_sequence_list, dim=0)
        y_pred_sequence = torch.cat(pred_sequence_list, dim=0)
        y_true_class = torch.cat(true_class_list, dim=0)
        y_pred_class = torch.cat(pred_class_list, dim=0)
        y_pred_class_probs = torch.cat(pred_class_probs_list, dim=0).numpy()
        return self._build_evaluation_result(
            model_name=model_name,
            baseline_type=baseline_type,
            fold_idx=fold_idx,
            epoch_idx=epoch_idx,
            num_parameters=num_parameters,
            y_true_sequence=y_true_sequence,
            y_pred_sequence=y_pred_sequence,
            y_true_class=y_true_class,
            y_pred_class=y_pred_class,
            y_pred_class_probs=y_pred_class_probs,
        )

    def _product_vector_for_row(self, row: pd.Series, graph_dict: Dict) -> np.ndarray:
        graph = graph_dict.get(row.get("graph_key"))
        if graph is None:
            return np.zeros(len(PRODUCT_FEATURES), dtype=np.float32)
        return extract_product_feature_vector(graph, product_dim=len(PRODUCT_FEATURES),
                                              device=torch.device("cpu")).cpu().numpy()

    def _build_traditional_features(self, dataframe: pd.DataFrame, graph_dict: Dict) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        features = []
        sequences = []
        classes = []
        for _, row in dataframe.iterrows():
            if "behavior" not in row.index:
                raise KeyError("The processed dataset must contain a behavior column.")
            behavior = np.asarray(parse_list_column(row["behavior"]), dtype=np.float32).reshape(-1)
            content = np.asarray(parse_list_column(row["comment_indices"]), dtype=np.float32).reshape(-1)
            if content.size > 0:
                content = content / max(content.max(), 1.0)
            product = self._product_vector_for_row(row, graph_dict)
            features.append(np.concatenate([behavior, content, product], axis=0))
            sequences.append(np.asarray(parse_list_column(row["sequence_label"]), dtype=np.float32).reshape(-1))
            classes.append(int(row["class_label"]))
        return np.vstack(features), np.vstack(sequences), np.asarray(classes, dtype=np.int64)

    def _make_lgb_regressor(self):
        if HAS_LIGHTGBM:
            return LGBMRegressor(n_estimators=self.config.training.lgb_n_estimators,
                                 learning_rate=self.config.training.lgb_learning_rate,
                                 num_leaves=self.config.training.lgb_num_leaves,
                                 random_state=self.config.training.random_state,
                                 verbose=-1)
        return HistGradientBoostingRegressor(max_iter=self.config.training.lgb_n_estimators,
                                             learning_rate=self.config.training.lgb_learning_rate,
                                             random_state=self.config.training.random_state)

    def _make_lgb_classifier(self):
        if HAS_LIGHTGBM:
            return LGBMClassifier(n_estimators=self.config.training.lgb_n_estimators,
                                  learning_rate=self.config.training.lgb_learning_rate,
                                  num_leaves=self.config.training.lgb_num_leaves,
                                  random_state=self.config.training.random_state, 
                                  verbose=-1)
        return HistGradientBoostingClassifier(max_iter=self.config.training.lgb_n_estimators,
                                              learning_rate=self.config.training.lgb_learning_rate,
                                              random_state=self.config.training.random_state)

    def _train_traditional_baseline(self, baseline_type: str, data: pd.DataFrame, graph_dict: Dict, model_name: str) -> \
            List[Dict]:
        training_config = self.config.training
        gkf = GroupKFold(n_splits=training_config.n_splits, shuffle=True, random_state=training_config.random_state)
        groups = data["live_room_id"].values
        results: List[Dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(data, groups=groups), start=1):
            print(
                f"\n========== Traditional baseline {model_name} | Fold {fold_idx}/{training_config.n_splits} ==========")
            train_df = data.iloc[train_idx].reset_index(drop=True)
            test_df = data.iloc[test_idx].reset_index(drop=True)
            x_train, y_train_sequence, y_train_class = self._build_traditional_features(train_df, graph_dict)
            x_test, y_test_sequence, y_test_class = self._build_traditional_features(test_df, graph_dict)
            sequence_models = []
            predictions = []
            for step in range(y_train_sequence.shape[1]):
                if baseline_type == "Linear Regression":
                    sequence_model = make_pipeline(StandardScaler(), LinearRegression())
                else:
                    sequence_model = self._make_lgb_regressor()
                sequence_model.fit(x_train, y_train_sequence[:, step])
                sequence_models.append(sequence_model)
                predictions.append(sequence_model.predict(x_test).reshape(-1, 1))
            y_pred_sequence = np.concatenate(predictions, axis=1)

            if len(np.unique(y_train_class)) < 2:
                class_model = DummyClassifier(strategy="most_frequent")
                class_model.fit(x_train, y_train_class)
            elif baseline_type == "Linear Regression":
                class_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                class_model.fit(x_train, y_train_class)
            else:
                class_model = self._make_lgb_classifier()
                class_model.fit(x_train, y_train_class)
            y_pred_class = class_model.predict(x_test)
            y_pred_class_probs = self._predict_full_class_proba(class_model, x_test)

            num_parameters = (
                    sum(count_traditional_model_parameters(model) for model in sequence_models)
                    + count_traditional_model_parameters(class_model)
            )

            evaluation = self._build_evaluation_result(
                model_name=model_name,
                baseline_type=baseline_type,
                fold_idx=fold_idx,
                epoch_idx=0,
                num_parameters=num_parameters,
                y_true_sequence=torch.tensor(y_test_sequence, dtype=torch.float32),
                y_pred_sequence=torch.tensor(y_pred_sequence, dtype=torch.float32),
                y_true_class=torch.tensor(y_test_class, dtype=torch.long),
                y_pred_class=torch.tensor(y_pred_class, dtype=torch.long),
                y_pred_class_probs=y_pred_class_probs,
            )
            results.append(evaluation)
            if training_config.demo_mode == 0:
                self._append_frame(
                    pd.DataFrame([evaluation]),
                    self.config.paths.baseline_epoch_output_path,
                )
                self._upsert_frame(
                    self.best_r2_rows([evaluation]),
                    self.config.paths.baseline_best_epoch_output_path,
                    keys=["model_name", "fold"],
                )
            if training_config.demo_mode == 1:
                break
        return results

    def _predict_full_class_proba(self, model, x_test: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(x_test)
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                final_estimator = list(model.named_steps.values())[-1]
                classes = getattr(final_estimator, "classes_", None)
            full_probs = np.zeros((x_test.shape[0], 3), dtype=np.float32)
            if classes is None:
                width = min(3, raw_probs.shape[1])
                full_probs[:, :width] = raw_probs[:, :width]
            else:
                for column_idx, class_label in enumerate(classes):
                    if int(class_label) in {0, 1, 2}:
                        full_probs[:, int(class_label)] = raw_probs[:, column_idx]
            row_sums = full_probs.sum(axis=1, keepdims=True)
            return full_probs / np.maximum(row_sums, 1e-12)
        pred = model.predict(x_test).astype(int)
        full_probs = np.zeros((x_test.shape[0], 3), dtype=np.float32)
        full_probs[np.arange(x_test.shape[0]), np.clip(pred, 0, 2)] = 1.0
        return full_probs

    def _build_evaluation_result(
            self,
            model_name: str,
            baseline_type: str,
            fold_idx: int,
            epoch_idx: int,
            num_parameters: int,
            y_true_sequence: torch.Tensor,
            y_pred_sequence: torch.Tensor,
            y_true_class: torch.Tensor,
            y_pred_class: torch.Tensor,
            y_pred_class_probs: np.ndarray,
    ) -> Dict:
        y_true_sequence = y_true_sequence.cpu()
        y_pred_sequence = y_pred_sequence.cpu()
        y_true_class = y_true_class.cpu()
        y_pred_class = y_pred_class.cpu()
        sequence_loss = float(F.mse_loss(y_pred_sequence, y_true_sequence).item())
        sequence_metrics = compute_regression_metrics(y_true_sequence, y_pred_sequence)
        cepstral_distance, dtw_similarity = sequence_shape_distances(y_true_sequence.numpy(), y_pred_sequence.numpy(),
                                                                     3)

        class_accuracy = float((y_pred_class == y_true_class).float().mean().item())
        class_f1 = float(f1_score(y_true_class.numpy(), y_pred_class.numpy(), average="weighted"))
        __, class_recall, _, _ = precision_recall_fscore_support(y_true_class.numpy(),
                                                                              y_pred_class.numpy(),
                                                                              average="macro", zero_division=0)
        try:
            class_auc = float(
                roc_auc_score(y_true_class.numpy(), y_pred_class_probs, multi_class="ovr", average="weighted"))
        except ValueError:
            class_auc = float("nan")
        shanghai_tz = pytz.timezone("Asia/Shanghai")
        return {
            "model_name": model_name,
            "baseline_type": baseline_type,
            "task_mode": "multi_task",
            "num_parameters": num_parameters,
            "loss": sequence_loss,
            "fold": fold_idx,
            "epoch": epoch_idx + 1,
            "time": datetime.now(shanghai_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "sequence_RMSE": sequence_metrics[0],
            "sequence_MAE": sequence_metrics[1],
            "sequence_R2": sequence_metrics[2],
            "sequence_WMAPE": sequence_metrics[3],
            "class_Accuracy": class_accuracy,
            "class_F1": class_f1,
            "class_AUC": class_auc,
            "class_Recall": float(class_recall),
            "avg_cepstral_distance": cepstral_distance,
            "avg_dtw_similarity": dtw_similarity,
        }

    @staticmethod
    def best_r2_rows(results: List[Dict]) -> pd.DataFrame:
        frame = pd.DataFrame(results)
        if frame.empty:
            return frame
        valid = frame.dropna(subset=["sequence_R2"])
        if valid.empty:
            best = frame.groupby(["model_name", "fold"], as_index=False).first()
            return best.drop(columns=["epoch"], errors="ignore")
        best_indices = valid.groupby(["model_name", "fold"])["sequence_R2"].idxmax()
        best = frame.loc[best_indices].sort_values(["model_name", "fold"]).reset_index(drop=True)
        # The epoch is used to select the best row, but it is intentionally
        # omitted from the best-result Excel files. Epoch-level files keep it.
        return best.drop(columns=["epoch"], errors="ignore")

    def save_results(self, results: List[Dict]) -> None:
        if not results:
            return
        training_config = self.config.training
        if training_config.demo_mode == 1:
            self._append_frame(
                self.best_r2_rows(results),
                self.config.paths.demo_output_path,
            )
            return

        self._append_frame(
            pd.DataFrame(results),
            self.config.paths.baseline_epoch_output_path,
        )
        self._upsert_frame(
            self.best_r2_rows(results),
            self.config.paths.baseline_best_epoch_output_path,
            keys=["model_name", "fold"],
        )

    @staticmethod
    def _upsert_frame(frame: pd.DataFrame, output_path: str, keys: List[str]) -> None:
        if frame.empty:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if os.path.exists(output_path):
            existing = pd.read_excel(output_path, sheet_name="Results")
            if not existing.empty and all(key in existing.columns for key in keys):
                incoming_keys = frame[keys].astype(str).agg("||".join, axis=1)
                existing_keys = existing[keys].astype(str).agg("||".join, axis=1)
                existing = existing.loc[~existing_keys.isin(set(incoming_keys))]
                frame = pd.concat([existing, frame], ignore_index=True)
        if all(key in frame.columns for key in keys):
            frame = frame.sort_values(keys).reset_index(drop=True)
        frame.to_excel(output_path, sheet_name="Results", index=False)

    @staticmethod
    def _append_frame(frame: pd.DataFrame, output_path: str) -> None:
        if frame.empty:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if os.path.exists(output_path):
            existing = pd.read_excel(output_path, sheet_name="Results")
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_excel(output_path, sheet_name="Results", index=False)
