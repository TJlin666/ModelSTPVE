from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import pytz
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from STPVE.config import ExperimentConfig
from STPVE.data.dataset import ForecastingDataset
from STPVE.data.hetero_graph_builder import load_product_attribute_index
from STPVE.data.text_builder import get_top_topic_indices, save_pickle, train_lda_model
from STPVE.models.model import STPVE
from STPVE.training.metrics import (
    compute_regression_metrics,
    sequence_shape_distances,
)
from STPVE.utils import count_parameters, worker_init_fn


class ExperimentTrainer:
    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device
        self.attribute_index = load_product_attribute_index(config.paths.product_attribute_path)

    def _build_model(self, vocab_size: int) -> STPVE:
        model_config = self.config.model
        data_config = self.config.data
        input_size = [data_config.channels, data_config.history_length, data_config.data_width]
        return STPVE(
            behavior_kernel_size=model_config.behavior_kernel_size,
            content_kernel_num=model_config.content_kernel_num,
            content_kernel_sizes=self.config.model.content_kernel_sizes,
            cnn_input_size=input_size,
            hidden_size_lstm=model_config.hidden_size_lstm,
            module_out_dim=model_config.module_out_dim,
            num_attr_nodes=self.attribute_index.num_attr_nodes,
            num_layers_lstm=model_config.num_layers_lstm,
            num_heads_graph=model_config.num_heads_graph,
            num_heads_lstm=model_config.num_heads_lstm,
            num_topics=data_config.lda_num_topics,
            prediction_length=data_config.prediction_length,
            topic_embed_dim=model_config.topic_embed_dim,
            topic_kernel_size=model_config.topic_kernel_size,
            vocab_size=vocab_size,
            vocab_embed_dim=model_config.vocab_embed_dim
        ).to(self.device)

    def train(
            self,
            data: pd.DataFrame,
            graph_dict: Dict,
            vocabulary: Dict[str, int],
            model_name: str = "STPVE",
            save_results: bool = True,
    ) -> List[Dict]:
        results = self._train_multitask(data, graph_dict, vocabulary, model_name)
        # Full runs write results once at the end of each fold.
        # Demo results are collected by run_demo.py and written once.
        if save_results and self.config.training.demo_mode == 1:
            self.save_results(results)
        return results

    def _train_multitask(self, data: pd.DataFrame, graph_dict: Dict, vocabulary: Dict[str, int], model_name: str) -> \
            List[Dict]:
        training_config = self.config.training
        fold_results: List[Dict] = []
        gkf = GroupKFold(n_splits=training_config.n_splits, shuffle=True, random_state=training_config.random_state)
        groups = data["live_room_id"].values
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(data, groups=groups), start=1):
            print(f"\n========== Fold {fold_idx}/{training_config.n_splits} | {model_name} ==========")
            fold_start = len(fold_results)
            train_df = data.iloc[train_idx].reset_index(drop=True)
            test_df = data.iloc[test_idx].reset_index(drop=True)
            train_df, test_df = self._fit_fold_lda_topics(train_df, test_df, fold_idx)
            train_loader = self._build_loader(train_df, shuffle=True)
            test_loader = self._build_loader(test_df, shuffle=False)
            model = self._build_model(len(vocabulary))
            num_parameters = count_parameters(model)

            optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
            sequence_loss_fn = torch.nn.MSELoss()
            class_loss_fn = torch.nn.CrossEntropyLoss()
            start_time = time.time()

            for epoch_idx in range(training_config.num_epochs):
                train_loss = self._train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    graph_dict=graph_dict,
                    sequence_loss_fn=sequence_loss_fn,
                    class_loss_fn=class_loss_fn,
                    epoch_idx=epoch_idx,
                )
                print(f"Epoch {epoch_idx + 1}/{training_config.num_epochs}, Loss: {train_loss:.4f}")
                evaluation = self.evaluate(model, test_loader, graph_dict, fold_idx, epoch_idx, model_name,
                                           num_parameters)
                fold_results.append(evaluation)

            if training_config.demo_mode == 0:
                # Write all epoch rows for the fold once, avoiding repeated Excel rewrites.
                current_fold_results = fold_results[fold_start:]
                self._append_frame(
                    pd.DataFrame(current_fold_results),
                    self.config.paths.stpve_epoch_output_path,
                )
                self._upsert_frame(
                    self.best_r2_rows(current_fold_results),
                    self.config.paths.stpve_best_epoch_output_path,
                    keys=["model_name", "fold"],
                )

            print(f"Fold training time: {time.time() - start_time:.2f}s")
            del model, train_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if training_config.demo_mode == 1:
                break
        return fold_results

    def _fit_fold_lda_topics(self, train_df: pd.DataFrame, test_df: pd.DataFrame, fold_idx: int):
        if "comment_text" not in train_df.columns or "comment_text" not in test_df.columns:
            raise KeyError(
                "`comment_text` is required for fold-level LDA training. "
                "Re-run preprocessing from raw data with use_preprocessed=False."
            )

        train_df = train_df.copy()
        test_df = test_df.copy()

        data_config = self.config.data
        vectorizer, lda_model = train_lda_model(
            train_df["comment_text"].fillna("").astype(str).tolist(),
            num_topics=data_config.lda_num_topics,
            lda_max_features=data_config.lda_max_features,
        )
        train_df["topic_indices"] = train_df["comment_text"].apply(
            lambda text: get_top_topic_indices(text, vectorizer, lda_model, self.config.data.lda_top_topics))
        test_df["topic_indices"] = test_df["comment_text"].apply(
            lambda text: get_top_topic_indices(text, vectorizer, lda_model, self.config.data.lda_top_topics))
        save_pickle(vectorizer, self.config.paths.lda_vectorizer_path)
        save_pickle(lda_model, self.config.paths.lda_model_path)
        return train_df, test_df

    def _build_loader(self, dataframe: pd.DataFrame, shuffle: bool) -> DataLoader:
        loader_kwargs = {
            "batch_size": self.config.training.batch_size,
            "shuffle": shuffle,
            "pin_memory": self.device.type == "cuda",
            "num_workers": 2,
            "worker_init_fn": worker_init_fn,
        }
        return DataLoader(ForecastingDataset(dataframe, use_graph=True), **loader_kwargs)

    def _train_one_epoch(
            self,
            model,
            loader,
            optimizer,
            graph_dict,
            sequence_loss_fn,
            class_loss_fn,
            epoch_idx: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        loss_weights = self.config.training.loss_weights
        progress = tqdm(enumerate(loader), total=len(loader),
                        desc=f"Epoch {epoch_idx + 1}/{self.config.training.num_epochs}")
        for batch_idx, batch in progress:
            behavior = batch["behavior"].to(self.device, non_blocking=True)
            comment_indices = batch["comment_indices"].to(self.device, non_blocking=True)
            topic_indices = batch["topic_indices"].to(self.device, non_blocking=True)
            sequence_labels = batch["sequence_label"].to(self.device, non_blocking=True)
            class_labels = batch["class_label"].to(self.device, non_blocking=True)

            sequence_outputs, class_outputs, _, _ = model(
                comment_indices=comment_indices,
                topic_indices=topic_indices,
                behavior=behavior,
                graph_dict=graph_dict,
                graph_keys=list(batch["graph_key"]),
            )
            sequence_loss = sequence_loss_fn(sequence_outputs, sequence_labels)
            class_loss = class_loss_fn(class_outputs, class_labels)
            loss = loss_weights.get("sequence", 1.0) * sequence_loss + loss_weights.get("class", 1.0) * class_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(
            self,
            model,
            loader,
            graph_dict,
            fold_idx: int,
            epoch_idx: int,
            model_name: str,
            num_parameters: int,
            task_mode: str = "multi_task",
    ) -> Dict:
        model.eval()
        true_sequence_list, pred_sequence_list = [], []
        true_class_list, pred_class_list, pred_class_probs_list = [], [], []
        categorical_attentions, spatial_attentions = [], []
        with torch.no_grad():
            for batch in loader:
                behavior = batch["behavior"].to(self.device, non_blocking=True)
                comment_indices = batch["comment_indices"].to(self.device, non_blocking=True)
                topic_indices = batch["topic_indices"].to(self.device, non_blocking=True)
                sequence_labels = batch["sequence_label"].to(self.device, non_blocking=True)
                class_labels = batch["class_label"].to(self.device, non_blocking=True)
                sequence_output, class_output, categorical_attention, spatial_attention = model(
                    comment_indices=comment_indices,
                    topic_indices=topic_indices,
                    behavior=behavior,
                    graph_dict=graph_dict,
                    graph_keys=list(batch["graph_key"]),
                )

                true_sequence_list.append(sequence_labels)
                pred_sequence_list.append(sequence_output)
                true_class_list.append(class_labels)
                pred_class_list.append(class_output.argmax(dim=1))
                pred_class_probs_list.append(F.softmax(class_output, dim=1).cpu())
                categorical_attentions.append(categorical_attention.detach().cpu())
                spatial_attentions.append(spatial_attention.detach().cpu())

        y_true_sequence = torch.cat(true_sequence_list, dim=0)
        y_pred_sequence = torch.cat(pred_sequence_list, dim=0)
        y_true_class = torch.cat(true_class_list, dim=0)
        y_pred_class = torch.cat(pred_class_list, dim=0)
        y_pred_class_probs = torch.cat(pred_class_probs_list, dim=0).numpy()

        sequence_loss = float(F.mse_loss(y_pred_sequence, y_true_sequence).item())
        sequence_metrics = compute_regression_metrics(y_true_sequence, y_pred_sequence)
        cepstral_distance, dtw_similarity = sequence_shape_distances(y_true_sequence.cpu().numpy(),
                                                                     y_pred_sequence.cpu().numpy(),
                                                                     3)
        class_accuracy = float((y_pred_class == y_true_class).float().mean().item())
        class_f1 = float(f1_score(y_true_class.cpu().numpy(), y_pred_class.cpu().numpy(), average="weighted"))
        __, class_recall, _, _ = precision_recall_fscore_support(
            y_true_class.cpu().numpy(), y_pred_class.cpu().numpy(), average="macro", zero_division=0
        )
        try:
            class_auc = float(
                roc_auc_score(y_true_class.cpu().numpy(), y_pred_class_probs, multi_class="ovr", average="weighted"))
        except ValueError:
            class_auc = float("nan")

        if task_mode == "sequence":
            class_accuracy = class_f1 = class_auc = class_recall = float("nan")
        elif task_mode == "trend":
            sequence_loss = float("nan")
            sequence_metrics = (float("nan"), float("nan"), float("nan"), float("nan"))
            cepstral_distance = dtw_similarity = float("nan")

        shanghai_tz = pytz.timezone("Asia/Shanghai")
        result = {
            "model_name": model_name,
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
        return result

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
            self.config.paths.stpve_epoch_output_path,
        )
        self._upsert_frame(
            self.best_r2_rows(results),
            self.config.paths.stpve_best_epoch_output_path,
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
