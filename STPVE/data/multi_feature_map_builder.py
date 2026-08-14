from STPVE.utils import _reshape_history_matrix
import pandas as pd


def build_multi_category_feature_map(
        group: pd.DataFrame,
        history_length: int,
        prediction_length: int,
        target_column: str,
) -> pd.DataFrame:
    """Create rolling-window samples compatible with the forecasting model.
    """
    passive_features = [
        "EnterCnt_s", "OnlineCnt_s", "LeaveCnt_s", "ShowCnt_s", "WatchRatio_s",
        "ShowCnt_l", "WatchCnt_l", "WatchRatio_l", "WatchConvRatio_l"
    ]

    interactive_features = [
        "FollowCnt_s", "ComCnt_s", "ClubCnt_s", "FollowRatio_s", "InterRatio_s",
        "StayTime_l", "FollowRatio_l", "AvgComCnt_l", "InterRatio_l"
    ]
    transactional_features = [
        "ConvCnt_s", "ConvAmt_s", "ConvOrder_s", "RepeatRatio_s", "GPM_s",
        "GPM_l", "RepeatRatio_l", "ClickRatio_l", "ConvAmt_l"
    ]
    group = group.sort_values("time").reset_index().copy()
    group["PASSIVE"] = group[passive_features].values.tolist()
    group["INTERACTIVE"] = group[interactive_features].values.tolist()
    group["TRANSACTIONAL"] = group[transactional_features].values.tolist()

    rows = []
    max_start = len(group) - history_length - prediction_length + 1
    if max_start <= 0:
        return pd.DataFrame()
    feature_frame = group[["PASSIVE", "INTERACTIVE", "TRANSACTIONAL"]].to_numpy()
    target_values = group[target_column].fillna(0.0).to_numpy()
    for start_idx in range(max_start):
        current_idx = start_idx + history_length - 1
        future_start = current_idx + 1
        future_end = future_start + prediction_length
        row = group.iloc[current_idx].copy()
        row["behavior"] = _reshape_history_matrix(
            feature_frame[start_idx: start_idx + history_length],
        )
        row["sequence_label"] = target_values[future_start:future_end].astype(float).tolist()
        rows.append(row)
    return pd.DataFrame(rows)
