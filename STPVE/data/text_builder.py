from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jieba
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def parse_comment_list(comment_value: Any) -> List[Dict[str, Any]]:
    if isinstance(comment_value, list):
        return comment_value
    if pd.isna(comment_value):
        return []
    try:
        parsed = ast.literal_eval(str(comment_value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def build_character_vocabulary(data: pd.DataFrame, comment_data_path: str) -> Tuple[Dict[str, int], pd.DataFrame]:
    combined_comments = pd.read_csv(comment_data_path, index_col=0)
    combined_comments["time"] = pd.to_datetime(combined_comments["time"])
    required_cols = {"live_room_id", "time", "live_comment"}
    missing_cols = required_cols - set(combined_comments.columns)
    if missing_cols:
        raise KeyError(f"Comment data is missing required columns: {missing_cols}")

    dup_count = combined_comments.duplicated(["live_room_id", "time"]).sum()
    if dup_count > 0:
        raise ValueError(
            f"Comment data contains {dup_count} duplicated live_room_id-time rows. "
            "Please aggregate comments before merging."
        )

    data = pd.merge(data, combined_comments, "left", ["live_room_id", "time"])
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    unique_rooms = data[["live_room_id"]].drop_duplicates()
    for _, room_info in tqdm(unique_rooms.iterrows(), total=len(unique_rooms), desc="Building character vocabulary"):
        room_comments = combined_comments[combined_comments["live_room_id"] == room_info["live_room_id"]]
        for comment_value in room_comments["live_comment"]:
            for comment in parse_comment_list(comment_value):
                content = str(comment.get("content", ""))
                for character in content:
                    if character not in vocabulary:
                        vocabulary[character] = len(vocabulary)
    return vocabulary, data


def text_to_indices(text: str, vocabulary: Dict[str, int], max_len: int = 200) -> List[int]:
    indices = [vocabulary.get(character, vocabulary["<UNK>"]) for character in text[:max_len]]
    if len(indices) < max_len:
        indices += [vocabulary["<PAD>"]] * (max_len - len(indices))
    return indices


def get_comment_indices_for_sample(
        row: pd.Series,
        data: pd.DataFrame,
        vocabulary: Dict[str, int],
        max_len: int = 200,
        window_minutes: int = 5,
) -> Tuple[List[int], str]:
    room_data = data[data["live_room_id"] == row["live_room_id"]].sort_values("time").reset_index(drop=True)
    current_positions = room_data.index[room_data["time"] == row["time"]].tolist()
    if not current_positions:
        return [vocabulary["<PAD>"]] * max_len, ""
    start_time = row["time"] - pd.Timedelta(minutes=window_minutes - 1)
    window_data = room_data[
        (room_data["time"] >= start_time)
        & (room_data["time"] <= row["time"])
        ]
    contents: List[str] = []
    for _, window_row in window_data.iterrows():
        for comment in parse_comment_list(window_row.get("live_comment", "")):
            content = str(comment.get("content", ""))
            if content:
                contents.append(content)
    combined_text = " ".join(contents)
    if not combined_text.strip():
        return [vocabulary["<PAD>"]] * max_len, combined_text
    return text_to_indices(combined_text, vocabulary, max_len), combined_text


def segment_chinese_text(text: str) -> str:
    return " ".join(jieba.cut(text))


def train_lda_model(all_texts: List[str], num_topics: int = 20, lda_max_features: int = 2000):
    tokenized_texts = []
    for text in tqdm(all_texts, desc="Tokenizing comments for LDA"):
        segmented_text = segment_chinese_text(text)
        tokenized_texts.append(segmented_text if segmented_text.strip() else "__EMPTY__")
    vectorizer = CountVectorizer(max_features=lda_max_features, stop_words=None, min_df=1)
    document_term_matrix = vectorizer.fit_transform(tokenized_texts)
    if document_term_matrix.shape[1] == 0:
        raise ValueError("No vocabulary was found for LDA. Check comments and tokenization.")
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=20)
    lda.fit(document_term_matrix)
    return vectorizer, lda


def get_top_topic_indices(text: str, vectorizer: CountVectorizer, lda: LatentDirichletAllocation, top_k: int = 5) -> \
        List[int]:
    if not str(text).strip():
        return [0] * top_k
    bow = vectorizer.transform([segment_chinese_text(str(text))])
    topic_distribution = lda.transform(bow)[0]
    return topic_distribution.argsort()[-top_k:][::-1].tolist()


def save_pickle(obj: Any, path: str | Path) -> None:
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)
