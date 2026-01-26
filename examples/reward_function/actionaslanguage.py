# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def reasoning_length_reward(response: str, max_tokens: int = 500) -> float:
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    reasoning = match.group(1) if match else ""
    token_count = len(re.findall(r"\S+", reasoning))
    return 0.0 if token_count > max_tokens else 1.0


def _try_literal_eval(text: str) -> Optional[Any]:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _parse_action_matrix(value: Any) -> Optional[list[list[float]]]:
    parsed = value
    if isinstance(value, str):
        try:
            boxed = extract_boxed_content(value)
        except Exception:
            boxed = None
        candidate = boxed if boxed else value
        parsed = _try_literal_eval(candidate)
        if parsed is None and "[" in candidate and "]" in candidate:
            start = candidate.find("[")
            end = candidate.rfind("]") + 1
            parsed = _try_literal_eval(candidate[start:end])

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 10:
        return None

    matrix: list[list[float]] = []
    for row in parsed:
        if not isinstance(row, (list, tuple)) or len(row) != 7:
            return None
        try:
            matrix.append([float(item) for item in row])
        except (TypeError, ValueError):
            return None

    return matrix


def _dense_action_reward(
    predicted: list[list[float]], target: list[list[float]], mode: str = "l1"
) -> list[list[float]]:
    if mode == "l1":
        dense: list[list[float]] = []
        for pred_row, target_row in zip(predicted, target):
            row: list[float] = []
            for pred, tgt in zip(pred_row, target_row):
                reward = 1.0 - abs(pred - tgt)
                if reward < 0.0:
                    reward = 0.0
                row.append(reward)
            dense.append(row)
        return dense
    if mode == "match":
        return [row[:] for row in target]
    raise ValueError(f"Unsupported dense_reward_mode: {mode}")


def _mean_matrix(matrix: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for row in matrix:
        for value in row:
            total += value
            count += 1
    return total / count if count else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    length_weight: float = 0.1,
    max_reasoning_tokens: int = 500,
    dense_reward_mode: str = "l1",
    return_dense: bool = False,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        action_source = reward_input.get("action", response)
        action_pred = _parse_action_matrix(action_source)
        action_target = _parse_action_matrix(reward_input.get("ground_truth"))
        format_score = format_reward(response)
        length_score = reasoning_length_reward(response, max_tokens=max_reasoning_tokens)
        dense_reward = None
        if action_pred is not None and action_target is not None:
            dense_reward = _dense_action_reward(action_pred, action_target, mode=dense_reward_mode)
            dense_score = _mean_matrix(dense_reward)
        else:
            dense_score = accuracy_reward(response, reward_input["ground_truth"])

        overall = (
            (1 - format_weight - length_weight) * dense_score
            + format_weight * format_score
            + length_weight * length_score
        )
        score: dict[str, Any] = {
            "overall": overall,
            "format": format_score,
            "length": length_score,
            "dense": dense_score,
        }
        if return_dense and dense_reward is not None:
            score["dense_reward"] = dense_reward
        scores.append(score)

    return scores
