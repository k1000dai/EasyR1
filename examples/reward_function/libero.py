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
import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer
import numpy as np
from transformers import AutoProcessor

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def string_to_action_tokens(action_string: str) -> list[int]:
    # Split the action string into individual actions
    # "1,2,3" -> [1,2,3]
    actions = action_string.split(",")
    # Convert each string to integar
    action_tokens = [int(action.strip()) for action in actions if action.strip()]
    return action_tokens


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def action_token_reward(response: str, ground_truth: str) -> float:
    # ground truth is a tokenized string of actions
    # matched \boxed{} parts in response
    answer = extract_boxed_content(response)
    try:
        answer_tokens = string_to_action_tokens(answer)
        answer_tokens = np.array([answer_tokens])
        ground_truth_tokens = string_to_action_tokens(ground_truth)
        ground_truth_tokens = np.array([ground_truth_tokens])
        anwer_action_detokenized = tokenizer.decode(answer_tokens)
        ground_truth_action_detokenized = tokenizer.decode(ground_truth_tokens)
        # answer_action_detokenized [1,10,7]
        error = np.mean(np.abs(np.array(anwer_action_detokenized) - np.array(ground_truth_action_detokenized)))
        return max(0, 1 - error)
    except Exception as e:
        print(f"Error in action_token_reward: {e}")
        return 0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        action_token_score = action_token_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + action_token_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "action_token": action_token_score,
            }
        )

    return scores
