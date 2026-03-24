from typing import List
import json
import logging
import os
from datetime import datetime

from data_service.typing.grpo_data import GRPOData
from utils.metrics import LocalMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_data(
    data: List[List[GRPOData]],
    output_path: str,
) -> None:
    """
    Save List[List[GRPOData]] to a JSON file format.
    Works for both rollout data and evaluation data.
    Only saves conversation data without statistics or metadata.

    Args:
        data: List of lists of GRPOData to save
        output_path: Path to output JSON file
    """

    def process_content(content):
        """Process message content, truncating image base64 data"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            processed_content = []
            for item in content:
                if isinstance(item, dict):
                    processed_item = item.copy()
                    if item.get("type") == "image" and "image" in item:
                        image_data = item["image"]
                        # Truncate base64 image data to first 50 characters
                        if len(image_data) > 50:
                            processed_item["image"] = image_data[:50] + "...[TRUNCATED]"
                    processed_content.append(processed_item)
                else:
                    processed_content.append(item)
            return processed_content
        else:
            return content

    def extract_prompt_and_response(conversation):
        """Extract prompt (all messages except last) and response (last message)"""
        messages = conversation.messages
        if len(messages) == 0:
            return [], None

        # Process prompt messages (all except last)
        prompt_messages = []
        for msg in messages[:-1]:
            processed_msg = {"role": msg.role, "content": process_content(msg.content)}
            prompt_messages.append(processed_msg)

        # Process response message (last one)
        last_msg = messages[-1]
        response = {"role": last_msg.role, "content": process_content(last_msg.content)}

        return prompt_messages, response

    # Process all data - simplified structure
    processed_data = []

    for batch_idx, batch in enumerate(data):
        if not batch:
            continue

        # Extract common prompt from first item
        first_item = batch[0]
        common_prompt, _ = extract_prompt_and_response(first_item.conversation)

        batch_data = {
            "batch_index": batch_idx,
            "batch_size": len(batch),
            "common_prompt": common_prompt,
            "solution": first_item.solution,
            "responses": [],
        }

        for item_idx, item in enumerate(batch):
            # Extract only the response part
            _, response = extract_prompt_and_response(item.conversation)
            response_data = {
                "response": response,
                "rewards": item.rewards,
                "reward_reasons": item.reward_reasons,
                "label": item.label,
                "response_token_num": len(item.response_token_ids),
                "stop_reason": item.stop_reason,
            }
            batch_data["responses"].append(response_data)

        processed_data.append(batch_data)

    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    logger.info(
        f"Successfully saved {len(data)} batches with {sum(len(batch) for batch in data)} total items to {output_path}"
    )


def save_data_generation_results(
    step: int,
    raw_data: List[List[GRPOData]],
    metrics: LocalMetrics,
    output_dir: str,
    suffix: str = "",
) -> None:
    """Save rollout data generation results."""
    # Save metrics as step_N_metrics.json
    if suffix:
        metrics_file = os.path.join(output_dir, f"step_{step}_metrics_{suffix}.json")
    else:
        metrics_file = os.path.join(output_dir, f"step_{step}_metrics.json")
    metrics_dict = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics.to_flat_dict() if hasattr(metrics, "to_flat_dict") else {},
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    # Save data as step_N.json using the unified save function
    if suffix:
        data_file = os.path.join(output_dir, f"step_{step}_{suffix}.json")
    else:
        data_file = os.path.join(output_dir, f"step_{step}.json")
    save_data(data=raw_data, output_path=data_file)

    if suffix:
        logger.info(f"Saved {suffix} data generation results for step {step}")
    else:
        logger.info(f"Saved data generation results for step {step}")
    logger.info(f"- Data: {data_file}")
    logger.info(f"- Metrics: {metrics_file}")


def save_evaluation_results(
    step: int,
    dataset_name: str,
    results: List[List[GRPOData]],
    metrics: LocalMetrics,
    output_dir: str,
) -> None:
    """Save evaluation results using GRPOData format.

    Args:
        step: Current training step
        dataset_name: Name of the evaluation dataset
        results: List of lists of GRPOData objects with evaluation results (supports n>1)
        metrics: LocalMetrics object with computed metrics
        output_dir: Output directory for saving results
    """
    # Save evaluation metrics as test_dataset_step_N_metrics.json
    metrics_file = os.path.join(
        output_dir,
        f"{dataset_name}_step_{step}_metrics.json",
    )
    metrics_dict = {
        "test_dataset": dataset_name,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics.to_flat_dict() if hasattr(metrics, "to_flat_dict") else {},
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    # Save evaluation data as test_dataset_step_N.json using unified save function
    data_file = os.path.join(output_dir, f"{dataset_name}_step_{step}.json")
    save_data(data=results, output_path=data_file)

    logger.info(f"Saved evaluation results for dataset '{dataset_name}'")
    logger.info(f"- Data: {data_file}")
    logger.info(f"- Metrics: {metrics_file}")
