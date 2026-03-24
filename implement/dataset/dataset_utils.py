from typing import List, Optional, Tuple, Union
import re

from datasets import (
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names,
    Dataset,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_configs_dataset(dataset_path: str, split: str) -> Dataset:
    """
    Load all available configs from a dataset and concatenate them.

    Args:
        dataset_path: The path/name of the dataset (e.g., 'MMMU/MMMU')
        split: The split to load (e.g., 'dev', 'validation', 'test')

    Returns:
        A concatenated dataset containing all configs
    """
    try:
        # Get all available configs
        configs = get_dataset_config_names(dataset_path)
        logger.info(f"Found {len(configs)} configs for {dataset_path}: {configs}")

        # Load each config and store in a list
        datasets_list = []
        for config in configs:
            try:
                dataset = load_dataset(dataset_path, config, split=split)
                logger.info(f"Loaded config '{config}' with {len(dataset)} samples")
                datasets_list.append(dataset)
            except Exception as e:
                logger.warning(f"Failed to load config '{config}': {e}")
                continue

        if not datasets_list:
            raise ValueError(f"No configs could be loaded for {dataset_path}")

        # Concatenate all datasets
        combined_dataset = concatenate_datasets(datasets_list)
        logger.info(f"Combined dataset has {len(combined_dataset)} total samples")

        return combined_dataset

    except Exception as e:
        logger.error(f"Error loading all configs for {dataset_path}: {e}")
        raise


def load_prompt(prompt_path: str):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def index_to_letter(index: int):
    if not isinstance(index, int):
        raise ValueError(f"Index must be an integer. Got: {type(index)}")
    if index < 0:
        raise ValueError(f"Index must be non-negative. Got: {index}")
    if index >= 26:
        raise ValueError(f"Index must be less than 26. Got: {index}")

    return chr(index + ord("A"))


def format_multi_choice(question: Optional[str], choices: Optional[List[str]]):
    if question is None:
        question = ""
    elif not isinstance(question, str):
        raise ValueError(f"Question must be a string or None. Got: {type(question)}")

    if choices is None:
        choices = []
    elif not isinstance(choices, list):
        raise ValueError(f"Choices must be a list or None. Got: {type(choices)}")
    elif not all(isinstance(choice, str) for choice in choices):
        raise ValueError(f"Choices must be a list of strings. Got: {type(choices)}")

    if len(choices) > 26:
        raise ValueError(f"Too many choices: {len(choices)}. Maximum allowed is 26.")

    if len(choices) == 0:
        return question.strip()

    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = index_to_letter(i)
        choice = choice.strip()
        formatted_choices.append(f"{letter}. {choice}")

    return (question.strip() + "\n" + "\n".join(formatted_choices)).strip()


def split_interleaved_prompt(
    prompt: str, pattern: str = r"<image \d+>"
) -> Tuple[List[Union[str, None]], int]:
    """
    Split interleaved text and image prompts.

    Args:
        prompt: The input prompt containing text and image placeholders
        pattern: Regex pattern to match image placeholders

    Returns:
        Tuple of (split_parts, image_count) where:
        - split_parts: List alternating between text parts (str) and image placeholders (None)
                      No empty strings are included
        - image_count: Number of images found
    """
    # Find all image matches
    image_matches = list(re.finditer(pattern, prompt))
    image_count = len(image_matches)

    if image_count == 0:
        # No images found, return the original prompt if not empty
        return ([prompt.strip()] if prompt.strip() else [], 0)

    # Split the prompt at image positions
    parts = []
    last_end = 0

    for match in image_matches:
        # Add text before this image (if not empty)
        text_part = prompt[last_end : match.start()].strip()
        if text_part:
            parts.append(text_part)

        # Add None to represent image placeholder
        parts.append(None)
        last_end = match.end()

    # Add remaining text after last image (if not empty)
    remaining_text = prompt[last_end:].strip()
    if remaining_text:
        parts.append(remaining_text)

    return parts, image_count


if __name__ == "__main__":

    def test_case(name: str, prompt: str, pattern: str = r"<image \d+>"):
        print(f"\n=== {name} ===")
        print(f"Input: {prompt}")
        parts, image_count = split_interleaved_prompt(prompt, pattern)
        print(f"Image count: {image_count}")
        print("Split parts:")
        for i, part in enumerate(parts):
            if part is None:
                print(f"  Part {i + 1}: [IMAGE_PLACEHOLDER]")
            else:
                print(f"  Part {i + 1}: {part[:50]}{'...' if len(part) > 50 else ''}")
        print()

    # Test case 1: Original example with interleaved text and images
    test_case(
        "Case 1: Interleaved text and images",
        """To find out if there is any relationship between teacher's pay and per pupil expenditure in public schools, Table 1 shows the LS regression results of a regression relating the average annual teacher salary in thousands of dollars (Salary) as a function of the spending on public schools per pupil in thousands of dollars (Expenditure) for the 51 states of the US in 1985. <image 1> Looking for evidence about differences between three geographical regions in the US: Northeast and North Central (21 states), South (17 states) and West (13 states), we defined three dummy variables: D1, which is equal to 1 if the state is in the West, and equal to 0 otherwise; D2 which is equal to 1 if the state is in the Northeast and North Central region, and equal to 0 otherwise; and D3 which is equal to 1 if the state is in the South, and equal to 0 otherwise. Adding these variables to the previous model yields the results shown in Table 2. <image 2> According to Tables 1 and 2:""",
    )

    # Test case 2: Consecutive images
    test_case(
        "Case 2: Consecutive images",
        "Here are two charts: <image 1><image 2> Please analyze both.",
    )

    # Test case 3: Only images
    test_case("Case 3: Only images", "<image 1><image 2><image 3>")

    # Test case 4: Starting with image
    test_case(
        "Case 4: Starting with image",
        "<image 1> This image shows important data. Please review it carefully.",
    )

    # Test case 5: Ending with image
    test_case("Case 5: Ending with image", "Please look at this diagram: <image 1>")

    # Test case 6: No images
    test_case(
        "Case 6: No images", "This is just a simple text prompt without any images."
    )

    # Test case 7: Empty prompt
    test_case("Case 7: Empty prompt", "")

    # Test case 8: Only whitespace
    test_case("Case 8: Only whitespace", "   \n\t  ")

    # Test case 9: Multiple consecutive images with text
    test_case(
        "Case 9: Multiple consecutive images with text",
        "First part <image 1><image 2><image 3> Middle part <image 4><image 5> Final part",
    )
