import logging
import json
import re
import random
from typing import Optional
from pathlib import Path

from datasets import Dataset

from data_service.typing.message import Conversation
from implement.dataset.dataset_utils import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainDatasetImpl:
    def __init__(
        self,
        dataset_path: str,
        system_prompt_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        self.dataset = self._load_dataset(dataset_path)

        self.system_prompt = (
            load_prompt(system_prompt_path) if system_prompt_path is not None else None
        )
        self.prompt_template = (
            load_prompt(template_path) if template_path is not None else None
        )

    @staticmethod
    def _has_boxed_answer(solution: str) -> bool:
        """Check if solution contains a boxed answer."""
        boxed_pattern = r'\\boxed\{(.+?)\}'
        matches = re.findall(boxed_pattern, solution)
        return len(matches) > 0

    @staticmethod
    def _load_dataset(dataset_path: str):
        """Load MATH dataset from directory structure"""
        logger.info(f"Loading MATH dataset from: {dataset_path}")
        data = []
        filtered_count = 0
        
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Iterate through all subdirectories (e.g., algebra, geometry, etc.)
        for subject_dir in sorted(dataset_dir.iterdir()):
            if subject_dir.is_dir():
                logger.info(f"Loading subject: {subject_dir.name}")
                # Iterate through all JSON files in the subject directory
                for json_file in sorted(subject_dir.glob("*.json")):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            item = json.load(f)
                            # Filter out items without boxed answers
                            solution = item.get("solution", "")
                            if TrainDatasetImpl._has_boxed_answer(solution):
                                data.append(item)
                            else:
                                filtered_count += 1
                    except Exception as e:
                        logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Total loaded samples: {len(data)}")
        logger.info(f"Filtered out {filtered_count} samples without boxed answers")
        
        # Shuffle the data
        random.shuffle(data)
        logger.info("Data shuffled")
        
        return Dataset.from_list(data)

    @staticmethod
    def extract_answer(solution: str) -> str:
        """Extract the final answer from the solution.
        The answer is typically in \\boxed{...} format.
        This method handles nested braces correctly."""
        
        # Find all occurrences of \boxed{
        boxed_start = r'\\boxed\{'
        positions = [m.start() for m in re.finditer(boxed_start, solution)]
        
        if not positions:
            logger.warning("No boxed answer found in solution")
            return ""
        
        def extract_balanced_braces(text, start_pos):
            """Extract content within balanced braces starting from start_pos.
            start_pos should point to the opening brace."""
            brace_count = 0
            i = start_pos
            
            while i < len(text):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the matching closing brace
                        return text[start_pos + 1:i]
                i += 1
            
            # If we get here, braces weren't balanced
            return ""
        
        # Extract all boxed answers
        answers = []
        for pos in positions:
            # Find the position of the opening brace after \boxed
            brace_pos = pos + len(r'\boxed')
            answer = extract_balanced_braces(solution, brace_pos)
            if answer:
                answers.append(answer)
        
        if answers:
            # Return the last boxed answer
            return answers[-1]
        
        logger.warning("No boxed answer found in solution")
        return ""

    def collate_fn(self, examples):
        conversations = []
        solutions = []

        for example in examples:
            question = example.get("problem")
            solution_text = example.get("solution", "")
            answer = self.extract_answer(solution_text)

            if self.prompt_template is not None:
                prompt = question + "\n" + self.prompt_template
            else:
                prompt = question

            message_list = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            if self.system_prompt is not None:
                message_list.insert(
                    0,
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                )

            conversation = Conversation(messages=message_list)
            conversations.append(conversation)
            solutions.append(answer)

        return conversations, solutions, {}


TestDatasetImpl = TrainDatasetImpl

