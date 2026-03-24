import logging
from typing import Optional

from datasets import load_dataset, Features, Value

from data_service.typing.message import Conversation
from implement.dataset.dataset_utils import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDatasetImpl:
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
    def _load_dataset(dataset_path):
        # Define unified schema with all string types
        unified_features = Features({
            'problem_idx': Value('string'),
            'problem': Value('string'),
            'answer': Value('string'),
            'problem_type': Value('string'),
        })

        logger.info(f"Loading dataset: {dataset_path}")
        dataset = load_dataset(dataset_path, split="train")
        num_samples = len(dataset)
        logger.info(f"Loaded {num_samples} samples from {dataset_path}")
        
        # Convert all fields to string type first
        def convert_to_str(example):
            return {
                'problem_idx': str(example['problem_idx']),
                'problem': str(example['problem']),
                'answer': str(example['answer']),
                'problem_type': str(example['problem_type']),
            }
        
        dataset = dataset.map(convert_to_str)
        
        # Cast to unified schema to ensure type consistency
        dataset = dataset.cast(unified_features)
        return dataset

    def collate_fn(self, examples):
        conversations = []
        solutions = []

        for example in examples:
            # Get problem and answer from the dataset
            question = example["problem"]
            answer = f"""${example["answer"]}$"""

            # Apply prompt template if provided
            if self.prompt_template is not None:
                prompt = question + self.prompt_template
            else:
                prompt = question

            # Build message list
            message_list = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            # Add system prompt if provided
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
