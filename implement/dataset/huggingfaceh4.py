import logging
from typing import Optional

from datasets import load_dataset

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
        take_every_n: int = 0,
        split: str = "test",
    ):
        self.dataset = self._load_dataset(dataset_path, split=split, take_every_n=take_every_n)

        self.system_prompt = (
            load_prompt(system_prompt_path) if system_prompt_path is not None else None
        )
        self.prompt_template = (
            load_prompt(template_path) if template_path is not None else None
        )

    @staticmethod
    def _load_dataset(dataset_path: str, take_every_n: int = 0, split: str = "test"):
        logger.info(f"Loading dataset from: {dataset_path} with split: {split}")
        
        # Load the test split (only split available)
        dataset = load_dataset(dataset_path, split=split)
        
        logger.info(f"Total loaded samples: {len(dataset)}")

        # Optionally select every N-th sample
        if isinstance(take_every_n, int) and take_every_n > 0:
            indices = list(range(0, len(dataset), take_every_n))
            dataset = dataset.select(indices)
            logger.info(
                f"Sampling every {take_every_n} samples -> kept {len(dataset)} samples"
            )
        return dataset

    def collate_fn(self, examples):
        conversations = []
        solutions = []

        for example in examples:
            question = example.get("problem")
            # The answer field already contains the extracted answer
            answer = example.get("answer", "")

            if self.prompt_template is not None:
                prompt = self.prompt_template.format(instruction=question)
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

