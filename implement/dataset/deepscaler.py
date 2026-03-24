import logging
from typing import Optional
import os
import glob
import pickle

from datasets import load_dataset
from math_verify import parse
from sympy import Basic

from data_service.typing.message import Conversation
from implement.dataset.dataset_utils import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_parseable(answer: str) -> bool:
    if not answer:
        return False
    try:
        answer = answer.strip()
        answer = answer.replace(" ", "")
        if not answer:
            return False
        if "," in answer:
            return False
        if len(answer) > 100:
            return False
        if not answer.startswith("$") or not answer.endswith("$"):
            answer = f"${answer}$"
        parsed = parse(answer, raise_on_error=True)
        if not isinstance(parsed, list) or len(parsed) != 2:
            # print("Not exactly two results")
            return False
        if (
            isinstance(parsed[0], Basic)
            and parsed[0].is_number
            and isinstance(parsed[1], str)
        ):
            return True
        return False
        # print("Result is not a SymPy object and a string")
    except Exception:
        return False


class TrainDatasetImpl:
    def __init__(
        self,
        dataset_path: str,
        system_prompt_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        self.dataset = self._load_dataset(dataset_path, "train")

        self.system_prompt = (
            load_prompt(system_prompt_path) if system_prompt_path is not None else None
        )
        self.prompt_template = (
            load_prompt(template_path) if template_path is not None else None
        )

    @staticmethod
    def _load_dataset(dataset_path: str, split: str):
        """Load DeepScaleR dataset from a local Parquet saved in our schema.

        Expectations:
        - Parquet has columns: source, id, question_type, image_num, subject, difficulty,
          question (pickled list with single problem string), solution, answer (LaTeX).
        - No images are included (image_num can be ignored).
        """
        # Resolve parquet file path(s)
        data_files = None
        if os.path.isdir(dataset_path):
            # common layout: <root>/<split>/*.parquet
            candidate_dir = os.path.join(dataset_path, split)
            pattern = os.path.join(candidate_dir, "*.parquet")
            files = sorted(glob.glob(pattern))
            if not files:
                # fallback: any parquet directly under dataset_path
                files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
            if not files:
                raise FileNotFoundError(f"No parquet files found under {dataset_path} (split={split})")
            data_files = files
        else:
            if not dataset_path.endswith(".parquet"):
                raise ValueError(f"Expected a .parquet file or directory, got: {dataset_path}")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Parquet file not found: {dataset_path}")
            data_files = dataset_path

        dataset = load_dataset("parquet", data_files=data_files, split="train")
        original_count = len(dataset)
        logger.info(f"Loaded {original_count} samples from Parquet: {data_files}")

        # Unpickle question into 'problem' string
        def decode_problem(example):
            raw = example.get("question", None)
            problem_text = None
            if raw is not None:
                try:
                    if isinstance(raw, (bytes, bytearray, memoryview)):
                        loaded = pickle.loads(bytes(raw))
                    else:
                        # some parquet readers may restore python objects already
                        loaded = raw
                    if isinstance(loaded, list) and loaded:
                        problem_text = loaded[0]
                    elif isinstance(loaded, str):
                        problem_text = loaded
                except Exception:
                    problem_text = None
            example["problem"] = problem_text if isinstance(problem_text, str) else ""
            return example

        dataset = dataset.map(decode_problem)

        # Shuffle the dataset
        logger.info("Shuffling dataset...")
        dataset = dataset.shuffle()
        logger.info("Dataset shuffled")

        return dataset

    def collate_fn(self, examples):
        conversations = []
        solutions = []

        for example in examples:
            # Get problem and answer from the dataset
            question = example["problem"]
            answer = example["answer"].strip()
            if not answer.startswith("$") or not answer.endswith("$"):
                answer = f"${answer}$"

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