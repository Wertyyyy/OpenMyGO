from typing import Optional, Dict
import importlib
import logging
import random

import fire
from tqdm import tqdm

from config.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_certain_dataset(
    impl_path: str,
    init_params: Dict,
    split: str,
    system_prompt_path: Optional[str] = None,
    template_path: Optional[str] = None,
):
    dataset_module = importlib.import_module(impl_path)
    if split == "train":
        dataset = dataset_module.TrainDatasetImpl(
            system_prompt_path=system_prompt_path,
            template_path=template_path,
            **init_params
        )
    elif split == "test":
        dataset = dataset_module.TestDatasetImpl(
            system_prompt_path=system_prompt_path,
            template_path=template_path,
            **init_params
        )
    else:
        raise ValueError(f"Invalid split: {split}")

    # Test basic properties
    logger.info(f"Dataset size: {len(dataset.dataset)}")
    assert len(dataset.dataset) > 0, "Dataset should not be empty"

    # Sample 5 random samples
    num_samples = min(5, len(dataset.dataset))
    indices = random.sample(range(len(dataset.dataset)), num_samples)
    for idx in indices:
        logger.info(f"====== sample {idx} ======")
        sample_data, solution, metadata = dataset.collate_fn([dataset.dataset[idx]])
        sample_data[0].pprint()
        logger.info(f"Solution: {solution[0]}")
        logger.info(f"Metadata: {metadata}")

    # Test dataset indexing
    for item in tqdm(dataset.dataset):
        dataset.collate_fn([item])


def test(config_file: str):
    logger.info(f"Starting test with config: {config_file}")

    config = load_config(config_file)

    logging.info(f"Testing training dataset, {config.dataset.train}")
    test_certain_dataset(
        impl_path=config.dataset.train.impl_path,
        init_params=config.dataset.train.init_params,
        split="train",
        system_prompt_path=config.dataset.system_prompt_path,
        template_path=config.dataset.template_path,
    )

    for name, item in config.dataset.test.items():
        logging.info(f"Testing test dataset: {name}, {item}")
        test_certain_dataset(
            impl_path=item.impl_path,
            init_params=item.init_params,
            split="test",
            system_prompt_path=config.dataset.system_prompt_path,
            template_path=config.dataset.template_path,
        )


if __name__ == "__main__":
    fire.Fire(test)
