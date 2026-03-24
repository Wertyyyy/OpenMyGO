from config.utils import get_path

_model_impl = "implement.model.qwen"
_model_name = "Qwen2.5-3B-Instruct"

_pregenerate_steps = 1
_run_name = f"qwen2.5-3b-pg{_pregenerate_steps}"

_total_gpu = 8
_gpu_for_generation = 2
_gpu_for_training = _total_gpu - _gpu_for_generation

_max_prompt_length = 512
_max_response_length = 2048

model = {
    "impl_path": _model_impl,
    "init_params": {
        "pretrained_model_name_or_path": get_path("model", _model_name),
        "use_cache": False,
    },
}

processor = {
    "impl_path": _model_impl,
    "init_params": {
        "pretrained_model_name_or_path": get_path("model", _model_name)
    },
    "apply_chat_template_params": {},
}

dataset = {
    "train": {
        "impl_path": "implement.dataset.gsm8k",
        "init_params": {
            "dataset_path": get_path("data", "GSM8K/train.jsonl"),
        },
    },
    "test": {
        "GSM8K": {
            "impl_path": "implement.dataset.gsm8k",
            "init_params": {
                "dataset_path": get_path("data", "GSM8K/test.jsonl"),
            },
        }
    },
    "system_prompt_path": None,
    "template_path": None,
}

reward = {
    "accuracy": {
        "impl_path": "implement.reward.latex_math_async",
        "init_params": {},
    },
}

network = {
    "vllm_port": 41000,
    "nccl_port": 42000,
    "tf_port": 40000,
    "data_port": 43000,
    "vllm_host": "0.0.0.0",
    "tf_host": "*.*.*.*",
    "data_host": "0.0.0.0",
}

tf_server = {
    "device": "cuda:0",
    "token_budget": 45000,
}

vllm_server = {
    "visible_devices": ",".join(str(i) for i in range(_gpu_for_training, _total_gpu)),
    "data_parallel_size": _gpu_for_generation,
    "llm_params": {
        "model": get_path("model", _model_name),
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 2048,
        "max_model_len": _max_prompt_length + _max_response_length + 100,
        "enforce_eager": False,
        "enable_prefix_caching": False,
    },
}

data_server = {
    "generation_sampling_params": {
        "temperature": 1.0,
        "n": 16,
        "max_tokens": _max_response_length,
    },
    "evaluation_sampling_params": {
        "temperature": 0.0,
        "n": 1,
        "max_tokens": _max_response_length,
    },
    "discard": {
        "length": True,
        "abort": True,
    },
    "max_prompt_length": _max_prompt_length,
    "max_length": _max_prompt_length + _max_response_length + 50,
    "pregenerate_steps": _pregenerate_steps,
    "global_batch_size": 16,
    "token_budget": 12000,
    "max_micro_step_num": 4,
    "gpu_num": _gpu_for_training,
}

training = {
    "lr": 1e-6,
    "total_steps": 465,
    "grpo_beta": 0,
    "save_dir": get_path("model", _run_name),
    "save_steps": 100,
    "eval_steps": 40,
    "project_name": "OpenMyGO",
    "run_name": _run_name,
    "max_grad_norm": 0.5,
    "loss_type": "global",
    "use_std": False,
}
