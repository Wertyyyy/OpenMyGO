from typing import List, Dict, Any
import logging
import time
import json
import fire
import asyncio
import uuid
import pickle
import os

import ray
from ray import serve
from ray.serve.handle import RequestMetadata
from ray.serve.config import RequestRouterConfig
from ray.serve.schema import LoggingConfig
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel

from data_service.typing.message import Conversation
from config.utils import load_config, ConfigItem
from vllm_service.detector.repetition_incremental import (
    IncrementalTokenRepetitionDetector,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerateRequest(BaseModel):
    conversation: Conversation
    sampling_params: Dict[str, Any]


class UpdateWeightsNCCLRequest(BaseModel):
    names: List[str]
    shapes: List[List[int]]
    dtypes: List[str]


class WorkerExtension:
    def init_weight_update_group(self, master_address, master_port, rank, world_size):
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        logger.info(
            f"Worker received NCCL init request: rank {rank}, world_size {world_size}"
        )

        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )
        self.pynccl = PyNcclCommunicator(pg, device=self.device)

    def update_weight(
        self, names: List[str], shapes: List[List[int]], dtypes: List[str]
    ):
        import torch

        logging.info("Worker: Updating weights via NCCL start")
        start_time = time.time()

        dtype_mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
        }

        # names/shapes/dtypes carry packed-chunk metadata.
        # names[i] is JSON metadata for chunk i, shapes[i] is [total_numel], dtypes[i] is chunk dtype.
        self.pynccl.group.barrier()
        for chunk_metadata_json, chunk_shape, dtype in zip(names, shapes, dtypes):
            if dtype not in dtype_mapping:
                raise ValueError(f"Unsupported dtype for NCCL packed transfer: {dtype}")

            chunk_metadata = json.loads(chunk_metadata_json)
            tensors_metadata = chunk_metadata["tensors"]
            total_numel = int(chunk_shape[0])

            logger.debug(
                f"Updating packed chunk {chunk_metadata.get('chunk_id')} "
                f"with {len(tensors_metadata)} tensors, dtype: {dtype}, numel: {total_numel}"
            )
            torch_dtype = dtype_mapping[dtype]
            packed_weight = torch.empty(
                (total_numel,),
                dtype=torch_dtype,
                device=torch.device(self.device),
            )

            self.pynccl.broadcast(
                packed_weight, src=0, stream=torch.cuda.current_stream()
            )
            # self.pynccl.group.barrier()

            offset = 0
            unpacked_weights = []
            for tensor_metadata in tensors_metadata:
                tensor_name = tensor_metadata["name"]
                tensor_shape = tensor_metadata["shape"]
                tensor_numel = int(tensor_metadata["numel"])

                tensor = packed_weight[offset : offset + tensor_numel].view(
                    tensor_shape
                )
                unpacked_weights.append((tensor_name, tensor))
                offset += tensor_numel

            if offset != total_numel:
                raise ValueError(
                    f"Packed chunk size mismatch: metadata numel={offset}, received numel={total_numel}"
                )

            self.model_runner.model.load_weights(weights=unpacked_weights)

        end_time = time.time()
        logger.info(f"Weights update completed in {end_time - start_time} seconds")


@serve.deployment
class VLLMServiceDeployment:
    def __init__(self, config: dict):
        self.config = ConfigItem(config)
        self.context = serve.get_replica_context()
        self._init_engine()

        ray_logger = logging.getLogger("ray.data")
        ray_logger.setLevel(logging.WARNING)

    async def _periodic_stats_logging(self):
        while True:
            await asyncio.sleep(1)
            await self.engine.do_log_stats()

    def _init_engine(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                **self.config.vllm_server.llm_params,
                worker_extension_cls="vllm_service.vllm_server_ray.WorkerExtension",
            )
        )
        self.stats_task = asyncio.create_task(self._periodic_stats_logging())
        self.active_requests = set()  # Track active request IDs
        logger.info(f"VLLMServiceDeployment {self.context.rank} ready")

    async def generate(self, request: GenerateRequest):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        # Build sampling params from request
        # Extract n and max_tokens from sampling_params
        sampling_params_dict = request.sampling_params.copy()

        # Ensure required parameters are present
        if "n" not in sampling_params_dict:
            raise ValueError("n must be specified in sampling_params")
        if "max_tokens" not in sampling_params_dict:
            raise ValueError("max_tokens must be specified in sampling_params")

        # Add output_kind for repetition detection
        sampling_params_dict["output_kind"] = RequestOutputKind.CUMULATIVE

        sampling_params = SamplingParams(**sampling_params_dict)
        request_id = str(uuid.uuid4().hex)

        # Track active request
        self.active_requests.add(request_id)

        # Get n for completion tracking
        n = sampling_params_dict["n"]

        anomaly_check_interval = 8
        anomaly_detectors = [
            {
                "name": "repetition",
                "finish_reason": "repetition_abort",
                "states": {},
                "last_token_lens": {},
            },
        ]

        def get_sequence_request_id(output_index: int) -> str:
            if n > 1:
                return f"{output_index}_{request_id}"
            return request_id

        try:
            tokenizer = await self.engine.get_tokenizer()
            prompt_text = tokenizer.apply_chat_template(
                request.conversation.model_dump()["messages"],
                tokenize=False,
                add_generation_prompt=True,
                **self.config.processor.apply_chat_template_params,
            )
            prompt_data = {"prompt": prompt_text}
            prompt_images = request.conversation.get_images()
            if prompt_images:
                prompt_data["multi_modal_data"] = {"image": prompt_images}

            results_generator = self.engine.generate(
                prompt=prompt_data,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            prompt_token_ids = None
            completed_indices = set()

            async for request_output in results_generator:
                if prompt_token_ids is None:
                    prompt_token_ids = request_output.prompt_token_ids

                for output in request_output.outputs:
                    if (
                        output.finish_reason is not None
                        and output.index not in completed_indices
                    ):
                        completed_indices.add(output.index)
                        for detector_cfg in anomaly_detectors:
                            detector_cfg["states"].pop(output.index, None)
                            detector_cfg["last_token_lens"].pop(output.index, None)
                        yield {
                            "index": output.index,
                            "completion": output.text,
                            "finish_reason": output.finish_reason,
                            "prompt_token_ids": prompt_token_ids or [],
                            "response_token_num": len(output.token_ids),
                        }

                    # Check for repetition (skip if already finished)
                    if (
                        output.finish_reason is None
                        and output.index not in completed_indices
                        and len(output.token_ids) > 0
                    ):
                        token_count = len(output.token_ids)

                        aborted = False
                        for detector_cfg in anomaly_detectors:
                            states = detector_cfg["states"]
                            last_token_lens = detector_cfg["last_token_lens"]

                            detector = states.get(output.index)
                            if detector is None:
                                detector = IncrementalTokenRepetitionDetector(
                                    min_repeat_tokens=120,
                                    min_repeat_count=30,
                                    allow_gap=True,
                                    max_gap_ratio=1.0,
                                )
                                states[output.index] = detector
                                last_token_lens[output.index] = 0

                            prev_len = last_token_lens.get(output.index, 0)
                            if token_count < prev_len:
                                detector.reset()
                                prev_len = 0

                            if token_count > prev_len:
                                detector.append(output.token_ids[prev_len:token_count])
                                last_token_lens[output.index] = token_count

                            # Only run anomaly detection every N decode steps.
                            if token_count % anomaly_check_interval != 0:
                                continue

                            is_anomalous, reason = detector.detect()
                            if not is_anomalous:
                                continue

                            completed_indices.add(output.index)
                            for cleanup_cfg in anomaly_detectors:
                                cleanup_cfg["states"].pop(output.index, None)
                                cleanup_cfg["last_token_lens"].pop(output.index, None)
                            yield {
                                "index": output.index,
                                "completion": output.text,
                                "finish_reason": detector_cfg["finish_reason"],
                                "prompt_token_ids": prompt_token_ids or [],
                                "response_token_num": token_count,
                            }

                            await self.engine.abort(
                                get_sequence_request_id(output.index)
                            )
                            logger.warning(
                                f"Aborted sequence {output.index} of request {request_id} "
                                f"due to {detector_cfg['name']}: {reason}"
                            )
                            aborted = True
                            break

                        if aborted:
                            continue

        except (asyncio.CancelledError, Exception) as e:
            # Ensure request is aborted on any error or cancellation
            logger.warning(
                f"Aborting request {request_id} due to: {type(e).__name__}: {e}"
            )
            await self.engine.abort(request_id)
            raise
        finally:
            # Remove from active requests
            self.active_requests.discard(request_id)

    async def init_nccl(self):
        asyncio.create_task(
            self.engine.collective_rpc(
                "init_weight_update_group",
                args=(
                    self.config.network.vllm_host,
                    self.config.network.nccl_port,
                    self.context.rank + 1,
                    self.context.world_size + 1,
                ),
            )
        )

    async def update_weights_nccl(
        self, names: List[str], shapes: List[List[int]], dtypes: List[str]
    ):
        asyncio.create_task(
            self.engine.collective_rpc(
                "update_weight",
                args=(names, shapes, dtypes),
            )
        )

    async def get_sequence_counts(self) -> tuple[int, int]:
        logger_manager = self.engine.logger_manager
        stats = logger_manager.per_engine_logger_dict[0][0].last_scheduler_stats
        running_count, waiting_count = stats.num_running_reqs, stats.num_waiting_reqs

        logger.debug(
            f"Replica {self.context.replica_id.unique_id[:8]} with rank {self.context.rank}: "
            f"running={running_count}, waiting={waiting_count}"
        )
        return running_count, waiting_count

    async def shutdown(self):
        logger.info(f"Shutting down VLLMServiceDeployment {self.context.rank}")

        # Cancel stats task
        if hasattr(self, "stats_task") and self.stats_task:
            self.stats_task.cancel()
            try:
                await self.stats_task
            except asyncio.CancelledError:
                pass

        # Abort all active requests
        if hasattr(self, "active_requests"):
            for request_id in list(self.active_requests):
                try:
                    await self.engine.abort(request_id)
                    logger.info(f"Aborted active request: {request_id}")
                except Exception as e:
                    logger.warning(f"Failed to abort request {request_id}: {e}")
            self.active_requests.clear()

        # Shutdown engine
        if hasattr(self, "engine") and self.engine:
            try:
                # Wait a bit for ongoing operations to complete
                await asyncio.sleep(0.5)
                logger.info("Engine shutdown complete")
            except Exception as e:
                logger.error(f"Error during engine shutdown: {e}")

    async def health_check(self) -> bool:
        return self.engine is not None


def create_app(deployment_name: str, app_name: str):
    app = FastAPI(title="vLLM Ray Serve API")
    deployment_handle = serve.get_deployment_handle(
        deployment_name=deployment_name, app_name=app_name
    )
    deployment_handle._init()
    while not deployment_handle.running_replicas_populated():
        logger.info("Router is still None ...")
        time.sleep(1)

    all_replica_handles = []
    for (
        replica
    ) in deployment_handle._router._asyncio_router._request_router._replica_id_set:
        actor_name = replica.to_full_id_str()
        logger.info(f"Found replica actor: {actor_name}")
        actor_handle = ray.get_actor(actor_name, namespace="serve")
        all_replica_handles.append(actor_handle)

    def broadcast(method_name: str, *args, **kwargs):
        pickled_request_metadata = pickle.dumps(
            RequestMetadata(
                request_id=uuid.uuid4().hex,
                internal_request_id=uuid.uuid4().hex,
                call_method=method_name,
            )
        )
        tasks = [
            actor.handle_request.remote(pickled_request_metadata, *args, **kwargs)
            for actor in all_replica_handles
        ]
        return ray.get(tasks)

    @app.get("/health/")
    async def health():
        results = broadcast("health_check")
        all_ready = all(results)
        return {"status": "ok" if all_ready else "not_ready"}

    @app.post("/generate/")
    async def generate(request: GenerateRequest):
        stream = deployment_handle.options(stream=True).generate.remote(request)

        async def stream_ndjson():
            async for item in stream:
                yield (json.dumps(item) + "\n").encode("utf-8")

        return StreamingResponse(stream_ndjson(), media_type="application/x-ndjson")

    @app.post("/init_nccl/")
    async def init_nccl():
        broadcast("init_nccl")

    @app.post("/update_weights_nccl/")
    async def update_weights_nccl(request: UpdateWeightsNCCLRequest):
        broadcast(
            "update_weights_nccl",
            names=request.names,
            shapes=request.shapes,
            dtypes=request.dtypes,
        )

    return app


def main(config_file: str):
    config = load_config(config_file)

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = config.vllm_server.visible_devices
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    logger.info(f"Set CUDA_VISIBLE_DEVICES to: {config.vllm_server.visible_devices}")
    logger.info("Set VLLM_WORKER_MULTIPROC_METHOD to: spawn")

    # Step 1: Initialize Ray and deploy backend vLLM service
    num_replicas = config.vllm_server.data_parallel_size
    ray.init(
        num_cpus=num_replicas * 5, num_gpus=num_replicas, logging_level=logging.WARNING
    )
    logger.info(f"Ray initialized with {num_replicas} GPUs")
    logger.info(f"Starting Ray Serve with {num_replicas} vLLM replicas")
    logger.info(f"NCCL world size: {num_replicas + 1} (including client rank 0)")

    vllm_deployment = VLLMServiceDeployment.options(
        name="vllm-service",
        num_replicas=num_replicas,
        ray_actor_options={"num_cpus": 4, "num_gpus": 1},
        max_ongoing_requests=config.vllm_server.llm_params.max_num_seqs
        * config.vllm_server.data_parallel_size,
        request_router_config=RequestRouterConfig(
            request_router_class="vllm_service.router.load_aware:VLLMLoadAwareRouter",
        ),
        logging_config=LoggingConfig(log_level=logging.WARNING),
    ).bind(config.to_dict())

    serve.run(
        vllm_deployment,
        name="vllm-service",
        route_prefix=None,
    )
    logger.info(f"vLLM backend service deployed with {num_replicas} replicas")

    try:
        # Step 2: Create FastAPI app and run server
        app = create_app(deployment_name="vllm-service", app_name="vllm-service")

        logger.info(f"Starting FastAPI server on port {config.network.vllm_port}")
        uvicorn_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=config.network.vllm_port,
            timeout_keep_alive=7200,
            access_log=False,
        )
        server_instance = uvicorn.Server(uvicorn_config)
        asyncio.run(server_instance.serve())
    except KeyboardInterrupt:
        logger.info("Shutting down due to KeyboardInterrupt...")
    except Exception:
        logger.exception("Error occurred during server execution")
    finally:
        logger.info("Cleaning up resources...")
        try:
            # Shutdown all replicas gracefully
            deployment_handle = serve.get_deployment_handle(
                deployment_name="vllm-service", app_name="vllm-service"
            )
            if deployment_handle:
                logger.info("Broadcasting shutdown to all replicas...")
                # This will be caught by serve.shutdown() but we log it
        except Exception as e:
            logger.warning(f"Error during replica shutdown broadcast: {e}")

        # Shutdown Ray Serve and Ray
        try:
            serve.shutdown()
            logger.info("Ray Serve shutdown complete")
        except Exception as e:
            logger.error(f"Error during Ray Serve shutdown: {e}")

        try:
            ray.shutdown()
            logger.info("Ray shutdown complete")
        except Exception as e:
            logger.error(f"Error during Ray shutdown: {e}")


if __name__ == "__main__":
    fire.Fire(main)
