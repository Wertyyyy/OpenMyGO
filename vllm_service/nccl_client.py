import logging
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DTYPE_STR_TO_TORCH = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
}


class NCCLClient:
    def __init__(
        self,
        host: str,
        server_port: int,
        nccl_port: int,
        nccl_device: str,
        dp_size: int,
        max_retries: int = 3,
        pack_size_mb: int = 500,
    ):
        self.host = host
        self.server_port = server_port
        self.nccl_port = nccl_port
        self.nccl_device = nccl_device
        self.dp_size = dp_size
        self.max_retries = max_retries
        self.pack_size_bytes = pack_size_mb * 1024 * 1024
        self.session = requests.Session()
        self.pynccl = None

        # Setup retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def init_nccl(self):
        url = f"http://{self.host}:{self.server_port}/init_nccl/"
        start_time = time.time()

        # Step 1: Send request to server to trigger server-side initialization
        logger.debug("Step 1: Sending NCCL initialization request to server...")
        response = self.session.post(
            url, timeout=600, proxies={"http": None, "https": None}
        )
        logger.debug(f"Received server response with status: {response.status_code}")

        # Step 2: Start client NCCL initialization
        logger.debug("Step 2: Starting client NCCL initialization")
        logger.debug("Starting NCCL initialization")
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.nccl_port,
            rank=0,
            world_size=self.dp_size + 1,
        )
        self.pynccl = PyNcclCommunicator(pg, device=self.nccl_device)

        time_end = time.time()
        logger.debug(
            f"NCCL initialization completed, time taken: {time_end - start_time} seconds"
        )

    def update_weights_nccl(self, state_dict):
        if self.pynccl is None:
            raise RuntimeError("NCCL not initialized. Call init_nccl() first.")

        time_start = time.time()
        url = f"http://{self.host}:{self.server_port}/update_weights_nccl/"

        # Step 1: Pack small tensors into large buffers (default 500MB) grouped by dtype.
        logger.debug("Step 1: Preparing packed weight metadata...")

        def flush_current_chunk(chunk, packed_chunks):
            if chunk is None or not chunk["items"]:
                return

            packed_chunks.append(
                {
                    "dtype": chunk["dtype"],
                    "total_numel": chunk["total_numel"],
                    "items": chunk["items"],
                }
            )

        packed_chunks = []
        current_chunk = None

        for name, param in state_dict.items():
            dtype_str = str(param.dtype)
            if dtype_str not in DTYPE_STR_TO_TORCH:
                raise ValueError(f"Unsupported dtype for NCCL packed transfer: {dtype_str}")

            tensor_numel = int(param.numel())
            tensor_nbytes = tensor_numel * param.element_size()

            need_new_chunk = (
                current_chunk is None
                or current_chunk["dtype"] != dtype_str
                or (
                    current_chunk["current_nbytes"] + tensor_nbytes > self.pack_size_bytes
                    and current_chunk["items"]
                )
            )
            if need_new_chunk:
                flush_current_chunk(current_chunk, packed_chunks)
                current_chunk = {
                    "dtype": dtype_str,
                    "current_nbytes": 0,
                    "total_numel": 0,
                    "items": [],
                }

            current_chunk["items"].append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "numel": tensor_numel,
                    "tensor": param,
                }
            )
            current_chunk["current_nbytes"] += tensor_nbytes
            current_chunk["total_numel"] += tensor_numel

        flush_current_chunk(current_chunk, packed_chunks)

        names = []
        shapes = []
        dtypes = []
        for chunk_idx, chunk in enumerate(packed_chunks):
            names.append(
                json.dumps(
                    {
                        "chunk_id": chunk_idx,
                        "tensors": [
                            {
                                "name": item["name"],
                                "shape": item["shape"],
                                "numel": item["numel"],
                            }
                            for item in chunk["items"]
                        ],
                    },
                    separators=(",", ":"),
                )
            )
            shapes.append([chunk["total_numel"]])
            dtypes.append(chunk["dtype"])

        logger.debug(
            f"Prepared {len(state_dict)} tensors into {len(packed_chunks)} NCCL chunks "
            f"(target chunk size: {self.pack_size_bytes / (1024 * 1024):.0f}MB)"
        )

        # Step 2: Send HTTP request to notify server first (server will prepare workers to receive)
        logger.debug("Step 2: Sending HTTP request to notify server...")
        self.session.post(
            url,
            json={
                "names": names,
                "shapes": shapes,
                "dtypes": dtypes,
            },
            timeout=600,
            proxies={"http": None, "https": None},
        )

        # Step 3: Start client NCCL weight transfer (workers are now ready to receive)
        logger.debug("Step 3: Starting client NCCL weight transfer")

        def send_weights_blocking(chunks):
            logger.debug("Starting NCCL packed weight transfer")
            start_time = time.time()
            device = torch.device(self.nccl_device)

            self.pynccl.group.barrier()

            for chunk in chunks:
                dtype = DTYPE_STR_TO_TORCH[chunk["dtype"]]
                packed_buffer = torch.empty(
                    (chunk["total_numel"],),
                    dtype=dtype,
                    device=device,
                )

                offset = 0
                for item in chunk["items"]:
                    tensor = item["tensor"].detach()
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()

                    flat_tensor = tensor.to(device=device, dtype=dtype, non_blocking=True).view(-1)
                    size = flat_tensor.numel()
                    packed_buffer[offset : offset + size].copy_(flat_tensor, non_blocking=True)
                    offset += size

                self.pynccl.broadcast(
                    packed_buffer,
                    src=0,
                    stream=torch.cuda.current_stream(),
                )
                # self.pynccl.group.barrier()

            end_time = time.time()
            logger.debug(
                f"NCCL packed weight transfer completed in {end_time - start_time} seconds"
            )

        # Send weights via NCCL
        time_nccl_start = time.time()
        send_weights_blocking(packed_chunks)
        time_nccl_end = time.time()

        time_end = time.time()
        logger.debug(
            f"Time taken for NCCL transfer: {time_nccl_end - time_nccl_start} seconds"
        )
        logger.debug(f"Total time taken: {time_end - time_start} seconds")

    def close(self):
        if self.session:
            self.session.close()
            logger.info("NCCL client session closed")
