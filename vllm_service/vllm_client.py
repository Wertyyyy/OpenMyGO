import logging
import time
import asyncio
import json
import aiohttp
from typing import Dict, Any, AsyncGenerator
from requests import ConnectionError

from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VLLMClient:
    def __init__(
        self,
        host: str,
        server_port: int,
        max_retries: int = 3,
    ):
        self.host = host
        self.server_port = server_port
        self.max_retries = max_retries
        self.session = None

        self.loop = asyncio.new_event_loop()

    async def initialize(self):
        await self._check_server()
        await self._create_session()

    async def _create_session(self):
        """Create aiohttp session with improved connection pool configuration"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=50,
                keepalive_timeout=300,
                enable_cleanup_closed=True,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )

            timeout = aiohttp.ClientTimeout(
                total=60000,
                connect=30000,
                sock_read=30000,
            )

            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

            logger.info(
                f"Created new session for VLLM client {self.host}:{self.server_port}"
            )

    async def _check_server(
        self, total_timeout: float = 180.0, retry_interval: float = 2.0
    ):
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response_data = await response.json()
                        if response.status == 200 and response_data["status"] == "ok":
                            logger.info("VLLM Server is up!")
                            return None
            except aiohttp.ClientError as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} seconds. "
                    ) from exc

            logger.info(
                f"VLLM Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            await asyncio.sleep(retry_interval)

    async def generate(
        self,
        conversation: Conversation,
        sampling_params: Dict[str, Any],
    ) -> AsyncGenerator[dict, None]:
        """
        Stream completions from server. Each yielded item corresponds to one completed sequence.
        
        Args:
            conversation: The conversation history
            sampling_params: Dictionary of sampling parameters (temperature, top_p, top_k, n, max_tokens, etc.)
        """
        url = f"http://{self.host}:{self.server_port}/generate/"

        await self._create_session()

        try:
            logger.debug(f"Making streaming generate request to {url}")

            request_data = {
                "conversation": conversation.model_dump(),
                "sampling_params": sampling_params,
            }

            async with self.session.post(
                url,
                json=request_data,
                headers={"Accept": "application/x-ndjson"},
            ) as response:
                if response.status == 200:
                    logger.debug("Streaming generate response started")
                    while True:
                        line = await response.content.readline()
                        if not line:
                            break

                        text = line.decode("utf-8").strip()
                        if not text:
                            continue

                        yield json.loads(text)
                else:
                    text = await response.text()
                    error_msg = (
                        f"VLLM generate request failed: {response.status}, {text}"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg)

        except aiohttp.ServerDisconnectedError as e:
            logger.warning(f"VLLM server disconnected during stream: {e}")
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
            raise
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            logger.warning(f"VLLM client streaming connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in VLLM generate: {e}")
            raise

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("VLLM client session closed")
