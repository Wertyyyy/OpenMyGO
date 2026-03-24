import logging
from typing import List, Optional
from ray.serve._private.request_router.request_router import RequestRouter
from ray.serve._private.request_router.common import PendingRequest
from ray.serve._private.request_router.replica_wrapper import RunningReplica

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VLLMRoundRobinRouter(RequestRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._next_replica_index = 0

        logger.info("VLLMRoundRobinRouter initialized")

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        if not candidate_replicas:
            logger.warning("No candidate replicas available")
            return []

        num_replicas = len(candidate_replicas)
        selected_index = self._next_replica_index % num_replicas
        selected_replica = candidate_replicas[selected_index]

        self._next_replica_index = (self._next_replica_index + 1) % num_replicas

        replica_id = selected_replica.replica_id.unique_id
        logger.debug(
            f"Selected replica {replica_id[:8]} "
            f"(index {selected_index}/{num_replicas - 1})"
        )

        return [[selected_replica]]

    def update_replicas(self, replicas: List[RunningReplica]):
        super().update_replicas(replicas)

        self._next_replica_index = 0

        logger.info(
            f"Replicas updated (count={len(replicas)}): reset round robin counter to 0"
        )
