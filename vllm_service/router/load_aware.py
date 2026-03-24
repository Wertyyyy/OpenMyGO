import asyncio
import logging
import sys
import time
from typing import Dict, List, Optional

import ray
from ray.serve._private.request_router.common import PendingRequest
from ray.serve._private.request_router.replica_wrapper import RunningReplica
from ray.serve._private.request_router.request_router import RequestRouter

logger = logging.getLogger(__name__)


class VLLMLoadAwareRouter(RequestRouter):
    """Load-aware router for Ray Serve replicas.

    Selection rule (same as requested):
    1) If a request pins a data_parallel_rank, use that rank directly.
    2) Otherwise pick engine with minimal `waiting * 4 + running`, scanning
       from a rotating start index to reduce tie bias.
    3) Immediately increase local waiting count for the chosen engine.

    Replica load is refreshed from replicas via `get_sequence_counts()` at most
    every 100ms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eng_start_index = 0
        self._client_count = 1

        self._lb_engines: List[List[int]] = []
        self._replica_ids: List[str] = []
        self._replica_by_id: Dict[str, RunningReplica] = {}

        self._last_stats_refresh = 0.0
        self._stats_refresh_interval_s = 0.1

        logger.info("VLLMLoadAwareRouter initialized")

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        if not candidate_replicas:
            logger.warning("No candidate replicas available")
            return []

        self._sync_replica_state(candidate_replicas)

        now = time.monotonic()
        if now - self._last_stats_refresh >= self._stats_refresh_interval_s:
            await self._refresh_lb_counts()
            self._last_stats_refresh = now

        requested_rank = self._extract_data_parallel_rank(pending_request)

        # Engines are in rank order.
        if requested_rank is None:
            current_counts = self._lb_engines
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from eng_start_index to help balancing when engines are empty.
                idx = (self._eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx

            # Increment local waiting count for better balancing between stats updates.
            current_counts[eng_index][0] += self._client_count
            self._eng_start_index = (eng_index + 1) % num_engines
        else:
            eng_index = requested_rank % len(self._replica_ids)

        replica_id = self._replica_ids[eng_index]
        chosen_engine = self._replica_by_id[replica_id]
        return [[chosen_engine]]

    def update_replicas(self, replicas: List[RunningReplica]):
        super().update_replicas(replicas)
        self._sync_replica_state(replicas)
        self._eng_start_index = 0

        logger.info(
            "Replicas updated (count=%s): reset load-balancer state",
            len(replicas),
        )

    def _sync_replica_state(self, replicas: List[RunningReplica]) -> None:
        sorted_replicas = sorted(replicas, key=self._replica_key)
        replica_ids = [self._replica_key(replica) for replica in sorted_replicas]

        if replica_ids == self._replica_ids:
            return

        old_counts = {rid: counts for rid, counts in zip(self._replica_ids, self._lb_engines)}

        self._replica_ids = replica_ids
        self._replica_by_id = {self._replica_key(replica): replica for replica in sorted_replicas}
        self._lb_engines = [list(old_counts.get(rid, [0, 0])) for rid in self._replica_ids]

        if self._eng_start_index >= len(self._replica_ids):
            self._eng_start_index = 0

    async def _refresh_lb_counts(self) -> None:
        if not self._replica_ids:
            return

        tasks = [self._fetch_counts(rid) for rid in self._replica_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is None:
                continue
            waiting, running = result
            self._lb_engines[i][0] = waiting
            self._lb_engines[i][1] = running

    async def _fetch_counts(self, replica_id: str) -> Optional[tuple[int, int]]:
        """Query `(waiting, running)` from one replica actor."""
        replica = self._replica_by_id.get(replica_id)
        if replica is None:
            return None

        actor_name = self._replica_key(replica)
        try:
            actor_handle = ray.get_actor(actor_name, namespace="serve")
            counts = await actor_handle.get_sequence_counts.remote()
            if not isinstance(counts, tuple) or len(counts) != 2:
                return None
            running, waiting = int(counts[0]), int(counts[1])
            return waiting, running
        except Exception as exc:
            logger.debug("Failed to refresh counts for replica %s: %s", actor_name, exc)
            return None

    @staticmethod
    def _extract_data_parallel_rank(
        pending_request: Optional[PendingRequest],
    ) -> Optional[int]:
        if pending_request is None:
            return None

        kwargs = getattr(pending_request, "kwargs", None) or {}
        rank = kwargs.get("data_parallel_rank")
        if isinstance(rank, int):
            return rank

        # Also support a nested request object carrying `data_parallel_rank`.
        for value in kwargs.values():
            nested_rank = getattr(value, "data_parallel_rank", None)
            if isinstance(nested_rank, int):
                return nested_rank

        return None

    @staticmethod
    def _replica_key(replica: RunningReplica) -> str:
        replica_id = replica.replica_id
        if hasattr(replica_id, "to_full_id_str"):
            return replica_id.to_full_id_str()
        if hasattr(replica_id, "unique_id"):
            return replica_id.unique_id
        return str(replica_id)
