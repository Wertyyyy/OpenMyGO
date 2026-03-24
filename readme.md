## What is This Repository?

This is an early-stage version of the MyGO (**My** Own **G**RP**O**) project. The main branch is currently private and contains research-related code that is not yet ready for public release.

The project was initiated in April 2025. While mainstream frameworks like TRL and Verl have already integrated GRPO support, I chose to implement my own GRPO training framework for several key reasons:

- **Ascend NPU/vLLM Support**: Mainstream GRPO frameworks do not support Ascend NPU or Ascend vLLM, forcing us to rely on Huggingface Transformers for inference with poor efficiency. Internal adaptation within Huawei appears to be progressing slowly.
- **Multimodal Support**: Some frameworks (e.g., TRL) lacked multimodal capabilities at the time.
- **vLLM Version Gap**: vLLM was in transition from V0 to V1, with V1 offering significant performance improvements (especially on Ascend NPU). However, mainstream GRPO frameworks predominantly use V0. Critically, the separation of scheduler and worker in vLLM V1 renders V0-compatible parameter update mathods obsolete.
- **Dependency Complexity**: Mainstream frameworks typically have extensive dependencies that can cause numerous issues on non-standard hardware (Ascend NPU + Kunpeng CPU), making debugging difficult.
- **Algorithm Simplicity**: GRPO is fundamentally a simple algorithm, and implementing it from scratch presents an interesting engineering challenge.

In summary, I could not find an existing framework that simultaneously supports Ascend NPU, Ascend vLLM (V1), multimodality, and maintains both high efficiency and code elegance.

## Architecture Choices

Mainstream GRPO frameworks (and similar algorithms like REINFORCE and RLOO) generally adopt one of three architectural patterns:

### Uniform

Weights are loaded using Huggingface Transformers, then rollouts and training occur in-place using the same library.

![alt text](assets/Uniform.drawio.svg)

**Advantages:** Simple structure, excellent GPU utilization during training, no overhead from memory loading/unloading or weights synchronization.

**Disadvantages:** Sampling efficiency of Huggingface Transformers is typically unacceptable.

### Co-locate

Both Transformers and vLLM operates on all GPUs. vLLM memory is freed before policy updates and reloaded before sampling.

![alt text](assets/Co-locate.drawio.svg)

**Advantages:** vLLM provides highly efficient sampling. Maximizes GPU VRAM utilization for training across all devices.

**Disadvantages:** Complex component structure and workflows. To mitigate long sequences blocking sampling progress, we must sample large batches for off-policy training, requiring old logprob tracking and importance sampling. This increases algorithm complexity and computational overhead.

### Async

GPUs are partitioned into two independent groups: one dedicated to training (with Huggingface Transformers), another to inference (with vLLM).

![alt text](assets/Async.drawio.svg)

**Advantages:** Clear component responsibilities, easier implementation and debugging, while maintaining high efficiency. No importance sampling overhead required. 

**Disadvantages:** Cannot fully utilize all GPUs for training. Still faces the long-sequence delay problem. However, we can mitigate this with an x-steps-off-policy strategy that overlaps multiple generation steps. Additionally, this approach requires separate weights synchronization between vLLM and Huggingface Transformers.

## OpenMyGO Architecture

I chose to design my GRPO training framework based on the **Async** architecture because for me, both high performance and clear, easy-to-maintain and extensible code are essential. To achieve this, I can accept the trade-off of not being able to fully utilize all GPUs for training.

![alt text](assets/Arch.drawio.svg)

Below is a brief introduction to each core component:

1. **Rollout DP** (`vllm_service/`)

   Multiple vLLM instances are orchestrated by Ray to provide data-parallel inference. At the same time, all vLLM instances and FSDP2 rank 0 form a stateless NCCL group. During inference, each vLLM DP worker runs independently while Ray handles load balancing. At the end of each training step, parameters are broadcast from FSDP2 rank 0 to all vLLM workers through the stateless NCCL group.

2. **FSDP2 SPMD Workers** (`grpo_trainer.py`)

   This is the policy training group, composed of multiple FSDP2 (Fully Sharded Data Parallel v2) workers, each holding a shard of the model parameters. At the beginning of each training step, every rank requests its own batch from the data service. At the end of the step, rank 0 gathers the updated parameters and broadcasts them to all vLLM workers.

3. **Reference DP** (`tf_service/`)

   Deployed on separate hardware, this group computes reference log probabilities for generated trajectories. Because the reference model is fixed and does not require parameter updates, it can run on dedicated devices. This isolation prevents resource contention with policy training.

4. **Data Service** (`data_service/`)

   Acts as the central coordinator of the entire data pipeline. It receives rollout requests from FSDP2 workers, dispatches them to both vLLM DP and Reference DP workers, collects generated trajectories, computes rewards and advantages (via the reward processing pool), and returns training batches to FSDP2 workers.

   ![Overlap](assets/Overlap.drawio.svg)

   By adopting an x-steps-off-policy strategy (x=2 in the figure above), the data service overlaps multiple rollout-generation cycles with policy updates. This keeps GPU resources highly utilized even when rollout generation is slower than policy optimization.

Compared with other asynchronous RL frameworks, and following the taxonomy in [Async RL Training Landscape](https://huggingface.co/blog/async-rl-training-landscape), OpenMyGO can be characterized as follows:

| Dimension | Description | OpenMyGO |
| --- | --- | --- |
| Orchestration Type | How does the system coordinate distributed components? | Native Python concurrency + HTTP microservices |
| Rollout Buffer Design | How do rollouts flow from inference to training, and how deep is the pipeline? | Bounded async queue (0-K) |
| Weight Synchronization Protocol (Transport) | How are new model weights delivered to inference servers after gradient updates? | NCCL + bucketing |
| Weight Synchronization Protocol (Interrupt Granularity) | In interrupt-based updates, when does generation pause to accept new weights? | Soft pause (drain in-flight requests) |
| Partial Rollout Handling | What happens to in-progress generations when a weight update arrives? | Implicit continuation |
| Staleness Management | How does the system handle rollouts generated by older policies? | Depth bounding without importance sampling correction |
| Distributed Training Backend & Parallelism | What training parallelism strategy is used, and how does it shape the async design? | FSDP2 |