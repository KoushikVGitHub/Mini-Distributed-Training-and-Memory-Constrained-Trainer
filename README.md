# Mini Distributed Training & Memory-Constrained Trainer (pure NumPy)

A complete deep-learning training stack built from scratch in **pure NumPy** — no PyTorch, no
autograd engine, no communication library. It reimplements, in ~40 small readable functions, the
core techniques modern frameworks (PyTorch DDP, DeepSpeed/ZeRO, Megatron) use to **scale models
across devices** and **fit them into a limited memory budget**.

The goal is to make the mechanics unambiguous: every gradient is derived by hand, every collective
(all-reduce, all-gather, reduce-scatter) is simulated over in-memory "workers," and every byte of
memory is accounted for explicitly.

## What's inside

The stack is a two-layer MLP (`Linear → ReLU → Linear`) trained on synthetic regression data, with
five systems layers built on top of a hand-written autograd core:

| Layer | Idea | Key functions |
|-------|------|---------------|
| **1. Manual autograd** | Forward + backward pass derived by hand, verified against the loss gradient | `mlp_forward`, `mlp_backward`, `linear_backward`, `relu_backward`, `mse_loss_and_grad` |
| **2. Gradient accumulation** | Split a batch into micro-batches and sum grads to emulate a large batch under a small memory cap | `split_into_micro_batches`, `accumulate_gradients`, `grad_accumulation_step` |
| **3. Activation checkpointing** | Trade compute for memory — cache only the block input and **recompute** activations during backward | `mlp_forward_checkpointed`, `recompute_block_activations`, `mlp_backward_checkpointed` |
| **4. Mixed precision** | fp16 compute with an fp32 master copy, static loss scaling, and NaN/Inf overflow skipping | `cast_to_half_precision`, `make_master_params`, `scale_loss`, `unscale_gradients`, `mixed_precision_step` |
| **5. Distributed training** | Data-parallel all-reduce (naive + **ring**) and **ZeRO**-style optimizer-state sharding | `all_reduce_mean`, `ring_all_reduce_mean`, `data_parallel_train_step`, `partition_optimizer_state`, `zero_optimizer_step` |

Everything comes together in `full_distributed_training_loop`, which runs end-to-end across N
simulated workers with gradient accumulation, checkpointing, mixed precision, and ZeRO-sharded Adam
all active at once — each optimization toggleable via a flag.

### Highlights worth a look

- **Hand-derived backprop** that provably matches the checkpointed path (`np.allclose` in the demo).
- **Ring all-reduce** (`ring_all_reduce_mean`) implemented as a proper reduce-scatter + all-gather
  over chunks — the bandwidth-optimal algorithm real GPU clusters use — not just an averaging loop.
- **ZeRO optimizer sharding**: Adam's `m`/`v` moments are partitioned across workers, each worker
  updates only its shard, then an all-gather reconstructs full parameters.
- **Explicit memory accounting** (`compare_memory_with_and_without_optimizations`) that reports the
  parameter / optimizer / activation byte breakdown and the savings ratio from applying all
  optimizations together.

## How to run

```bash
python scaffold.py
```

The scaffold runs a guided demo: builds data and the model, runs forward/backward, checks the
checkpointed path against full backprop, does a data-parallel SGD step and a ZeRO Adam step, prints
a memory-savings report, and finally trains end-to-end and prints the loss trajectory.

**Requirements:** Python 3.8+ and NumPy (`pip install numpy`).

## Repository layout

```
model.py      # All 40 building blocks, from make_synthetic_regression_batch to
              # full_distributed_training_loop, grouped by the five layers above.
scaffold.py   # Runnable end-to-end demo that exercises each layer and prints results.
```

## Why this exists

Frameworks hide all of this behind `.backward()`, `DistributedDataParallel`, and
`deepspeed.initialize()`. Rebuilding it in NumPy — deriving each gradient, simulating each
collective, and counting each byte — turns those abstractions into things you can actually reason
about: *where the memory goes, what a collective really costs, and why ZeRO and checkpointing work.*
