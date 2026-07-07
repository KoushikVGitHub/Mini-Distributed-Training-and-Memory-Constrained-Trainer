"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Mini distributed training scaffold: end-to-end demo of MLP training with
gradient accumulation, checkpointing, mixed precision, data parallel, and ZeRO."""
import numpy as np

from solution import (
    make_synthetic_regression_batch,
    init_mlp_params,
    linear_forward,
    relu_forward,
    mlp_forward,
    mse_loss_and_grad,
    linear_backward,
    relu_backward,
    first_linear_backward,
    mlp_backward,
    split_into_micro_batches,
    accumulate_gradients,
    scale_accumulated_gradients,
    grad_accumulation_step,
    mlp_forward_checkpointed,
    recompute_block_activations,
    mlp_backward_checkpointed,
    estimate_checkpointing_memory_savings,
    cast_to_half_precision,
    make_master_params,
    scale_loss,
    unscale_gradients,
    has_non_finite_gradients,
    mixed_precision_step,
    shard_dataset_across_workers,
    compute_local_gradients,
    all_reduce_mean,
    ring_all_reduce_mean,
    data_parallel_train_step,
    bucket_gradients,
    init_adam_state,
    partition_optimizer_state,
    local_shard_adam_update,
    all_gather_param_shards,
    zero_optimizer_step,
    compute_param_memory_bytes,
    compute_optimizer_memory_bytes,
    compute_peak_activation_memory_bytes,
    compare_memory_with_and_without_optimizations,
    full_distributed_training_loop,
)


if __name__ == "__main__":
    np.random.seed(0)

    # --- 1. Data + model ---
    batch_size, in_dim, hidden_dim, out_dim = 32, 8, 16, 4
    x, y = make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed=0)
    params = init_mlp_params(in_dim, hidden_dim, out_dim, seed=0)
    print(f"Data: x{ x.shape}, y{y.shape}")
    print(f"Params: " + ", ".join(f"{k}{v.shape}" for k, v in params.items()))

    # --- 2. Forward + loss ---
    y_pred, cache = mlp_forward(x, params)
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    print(f"Initial MSE loss: {loss:.6f}")

    # --- 3. Backward ---
    grads = mlp_backward(dy_pred, cache, params)
    print(f"Grad norms: " + ", ".join(f"{k}={np.linalg.norm(v):.4f}" for k, v in grads.items()))

    # --- 4. Gradient accumulation step ---
    accum_grads = grad_accumulation_step(x, y, params, micro_batch_size=8)
    print(f"Accumulated grad norm (W1): {np.linalg.norm(accum_grads['W1']):.4f}")

    # --- 5. Checkpointed forward/backward ---
    y_pred_ckpt, light_cache = mlp_forward_checkpointed(x, params)
    grads_ckpt = mlp_backward_checkpointed(dy_pred, light_cache, params)
    print(f"Checkpoint matches full backward: "
          f"{np.allclose(grads_ckpt['W1'], grads['W1'], atol=1e-6)}")
    saved = estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, 4)
    print(f"Checkpointing saves ~{saved} bytes of activations")

    # --- 6. Data-parallel step ---
    new_params = data_parallel_train_step(x, y, params, num_workers=4, lr=1e-2)
    y_pred_after, _ = mlp_forward(x, new_params)
    loss_after, _ = mse_loss_and_grad(y_pred_after, y)
    print(f"Loss after one data-parallel SGD step: {loss_after:.6f}")

    # --- 7. ZeRO sharded Adam step ---
    adam_state = init_adam_state(params)
    worker_states = partition_optimizer_state(adam_state, num_workers=2)
    zero_params, worker_states = zero_optimizer_step(params, grads, worker_states, lr=1e-3)
    print(f"ZeRO updated W2 norm: {np.linalg.norm(zero_params['W2']):.4f}")

    # --- 8. Memory accounting ---
    mem_report = compare_memory_with_and_without_optimizations(x, params, num_workers=2)
    print("Memory comparison (bytes):")
    for k, v in mem_report.items():
        print(f"  {k}: {v}")

    # --- 9. End-to-end distributed training loop ---
    x_big, y_big = make_synthetic_regression_batch(64, in_dim, out_dim, seed=1)
    result = full_distributed_training_loop(
        x_big, y_big,
        num_workers=2, num_steps=10, micro_batch_size=8,
        lr=1e-3, hidden_dim=hidden_dim,
        use_checkpointing=True, use_mixed_precision=True, use_zero=True,
        seed=0,
    )
    losses = result['loss_history']
    print(f"Loss history ({len(losses)} steps): "
          f"start={losses[0]:.4f}, end={losses[-1]:.4f}")
