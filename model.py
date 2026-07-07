"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_synthetic_regression_batch
def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    # Goal: seed numpy, sample x, build a hidden teacher, and produce noisy targets y.
    
    # 1. Set the global NumPy seed
    np.random.seed(seed)

    # 2. Sample random inputs (x)
    x = np.random.standard_normal((batch_size,in_dim))

    # 3. Build the "hidden teacher" (true weights)
    true_weights = np.random.standard_normal((in_dim,out_dim))

    # 4. Calculate exact targets (x @ true_weights)
    y_exact = x @ true_weights

    # 5. Add small noise so it's not a perfectly pristine line
    # Multiplying by 0.1 keeps the noise small, ensuring residuals < 0.3
    noise = np.random.standard_normal((batch_size,out_dim))*0.1

    y = y_exact + noise

    return x, y

# Step 2 - init_mlp_params
def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    # Goal: return a dict {'W1','b1','W2','b2'} with He-initialized weights and zero biases.
    np.random.seed(seed)

    W1 = np.random.standard_normal((in_dim, hidden_dim)) * (2.0 /in_dim)**(1/2)
    W2 = np.random.standard_normal((hidden_dim, out_dim)) * (2.0 /hidden_dim)**(1/2)

    b1 = np.zeros((hidden_dim,))
    b2 = np.zeros((out_dim,))

    return {'W1': W1,'b1': b1,'W2': W2,'b2': b2}

# Step 3 - linear_forward
def linear_forward(x, w, b):
    # Goal: apply y = x @ w + b and return the resulting (N, out_dim) array
    return x @ w + b

# Step 4 - relu_forward
def relu_forward(x):
    # Goal: apply the ReLU activation elementwise and return an array of the same shape.
    return np.maximum(x,0)

# Step 5 - mlp_forward
def mlp_forward(x, params):
    # Goal: run the two-layer MLP forward and return (y_pred, cache) with keys 'x','z1','a1','z2'.
    z1 = linear_forward(x,params['W1'], params['b1'])

    a1 = relu_forward(z1)

    z2 = linear_forward(a1, params['W2'], params['b2'])

    cache = {'x': x, 'a1': a1, 'z1': z1, 'z2': z2}

    return z2, cache

# Step 6 - mse_loss_and_grad
def mse_loss_and_grad(y_pred, y_true):
    # Goal: compute mean squared error loss and its gradient with respect to y_pred
    N = y_pred.size
    
    L = np.mean((y_pred - y_true)**2)

    grad = 2/N * (y_pred - y_true)

    return L, grad

# Step 7 - linear_backward
import numpy as np

def linear_backward(d_out, x, w):
    # Goal: backprop through y = x @ w + b and return (dx, dw, db)
    dx = d_out @ w.T

    dw = x.T @ d_out

    db = np.sum(d_out, axis=0)

    return dx, dw, db

# Step 8 - relu_backward
def relu_backward(d_out, z):
    # Goal: backprop through ReLU using the pre-activation z, return dz with same shape.
    # (z > 0) creates a boolean mask matrix of Trues (1) and Falses (0).
    # Multiplying d_out by this mask acts as a perfect gate.
    
    dz = d_out * (z > 0)
    return dz

# Step 9 - first_linear_backward
def first_linear_backward(d_z1, x, w1):
    # Goal: return gradients (dx, dW1, db1) for z1 = x @ w1 + b1 given d_z1.
    return linear_backward(d_z1,x,w1)

# Step 10 - mlp_backward
def mlp_backward(dy_pred, cache, params):
    # Goal: run the full MLP backward pass returning grads dict with keys W1,b1,W2,b2
    da1, dW2, db2 = linear_backward(dy_pred, cache['a1'], params['W2'])

    dz1 = relu_backward(da1, cache['z1'])

    dx, dW1, db1 = first_linear_backward(dz1, cache['x'], params['W1'])

    grad_params = {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}

    return grad_params

# Step 11 - split_into_micro_batches
def split_into_micro_batches(x, y, micro_batch_size):
    # Goal: split (x, y) into contiguous micro batches of at most micro_batch_size rows.
    N = x.shape[0]
    batches = []

    # Iterate through the array in steps of micro_batch_size
    for i in range(0, N, micro_batch_size):
        x_mb = x[i : i + micro_batch_size]
        y_mb = y[i : i + micro_batch_size]

        batches.append((x_mb,y_mb))

    return batches

# Step 12 - accumulate_gradients
def accumulate_gradients(accum_grads, new_grads):
    # Goal: return a dict whose values are elementwise sums of accum_grads and new_grads.
    if accum_grads is None:
        return new_grads

    # Combine the lists element-wise for matching keys
    result = {
        key: accum_grads[key] + new_grads[key]
        for key in accum_grads.keys() if key in new_grads
    }
    return result

# Step 13 - scale_accumulated_gradients
def scale_accumulated_gradients(accum_grads, num_micro_batches):
    # Goal: divide each gradient tensor by num_micro_batches and return a new dict
    divided_result = {
        key : value/num_micro_batches 
        for key, value in accum_grads.items()
    }

    return divided_result

# Step 14 - grad_accumulation_step
def grad_accumulation_step(x, y, params, micro_batch_size):
    # Goal: run forward/backward on each micro batch and combine grads to match a full-batch step.
    batches = split_into_micro_batches(x, y, micro_batch_size)

    # Initialize the accumulator
    accum_grads = None

    for x_mb, y_mb in batches:
        # 1. Forward pass
        y_pred, cache = mlp_forward(x_mb, params)

        # 2. Compute loss and the derivative of the prediction (dy_pred)
        loss, dy_pred = mse_loss_and_grad(y_pred, y_mb)

        # 3. Backward pass using dy_pred
        new_grads = mlp_backward(dy_pred, cache, params)

        # 4. Update the accumulated gradients state
        accum_grads = accumulate_gradients(accum_grads, new_grads)
    
    final_grads = scale_accumulated_gradients(accum_grads, len(batches))

    return final_grads

# Step 15 - mlp_forward_checkpointed
def mlp_forward_checkpointed(x, params):
    # Goal: forward pass that caches only the block input x, not intermediates.
    
    z1 = linear_forward(x, params['W1'], params['b1'])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1,params['W2'], params['b2'])

    cache = {'x': x}

    return z2, cache

# Step 16 - recompute_block_activations
def recompute_block_activations(x, params):
    # Goal: recompute z1, a1, z2 from x and params and return them in a cache dict
    z1 = linear_forward(x,params['W1'], params['b1'])

    a1 = relu_forward(z1)

    z2 = linear_forward(a1, params['W2'], params['b2'])

    cache = {'x': x, 'a1': a1, 'z1': z1, 'z2': z2}

    return cache

# Step 17 - mlp_backward_checkpointed
def mlp_backward_checkpointed(dy_pred, light_cache, params):
    # Goal: recompute activations from light_cache['x'] and run the standard MLP backward
    cache = recompute_block_activations(light_cache['x'], params)

    da1, dW2, db2 = linear_backward(dy_pred, cache['a1'], params['W2'])

    dz1 = relu_backward(da1, cache['z1'])

    dx, dW1, db1 = first_linear_backward(dz1, cache['x'], params['W1'])

    grad_params = {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}

    return grad_params

# Step 18 - estimate_checkpointing_memory_savings
def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    # Goal: estimate activation memory in bytes for full vs checkpointed forward on the two-layer MLP.
    
    # 1. Calculate the byte size of each cached tensor
    # Shape of x: (batch_size, in_dim)
    mem_x = batch_size * in_dim * dtype_bytes

    # Shape of z1 (pre-activation) and a1 (post-activation): (batch_size, hidden_dim)
    mem_z1 = batch_size * hidden_dim * dtype_bytes
    mem_a1 = batch_size * hidden_dim * dtype_bytes

    # 2. Standard Forward
    # Retains the input, the pre-activation (for ReLU grad), and post-activation (for layer 2 grad)
    full_bytes = mem_x + mem_z1 + mem_a1

    # 3. Checkpointed Forward
    # Retains only the input to the block. z1 and a1 are cleared and recomputed on demand.
    checkpoint_bytes = mem_x
    
    # 4. Calculate savings
    savings = full_bytes - checkpoint_bytes

    return {
        'full_bytes': full_bytes,
        'checkpoint_bytes': checkpoint_bytes,
        'saved_bytes': savings
        }

# Step 19 - cast_to_half_precision
def cast_to_half_precision(values):
    # Goal: Return a new dict mapping each key to its array converted to float16.
    
    half = {
        val : values[val].astype(np.float16)
        for val in values
        }
    
    return half

# Step 20 - make_master_params
def make_master_params(params):
    # Goal: return a dict mapping the same keys to independent float32 copies of each array.
    master_params = {
        val : params[val].astype(np.float32)
        for val in params
    }

    return master_params

# Step 21 - scale_loss
def scale_loss(loss, dy_pred, scale):
    # Goal: Scale the scalar loss and the upstream gradient dy_pred by the fixed loss scale.
    sl = loss * scale
    sdy = dy_pred * scale

    return sl, sdy

# Step 22 - unscale_gradients
def unscale_gradients(grads, scale):
    # Goal: divide every gradient tensor by scale and return a new float32 dict
    
    out = {
        grad : grads[grad].astype(np.float32)/scale
        for grad in grads
        }
    
    return out

# Step 23 - has_non_finite_gradients
def has_non_finite_gradients(grads):
    # Goal: return True if any array in grads contains NaN or Inf, else False
    
    for grad_name, grad_array in grads.items():
        if not np.isfinite(grad_array).all():
            return True
    
    return False

# Step 24 - mixed_precision_step
def mixed_precision_step(x, y, master_params, scale, lr):
    # Goal: run fp16 forward/backward, unscale grads, skip on overflow, else SGD-update fp32 master.
    # 1. Cast inputs and a view of the master parameters to fp16
    half_params = cast_to_half_precision(master_params)
    x_16 = x.astype(np.float16)
    y_16 = y.astype(np.float16)

    # 2. Forward pass & unscaled loss (in fp16)
    y_pred, cache = mlp_forward(x_16, half_params)
    loss_16, dy_pred = mse_loss_and_grad(y_pred, y_16)

    # Save the unscaled loss as an fp32 float to return at the end
    unscaled_loss_32 = float(loss_16)

    # 3. Scale loss and run backward pass (in fp16)
    sl, sdy = scale_loss(loss_16, dy_pred, scale)
    scaled_grads = mlp_backward(sdy, cache, half_params)

    # 4. Unscale the gradients back to fp32
    unscaled_grads = unscale_gradients(scaled_grads, scale)

    # 5. Check for overflow and conditionally apply SGD
    # We check the unscaled gradients for Inf/NaN
    skipped =  has_non_finite_gradients(unscaled_grads)

    master_params_copy = make_master_params(master_params)

    new_master_params = {}

    for key in master_params:
        fp32_weight = master_params_copy[key].astype(np.float32)
        if not skipped:
            # If gradients are clean, apply SGD to the fp32 master parameters.
            updated_weight = fp32_weight - lr * unscaled_grads[key]
            new_master_params[key] = updated_weight.astype(np.float32)
        else:
            # Skip update, just pass the float32 copy forward
            new_master_params[key] = fp32_weight

    return unscaled_loss_32, new_master_params, skipped

# Step 25 - shard_dataset_across_workers
def shard_dataset_across_workers(x, y, num_workers):
    # Goal: split x and y into num_workers contiguous shards along axis 0

    # 1. Split x and y independently
    x_shards = np.array_split(x, num_workers, axis = 0)
    y_shards = np.array_split(y, num_workers, axis = 0)

    # 2. Pair them up so each worker gets an (x_shard, y_shard) tuple
    worker_shards = list(zip(x_shards, y_shards))
    return worker_shards

# Step 26 - compute_local_gradients
def compute_local_gradients(x, y, params):
    """Compute parameter gradients for one worker's data shard.

    Forward (mlp_forward) -> loss gradient (mse_loss_and_grad) -> backward
    (mlp_backward). Return a grads dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    # Goal: forward, then mse loss gradient, then backward; return grads
    
    # 1. Forward pass
    y_pred, cache = mlp_forward(x, params)
    # 2. Compute loss and the derivative of the prediction (dy_pred)
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    # 3. Backward pass
    grads = mlp_backward(dy_pred, cache, params)

    return grads

# Step 27 - all_reduce_mean
def all_reduce_mean(per_worker_grads):
    # Goal: average a list of gradient dicts elementwise across workers
    
    mean = {
        key: sum(gradient[key] for gradient in per_worker_grads)/len(per_worker_grads)
        for key in per_worker_grads[0]
        }
    
    return mean

# Step 28 - ring_all_reduce_mean
def ring_all_reduce_mean(per_worker_arrays):
    # Goal: average arrays across workers via ring reduce-scatter then all-gather over chunks.
    
    num_workers = len(per_worker_arrays)
    
    # 1. Chunking
    # Split each worker's array into `num_workers` contiguous chunks.
    # state[w][c] gives worker w's chunk c.
    state = [ 
    [chunk.copy() for chunk in np.array_split(arr, num_workers)]
    for arr in per_worker_arrays
    ]

    # 2. Phase 1: Reduce-Scatter
    for s in range(num_workers - 1):
        # Snapshot the current step to avoid in-place update corruption
        current_state = [[chunk.copy() for chunk in worker] for worker in state]

        for w in range(num_workers):
            send_chunk_idx = (w - s) % num_workers
            receiving_worker_idx = (w + 1) % num_workers

            # Worker (w) sends its chunk to Worker (w+1), which adds it to its own
            state[receiving_worker_idx][send_chunk_idx] = (
                current_state[receiving_worker_idx][send_chunk_idx] +
                current_state[w][send_chunk_idx]
            )
    
    #3. Phase 2: All-Gather
    for s in range(num_workers - 1):
        # Snapshot again
        current_state = [[chunk.copy() for chunk in worker] for worker in state]

        for w in range(num_workers):
            # The chunk being passed along is the one we fully reduced (or just received)
            send_chunk_idx = (w + 1 - s) % num_workers
            receiving_worker_idx = (w + 1) % num_workers

            # Worker (w) sends its fully reduced chunk to Worker (w+1), which overwrites its own
            state[receiving_worker_idx][send_chunk_idx] = current_state[w][send_chunk_idx].copy()

    # 4. Reconstruct and Calculate Mean
    output = []
    for w in range(num_workers):
        full_sum_array = np.concatenate(state[w])
        mean_array = full_sum_array / num_workers
        output.append(mean_array)

    final_array = np.array(output)

    return final_array[0]

# Step 29 - data_parallel_train_step
def data_parallel_train_step(x, y, params, num_workers, lr):
    # Goal: shard the batch, compute local gradients, all-reduce mean them, then SGD update params.
    shards = shard_dataset_across_workers(x, y, num_workers)

    all_grads = []
    for x_s, y_s in shards:
        grads = compute_local_gradients(x_s, y_s, params)
        all_grads.append(grads)
    
    mean = all_reduce_mean(all_grads)

    new_params = {}

    for key in mean:
        updated_weight = params[key] - lr * mean[key]
        new_params[key] = updated_weight
    
    return new_params

# Step 30 - bucket_gradients
def bucket_gradients(grads, bucket_size):
    # Goal: pack flattened gradients into fixed-size buckets and return (buckets, meta).
    buckets = []
    meta = []
    
    current_bucket = []
    current_size = 0
    bucket_index = 0

    # 1. Walk the parameters in a stable order
    # Sorting the keys is CRITICAL. Every worker must process the tensors in the 
    # exact same order, or their ring-reduce operations will deadlock or corrupt data.
    for name in sorted(grads.keys()):
        grad = grads[name]
        shape = grad.shape
        n = grad.size
        
        # 2. Decide whether it fits in the current bucket
        # We only seal if the bucket isn't empty AND adding this exceeds the limit.
        if current_size > 0 and (current_size + n > bucket_size):
            # Seal the current bucket
            buckets.append(np.concatenate(current_bucket))
            
            # Reset for the new bucket
            current_bucket = []
            current_size = 0
            bucket_index += 1
            
        # 3. Record where this tensor lives
        start = current_size
        end = current_size + n
        meta.append((name, shape, start, end, bucket_index))
        
        # 4. Append to the current bucket
        current_bucket.append(grad.flatten())
        current_size += n
        
    # 5. Seal the final bucket (if there are any leftovers)
    if current_bucket:
        buckets.append(np.concatenate(current_bucket))
        
    return buckets, meta

# Step 31 - init_adam_state
def init_adam_state(params):
    # Goal: build Adam state with zero first/second moments per param and step counter t=0.
    state = {}

    # Initialize the step counter
    state['t'] = 0

    # Initialize first (m) and second (v) moments
    state['m'] = {}
    state['v'] = {}

    for p_name, p_value in params.items():
        # np.zeros_like creates an array of zeros with the same shape and type
        state['m'][p_name] = np.zeros_like(p_value)
        state['v'][p_name] = np.zeros_like(p_value)
    
    return state

# Step 32 - partition_optimizer_state
def partition_optimizer_state(state, num_workers):
    # Goal: split each Adam moment tensor into num_workers contiguous flat shards.
    workers = []
    for _ in range(num_workers):
        workers.append({
            't': state.get('t', 0),
            'm': {},
            'v': {},
            'shard_slices': {},
            'shapes': {} # Stored for later tensor reconstruction
        })
        
    for p_name in state['m']:
        m_tensor = state['m'][p_name]
        v_tensor = state['v'][p_name]
        
        orig_shape = m_tensor.shape
        
        # Flatten the arrays
        m_flat = m_tensor.reshape(-1)
        v_flat = v_tensor.reshape(-1)
        
        N = m_flat.size
        base = N // num_workers
        rem = N % num_workers
        
        start = 0
        for w in range(num_workers):
            chunk_size = base + 1 if w < rem else base
            end = start + chunk_size
            
            # Slice and explicitly .copy() to prevent memory aliasing
            workers[w]['m'][p_name] = m_flat[start:end].copy()
            workers[w]['v'][p_name] = v_flat[start:end].copy()
            
            # Store metadata
            workers[w]['shard_slices'][p_name] = (start, end)
            workers[w]['shapes'][p_name] = orig_shape
            
            start = end
            
    return workers

# Step 33 - local_shard_adam_update
def local_shard_adam_update(params, grads, worker_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # Goal: Apply an Adam update to only the local shard of each parameter using its owned moment shards.
# 1. Advance the step counter
    worker_state['t'] += 1
    t = worker_state['t']
    
    updated_shards = {}
    
    for p_name in worker_state['m']:
        start, end = worker_state['shard_slices'][p_name]
        
        # 2. Extract views of the gradient and parameter shards 
        g_shard = grads[p_name].reshape(-1)[start:end]
        p_shard = params[p_name].reshape(-1)[start:end]
        
        # 3. Retrieve old moments
        m_shard = worker_state['m'][p_name]
        v_shard = worker_state['v'][p_name]
        
        # 4. Update running averages
        m_shard = beta1 * m_shard + (1 - beta1) * g_shard
        v_shard = beta2 * v_shard + (1 - beta2) * (g_shard ** 2)
        
        # Save updated moments back to state
        worker_state['m'][p_name] = m_shard
        worker_state['v'][p_name] = v_shard
        
        # 5. Bias correction
        m_hat = m_shard / (1 - beta1 ** t)
        v_hat = v_shard / (1 - beta2 ** t)
        
        # 6. Calculate step size
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # 7. Create a BRAND NEW array for the updated shard. 
        # (p_shard - step) allocates new memory rather than mutating p_shard.
        updated_shards[p_name] = p_shard - step
        
    return updated_shards, worker_state

# Step 34 - all_gather_param_shards
def all_gather_param_shards(param_shards_per_worker, shapes, shard_slices_per_worker):
    # Goal: all-gather per-worker 1D parameter shards and restore original shapes.
    full_params = {}
    
    if not param_shards_per_worker:
        return full_params
        
    for p_name, orig_shape in shapes.items():
        # 1. Determine total elements and allocate a full flat buffer
        total_elements = np.prod(orig_shape)
        
        # Infer the data type from the first worker's shard
        dtype = param_shards_per_worker[0][p_name].dtype
        
        # Pre-allocate the flattened array
        flat_param = np.empty(total_elements, dtype=dtype)
        
        # 2. Each worker populates its slice of the flat buffer
        for w_idx in range(len(param_shards_per_worker)):
            shard = param_shards_per_worker[w_idx][p_name]
            start, end = shard_slices_per_worker[w_idx][p_name]
            
            flat_param[start:end] = shard
            
        # 3. Reshape the reassembled flat array back to its original dimensions
        full_params[p_name] = flat_param.reshape(orig_shape)
        
    return full_params

# Step 35 - zero_optimizer_step
def zero_optimizer_step(params, grads, worker_states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # Goal: run a full ZeRO step: each worker updates its shard, then all-gather rebuilds full params
    param_shards_per_worker = []
    shard_slices_per_worker = []
    
    for ws in worker_states:
        shards, updated_ws = local_shard_adam_update(
            params, grads, ws, lr, beta1, beta2, eps
        )
        
        param_shards_per_worker.append(shards)
        shard_slices_per_worker.append(updated_ws['shard_slices'])
        
    shapes = worker_states[0]['shapes']
    
    # all_gather_param_shards builds a brand new dictionary of tensors
    new_params = all_gather_param_shards(
        param_shards_per_worker, shapes, shard_slices_per_worker
    )
    
    # Return the new params instead of overwriting the old ones
    return new_params, worker_states

# Step 36 - compute_param_memory_bytes
def compute_param_memory_bytes(params):
    # Goal: sum the total bytes occupied by every parameter array in the dict.
    return sum(p.nbytes for p in params.values())

# Step 37 - compute_optimizer_memory_bytes
def compute_optimizer_memory_bytes(state, num_workers=1, sharded=False):
    # Goal: return per-worker bytes of Adam state (m and v), dividing by num_workers if sharded.
    # Sum the memory used by the first (m) and second (v) moments
    m_bytes = sum(tensor.nbytes for tensor in state.get('m', {}).values())
    v_bytes = sum(tensor.nbytes for tensor in state.get('v', {}).values())
    
    total_bytes = m_bytes + v_bytes
    
    # If ZeRO sharding is active, the memory is distributed across workers
    if sharded and num_workers > 0:
        return total_bytes // num_workers
        
    return total_bytes

# Step 38 - compute_peak_activation_memory_bytes
def compute_peak_activation_memory_bytes(x, params, checkpointed=False):
    # Goal: return total bytes of activations retained by the forward cache
    batch_size = x.shape[0]
    
    # The cache must always store at least the input array `x`
    total_elements = x.size
    
    if checkpointed:
        return total_elements * x.itemsize
        
    # Find all weight matrices and sort them by layer index (W1, W2, etc.)
    w_keys = sorted([k for k in params.keys() if k.startswith('W')], 
                    key=lambda k: int(k.replace('W', '')))
    
    for i, w_name in enumerate(w_keys):
        out_dim = params[w_name].shape[1]
        
        if i < len(w_keys) - 1:
            # Hidden layers: We cache pre-activations (z) AND activations (a)
            # Therefore, we add the output dimension size twice.
            total_elements += 2 * batch_size * out_dim
        else:
            # Final output layer: We only cache pre-activations (logits)
            total_elements += batch_size * out_dim
            
    # Multiply the total scalar elements by the byte-size of the data type (e.g., 8 for float64)
    return total_elements * x.itemsize

# Step 39 - compare_memory_with_and_without_optimizations
def compare_memory_with_and_without_optimizations(x, params, num_workers):
    # Goal: report baseline vs optimized per-worker memory (params, optimizer, activations) and savings ratio.
    # Count total parameter elements
    param_elements = sum(p.size for p in params.values())
    
    # 1. Parameter Memory
    # Baseline: FP32 (4 bytes per element)
    # Optimized: Mixed Precision FP16 (2 bytes per element). (Not sharded in ZeRO-1/2)
    base_params = param_elements * 4
    opt_params = param_elements * 2
    
    # 2. Optimizer Memory
    # Adam stores 2 moments (m and v). Standard is FP32, so 2 * 4 = 8 bytes per param.
    base_opt = param_elements * 8
    opt_opt = base_opt // num_workers # ZeRO shards the optimizer state
    
    # 3. Activation Memory
    # Baseline: Full cache, matching input dtype
    base_act = compute_peak_activation_memory_bytes(x, params, checkpointed=False)
    
    # Optimized: Checkpointed, and cast to FP16 for mixed precision
    x_16 = x.astype(np.float16)
    opt_act = compute_peak_activation_memory_bytes(x_16, params, checkpointed=True)
    
    # 4. Totals
    base_total = base_params + base_opt + base_act
    opt_total = opt_params + opt_opt + opt_act
    
    # 5. Savings Ratio (Fraction representing how much was saved)
    savings_ratio = (base_total - opt_total) / base_total
    
    return {
        'baseline_bytes': base_total,
        'optimized_bytes': opt_total,
        'savings_ratio': savings_ratio,
        'breakdown_baseline': {
            'params': base_params,
            'optimizer': base_opt,
            'activations': base_act
        },
        'breakdown_optimized': {
            'params': opt_params,
            'optimizer': opt_opt,
            'activations': opt_act
        }
    }

# Step 40 - full_distributed_training_loop
def full_distributed_training_loop(x, y, num_workers=2, num_steps=10, micro_batch_size=8, lr=1e-3, hidden_dim=16, use_checkpointing=True, use_mixed_precision=True, use_zero=True, seed=0):
    # Goal: run end-to-end distributed memory-aware training and return loss_history and final_params.
    in_dim = x.shape[1]
    out_dim = y.shape[1]
    
    # 1. Initialize parameters
    params = init_mlp_params(in_dim, hidden_dim, out_dim, seed=seed)
    
    # Keep master params in their original high precision (fp64)
    master_params = {k: v.copy() for k, v in params.items()}
    
    # 2. Setup ZeRO optimizer state if enabled
    if use_zero:
        adam_state = init_adam_state(master_params)
        worker_states = partition_optimizer_state(adam_state, num_workers)
        
    scale = 1024.0 # Static loss scaling factor for mixed precision
    
    # 3. Shard dataset across workers
    shards = shard_dataset_across_workers(x, y, num_workers)
    
    loss_history = []
    
    for step in range(num_steps):
        all_worker_grads = []
        step_loss = 0.0
        
        # Simulate parallel execution across workers
        for w in range(num_workers):
            x_local, y_local = shards[w]
            micro_batches = split_into_micro_batches(x_local, y_local, micro_batch_size)
            
            accum_grads = None
            worker_total_loss = 0.0
            
            # Iterate through micro-batches for Gradient Accumulation
            for x_mb, y_mb in micro_batches:
                
                # Setup Precision
                if use_mixed_precision:
                    active_params = cast_to_half_precision(master_params)
                    x_in = x_mb.astype(np.float16)
                    y_in = y_mb.astype(np.float16)
                else:
                    active_params = master_params
                    x_in = x_mb
                    y_in = y_mb
                    
                # Forward Pass (Memory-Aware)
                if use_checkpointing:
                    y_pred, cache = mlp_forward_checkpointed(x_in, active_params)
                else:
                    y_pred, cache = mlp_forward(x_in, active_params)
                    
                loss, dy_pred = mse_loss_and_grad(y_pred, y_in)
                worker_total_loss += float(loss)
                
                # Backward Pass
                if use_mixed_precision:
                    sl, sdy = scale_loss(loss, dy_pred, scale)
                    
                    if use_checkpointing:
                        scaled_grads = mlp_backward_checkpointed(sdy, cache, active_params)
                    else:
                        scaled_grads = mlp_backward(sdy, cache, active_params)
                        
                    new_grads = unscale_gradients(scaled_grads, scale)
                else:
                    if use_checkpointing:
                        new_grads = mlp_backward_checkpointed(dy_pred, cache, active_params)
                    else:
                        new_grads = mlp_backward(dy_pred, cache, active_params)
                        
                # Accumulate Micro-batch Gradients
                accum_grads = accumulate_gradients(accum_grads, new_grads)
                
            # Scale accumulated gradients by number of micro-batches
            final_local_grads = scale_accumulated_gradients(accum_grads, len(micro_batches))
            all_worker_grads.append(final_local_grads)
            
            # Average micro-batch losses for this worker
            step_loss += (worker_total_loss / len(micro_batches))
            
        # Record mean loss across all workers
        loss_history.append(step_loss / num_workers)
        
        # Synchronization (All-Reduce Gradients)
        mean_grads = all_reduce_mean(all_worker_grads)
        
        # Parameter Update
        skip_update = False
        if use_mixed_precision:
            skip_update = has_non_finite_gradients(mean_grads)
            
        if not skip_update:
            if use_zero:
                # ZeRO-sharded Adam update
                master_params, worker_states = zero_optimizer_step(
                    master_params, mean_grads, worker_states, lr=lr
                )
            else:
                # Standard DP SGD update
                master_params = {
                    k: master_params[k] - lr * mean_grads[k] 
                    for k in master_params
                }
                
    return {'loss_history': loss_history, 'final_params': master_params}

