# Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Build a complete training stack in pure NumPy that mirrors how modern frameworks scale models across devices and squeeze them into limited memory. This repository showcases implementing an MLP with manual autograd and then additionally layer on gradient accumulation, activation checkpointing, mixed precision, data parallel all-reduce, and ZeRO-style optimizer sharding to train under realistic memory budgets.

## How to run

```bash
python scaffold.py
```

## Steps

* \[x] **1.** make\_synthetic\_regression\_batch
* \[x] **2.** init\_mlp\_params
* \[x] **3.** linear\_forward
* \[x] **4.** relu\_forward
* \[x] **5.** mlp\_forward
* \[x] **6.** mse\_loss\_and\_grad
* \[x] **7.** linear\_backward
* \[x] **8.** relu\_backward
* \[x] **9.** first\_linear\_backward
* \[x] **10.** mlp\_backward
* \[x] **11.** split\_into\_micro\_batches
* \[x] **12.** accumulate\_gradients
* \[x] **13.** scale\_accumulated\_gradients
* \[x] **14.** grad\_accumulation\_step
* \[x] **15.** mlp\_forward\_checkpointed
* \[x] **16.** recompute\_block\_activations
* \[x] **17.** mlp\_backward\_checkpointed
* \[x] **18.** estimate\_checkpointing\_memory\_savings
* \[x] **19.** cast\_to\_half\_precision
* \[x] **20.** make\_master\_params
* \[x] **21.** scale\_loss
* \[x] **22.** unscale\_gradients
* \[x] **23.** has\_non\_finite\_gradients
* \[x] **24.** mixed\_precision\_step
* \[x] **25.** shard\_dataset\_across\_workers
* \[x] **26.** compute\_local\_gradients
* \[x] **27.** all\_reduce\_mean
* \[x] **28.** ring\_all\_reduce\_mean
* \[x] **29.** data\_parallel\_train\_step
* \[x] **30.** bucket\_gradients
* \[x] **31.** init\_adam\_state
* \[x] **32.** partition\_optimizer\_state
* \[x] **33.** local\_shard\_adam\_update
* \[x] **34.** all\_gather\_param\_shards
* \[x] **35.** zero\_optimizer\_step
* \[x] **36.** compute\_param\_memory\_bytes
* \[x] **37.** compute\_optimizer\_memory\_bytes
* \[x] **38.** compute\_peak\_activation\_memory\_bytes
* \[x] **39.** compare\_memory\_with\_and\_without\_optimizations
* \[x] **40.** full\_distributed\_training\_loop

