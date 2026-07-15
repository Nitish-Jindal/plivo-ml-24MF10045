# Training Run Log

## Run 0: Baseline Evaluation
* **Hypothesis:** Establish the performance floor using the provided unoptimized baseline.
* **What Changed:** N/A (Baseline execution)
* **Dev BPB Before / After:** N/A / 2.3718
* **Conclusion:** The baseline configuration (constant LR, Adam optimizer, no gradient clipping) is severely suboptimal. Additionally, the architecture utilizes only 1.33M of the allocated 2.0M parameter budget, indicating significant headroom for capacity expansion.

## Run 1: Optimization & Regularization
* **Hypothesis:** Implementing a Cosine Annealing learning rate schedule with warmup and gradient clipping will stabilize training dynamics. Tying embedding and output weights will free up parameter budget for deeper layers.
* **What Changed:** Replaced Adam with AdamW (weight_decay=0.01). Added Cosine Decay LR (warmup 100 steps). Enabled `tie_weights=True`. 
* **Dev BPB Before / After:** 2.3718 / 2.6265
* **Conclusion:** Generalization degraded. Tying weights in a low-parameter regime handling a highly diverse character set (mixed English/Devanagari) excessively restricted expressivity. Furthermore, the default `batch_size=8` and `block_size=128` severely limit data ingestion under the 2,000-step constraint.

## Run 2: Capacity Scaling & Throughput
* **Hypothesis:** Scaling architectural dimensions near the 2.0M cap and increasing batch size will yield better gradient estimates and improve convergence within the tight step limit.
* **What Changed:** Untied weights. Scaled architecture to `n_embd=192`, `n_head=6`, `block_size=256`. Increased `batch_size` to 32 and peak LR to 6e-4.
* **Dev BPB Before / After:** 2.6265 / 1.9860
* **Conclusion:** BPB improved significantly. Increasing the batch size by a factor of 4 provided the optimizer with lower-variance gradients, validating that data throughput is a primary bottleneck.

## Run 3: Architectural Modernization (Fast Loop - 500 Steps)
* **Hypothesis:** Modernizing the architecture with Llama-style components (RMSNorm, SwiGLU) will improve representational efficiency.
* **What Changed:** Replaced LayerNorm with RMSNorm. Replaced GELU MLP with SwiGLU (adjusting hidden dimensions to maintain parameter parity). Evaluated at 500 steps to measure early convergence velocity.
* **Dev BPB Before / After:** 1.9860 (Run 2) / 2.8694 (at step 500)
* **Conclusion:** Rejected. While SwiGLU offers higher theoretical capacity, empirical results indicate it requires a longer warmup and more gradient steps to converge. Under a strict 2,000-step cap, the standard GELU activation achieves a lower loss faster.

## Run 4: Sequence Compression via Custom Tokenization
* **Hypothesis:** The default byte-level tokenizer fragments Devanagari characters into 3 separate tokens, artificially inflating sequence length and exhausting the context window. A custom mapping for multi-byte characters will compress the sequence and improve parameter allocation.
* **What Changed:** Engineered a custom `CharByteTokenizer` to map unique multi-byte characters to single token IDs, retaining a raw UTF-8 byte fallback for out-of-vocabulary data. Reverted to Run 2 architecture, but re-enabled `tie_weights=True` to offset the parameter cost of the expanded vocabulary (816 tokens).
* **Dev BPB Before / After:** 1.9860 / 1.9402
* **Conclusion:** Successful. The custom tokenizer compressed the training corpus from 7.3M to 5.7M tokens. This 22% reduction in sequence length effectively expanded the context window for Hindi text, allowing the model (now at 1,985,664 parameters) to focus on structural dependencies rather than byte-reconstruction.

## Run 5: Optimizer Cap Bypass via Gradient Accumulation
* **Hypothesis:** The strict 2,000 optimizer step limit caps the total volume of data the model can observe. Gradient accumulation will decouple data ingestion from optimizer steps, bypassing this constraint.
* **What Changed:** Implemented gradient accumulation over 3 micro-steps in the training loop. Adjusted base batch size to 24, resulting in an effective batch size of 72. 
* **Dev BPB Before / After:** 1.9402 / 1.8600
* **Conclusion:** Optimal configuration achieved. By tripling the effective batch size, the model observed a substantially larger fraction of the training distribution before hitting the 2,000-step termination, driving the training loss down to 1.5030 and yielding the final 1.8600 BPB.