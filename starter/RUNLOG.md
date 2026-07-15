# Training Run Log

## Run 0: The Baseline
* **Hypothesis:** The provided baseline is deliberately unoptimized to establish a ceiling score.
* **What Changed:** N/A (Baseline execution)
* **Dev BPB Before / After:** N/A / 2.3718
* **Conclusion:** The baseline uses a constant learning rate, standard Adam, no gradient clipping, and underutilizes the parameter cap (1.33M / 2.0M). It needs aggressive optimization.

## Run 1: Optimizer & Schedule Overhaul
* **Hypothesis:** Adding a Cosine Annealing learning rate schedule with a warmup period and gradient clipping will stabilize training. Tying embedding weights will save parameters.
* **What Changed:** Replaced Adam with AdamW (weight_decay=0.01). Added Cosine Decay LR (warmup 100 steps). Tied weights. 
* **Dev BPB Before / After:** 2.3718 / 2.6265
* **Conclusion:** The bpb worsened. Tying the weights in a small model with a diverse character set restricted expressivity too heavily. Furthermore, the default batch size (8) and block size (128) are starving the model of data.

## Run 2: Maximize Parameter Budget & Throughput
* **Hypothesis:** Scaling the model closer to the 2M cap and drastically increasing batch size will force the model to learn more within the tight step limit.
* **What Changed:** Untied weights. Increased `n_embd` to 192, `n_head` to 6, `block_size` to 256. Increased batch size to 32 (4x throughput) and peak LR to 6e-4.
* **Dev BPB Before / After:** 2.6265 / 1.9860
* **Conclusion:** Massive success. Pushing more data through a larger architecture gave the optimizer much better gradients, breaking the 2.0 bpb barrier. 

## Run 3: The Character-Aware Tokenizer
* **Hypothesis:** The byte-level tokenizer is destroying the context window for Hindi text (3 tokens per Devanagari character). A custom vocab for multi-byte characters will compress the sequence and boost efficiency. 
* **What Changed:** Built a custom `CharByteTokenizer` that assigns a single ID to unique multi-byte characters from the corpus, falling back to bytes for unseen text. Reverted to Run 2 architecture, but re-tied weights to afford the expanded vocabulary (816).
* **Dev BPB Before / After:** 1.9860 / 1.9402
* **Conclusion:** The tokenizer compressed the training tokens from 7.3M to 5.7M. This effective expansion of the context window yielded our best model yet, maximizing our parameter budget at 1,985,664.