Title: Joint KV Cache Sparsification and Quantization for Efficient Self-Speculative Decoding
Team Members: Harry Hu (yuehanh), Jerry Liang (zhanminl), Soham Khatavkar (skhatavk)

Introduction
Large language model serving is increasingly bottlenecked by the growth of the key-value (KV) cache during autoregressive decoding. Self-speculative decoding improves inference efficiency by using a cheaper draft version of the same model to propose tokens, which are then verified by the full model. Recent work such as QuantSpec shows that a quantized draft cache can preserve high acceptance rates while significantly improving throughput. Meanwhile, prior work such as H2O and SnapKV shows that attention is highly sparse, with most attention mass concentrated on a small set of heavy-hitter tokens, enabling substantial KV-cache reduction with limited quality degradation.

These two compression directions, however, have mostly been studied independently. We propose to study their interaction in the draft path of self-speculative decoding. Our hypothesis is that combining sparsification and quantization can reduce draft-side memory and compute cost more than either method alone, but may also lower proposal quality and thus reduce acceptance rate. This creates a systems tradeoff between draft efficiency and verification efficiency that has not been well characterized.

Problem
Our project asks whether jointly applying KV-cache sparsification and quantization to the draft path of self-speculative decoding can improve end-to-end decoding throughput beyond using only one technique at a time. In our setup, the verifier always uses the full-precision KV cache, so the final outputs remain identical to standard decoding. The challenge is that a more aggressively compressed draft cache generates proposals faster, but may decrease acceptance rate and increase wasted verification work. We aim to characterize this tradeoff and identify operating points that maximize throughput under memory constraints.

Status Quo
QuantSpec studies hierarchical quantized KV caches for self-speculative decoding and reports substantial speedups while maintaining high acceptance rates. H2O and SnapKV introduce attention-score-based token eviction methods for standard LLM inference by retaining heavy-hitter tokens together with a recent-token window. KIVI and MiniKV further show that aggressive KV-cache quantization can be effective in practice. However, prior work has not directly studied the interaction between sparsification and quantization within the draft path of self-speculative decoding. Our project focuses on this gap and evaluates whether these two compression mechanisms are complementary or conflicting in practice.

 
High-Level Implementation Plan
We will build a self-speculative decoding prototype in PyTorch for Llama-2-7B using Hugging Face components and vLLM-style KV-cache abstractions where possible, running on Modal GPU instances. The verifier will use a full FP16 KV cache as the source of truth, while the draft path will operate on a compressed cache. For the draft cache, we will apply SnapKV-style heavy-hitter selection with a fixed recent window, then quantize the retained KV entries to INT8 or INT4. The draft will propose K candidate tokens, which the verifier will check against the full cache. We will compare four settings: standard autoregressive decoding, quantization-only draft compression, sparsification-only draft compression, and the combined method.

Evaluation
We will evaluate each method on 50–100 prompts from MT-Bench while sweeping sparsification ratios, quantization levels, draft lengths, and context lengths. Our primary metrics are acceptance rate, tokens per second, peak KV-cache memory, and per-token latency. We will also analyze how the combined method changes the tradeoff frontier relative to quantization-only and sparsification-only baselines. If time permits, we will explore periodic heavy-hitter refresh and per-head token selection as extensions.
References
 Li, Yuhong, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM Knows What You are Looking for Before Generation. NeurIPS, 2024.
Liu, Zirui, Jiayi Yuan, Hong Jin, Shaofeng Zhong, Zizheng Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML, 2024.
Tiwari, Rishabh, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache. ICML, 2025.
Zhang, Zhenyu, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. NeurIPS, 2023.
Sharma, Akshat, Hangliang Ding, Jianping Li, Neel Dani, and Minjia Zhang. MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache. arXiv:2411.18077, 2024.



