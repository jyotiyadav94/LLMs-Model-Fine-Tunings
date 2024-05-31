# Named Entity Recognition on Italian Emails

## Overview

This project aims to identify the best Open Large Language Model (LLM) for the Named Entity Recognition (NER) task on Italian emails. The models being compared are:

1. **LLama-2 7B**
2. **Llama-3 8B**
3. **Microsoft Phi1.5**
4. **Mistral 7B**
5. **Mistral 8*7B**
6. **Gemma 7B**

## Model Characteristics

### 1. LLaMA-2 7B

**Research Paper:** [LLaMA-2: Summarization](https://arxiv.org/abs/2307.09288)

LLaMA-2 is a versatile, open-source LLM designed for multiple tasks such as **text generation, language translation, and creative writing**. It is available in various sizes, ranging from 7 billion ,13 billion,70 billion parameters, with larger models generally delivering better performance. Safety measures have been incorporated to filter out harmful content and enhance usability.

* Llama 2, an updated version of Llama 1, trained on a new mix of publicly available data. We also
increased the size of the pretraining corpus by 40%, doubled the context length of the model, and
adopted grouped-query attention (Ainslie et al., 2023). We are releasing variants of Llama 2 with
7B, 13B, and 70B parameters. We have also trained 34B variants, which we report on in this paper
but are not releasing.§

* Llama 2-Chat, a fine-tuned version of Llama 2 that is optimized for dialogue use cases. We release
variants of this model with 7B, 13B, and 70B parameters as well.

**Key Features:**
- 7 billion parameters
- Trained on a diverse, extensive dataset
- Improved context length (up to 4,096 tokens)
- Enhanced safety and usability

**Repository**: [LLaMA-2 7B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/tree/main/Llama2-7B)

### 2. LLaMA-3 8B

**Research Paper:** [LLaMA-3: Enhanced Capabilities](https://arxiv.org/abs/2404.19553)

LLaMA-3, released by Meta, includes significant enhancements over its predecessor, LLaMA-2 & LLaMA-1. It comes in 8 billion and 70 billion parameter variants, offering improved performance and larger context sizes. LLaMA-3's context length has been increased from 4,096 to 8,192 tokens, with potential for further expansion up to 80K. This model has been trained on a massive dataset of 15 trillion tokens, incorporating high-quality non-English data to boost multilingual capabilities. Advanced training techniques, such as instruction fine-tuning, have optimized LLaMA-3 for chat and dialogue applications. Meta emphasizes safety with a comprehensive, multi-level development approach.

**Key Features:**
- 8 billion parameters
- Enhanced context length (up to 8,192 tokens)
- Training on diverse, high-quality data
- Advanced instruction fine-tuning

**Repository**: [LLaMA-3 8B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/blob/main/Llama3-8B/Llama_3_8b_2x_faster_finetuning.ipynb)

### 3. Microsoft Phi-1.5

  **Research Paper:** [LLaMA-3: Enhanced Capabilities](https://arxiv.org/abs/2309.05463)
  
  This paper introduces phi-1.5, a 1.3 billion parameter language model trained primarily on a specially curated "textbook-quality" synthetic dataset. The key findings are:
- Phi-1.5 achieves performance comparable to models **10x larger on common sense reasoning and language understanding tasks**, and even exceeds larger models on **multi-step reasoning tasks like math and coding.**
- The high performance despite smaller size challenges the notion that capabilities of large language models are solely determined by scale, suggesting data quality plays a crucial role.
- Training on synthetic textbook-like data attenuates toxic content generation compared to web data, though phi-1.5 is not immune.
- Phi-1.5 exhibits many traits of much larger models like step-by-step reasoning and in-context learning, both positive and negative like hallucinations.
- The authors open-source phi-1.5 to facilitate research on urgent topics around large language models like in-context learning, bias mitigation, and hallucinations.
- The paper demonstrates phi-1.5's flexible capabilities on tasks like question-answering, coding, and open-ended generation through various prompting techniques.

**Key Features:**
- 1.3 billion parameters
- Context length of 2,048 tokens
- High performance on reasoning and language understanding tasks
- Reduced toxic content generation

**Repository**: [Microsoft Phi-1.5-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/blob/main/Microsoft%20Phi1.5/Finetuning_Gathnex_Phi_1_5.ipynb)

### 4. Mistral 7B

**Research Paper:** [Mistral 7B](https://arxiv.org/abs/2310.06825)

Mistral 7B is a 7-billion parameter language model designed for high performance and efficiency. It outperforms the best open 13B model (Llama 2) on all evaluated benchmarks and even surpasses the best 34B model (Llama 1) in reasoning, mathematics, and code generation tasks.

**Technical Aspects:**
- Mistral 7B has a context length of 8,192 tokens. This is double the context length of many previous models and allows Mistral 7B to process and generate longer sequences of text, making it suitable for tasks like summarizing long documents or engaging in extended conversations.
- Grouped-Query Attention (GQA) This mechanism speeds up inference and reduces memory requirements during decoding, allowing for larger batch sizes and higher throughput.
- Sliding Window Attention (SWA) This allows the model to handle longer sequences of text effectively at a reduced computational cost. It achieves this by limiting the attention span of each token to a window of the most recent tokens, thereby reducing the quadratic complexity of traditional attention mechanisms.
- Rolling Buffer Cache This further optimizes memory usage by storing only the most recent keys and values within a fixed-size cache.
- Pre-fill and Chunking These techniques are used to efficiently process long input sequences by pre-filling the cache with the prompt and breaking down large prompts into smaller chunks.
- Mistral 7B is released under the Apache 2.0 license, making it open source and available for both research and commercial use.
- The model is designed for easy fine-tuning on various tasks. A fine-tuned version, Mistral 7B-Instruct, outperforms Llama 2 13B-Chat on both human and automated benchmarks.
- The paper emphasizes the importance of balancing high performance with efficiency in large language models.
- The authors suggest that their work with Mistral 7B opens up new possibilities for achieving high performance with smaller models, potentially leading to more cost-effective and sustainable AI systems.

**Repository**: [Mistral 7B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/tree/main/Mistral7B)

### 5. Mistral 8* 7B

**Research Paper:** [Mistral 8X7B](https://arxiv.org/abs/2401.04088)

Mixtral 8x7B is a sparse mixture of experts model (SMoE) with the same architecture as Mistral 7B, but each layer is composed of 8 feedforward blocks (experts). A router network selects two experts to process the current state and combine their outputs for every token at each layer. Although each token only sees two experts, the selected experts can be different at each time step. As a result, each token has access to 47B parameters but only uses 13B active parameters during inference.

**Key Features:**
- **Sparse Mixture of Experts:** Architecture with 8 feedforward blocks per layer
- **Router Network:** Selects two experts per token per layer
- **Efficient Inference:** Access to 47B parameters with only 13B active during inference
- **Context Length:** Trained with a context size of 32k tokens

**Performance:**
- Outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks
- Excels in mathematics, code generation, and multilingual benchmarks
- Fine-tuned version, Mixtral 8x7B-Instruct, surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B-chat on human benchmarks

**Repository**: [Mistral8X7B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/blob/main/Mistral8*7B/Mixtral_fine_tuning8X7B.ipynb)

### 6. Gemma 7B

**Research Paper:** [Gemma-7B](https://arxiv.org/abs/2403.08295)

This paper introduces Gemma, a family of open language models based on the research and technology used to create Google's Gemini models. The key points are:

- Gemma comes in two sizes: a 7 billion parameter model and a 2 billion parameter model, designed for different computational constraints and applications.
- The models are trained on up to 6 trillion tokens of text data, using architectures and training recipes inspired by Gemini.
- Gemma advances state-of-the-art performance compared to other open models of similar or larger size on a wide range of benchmarks, including question answering, reasoning, mathematics, and coding.
- The paper provides details on the model architecture, training infrastructure, pretraining and finetuning procedures, including supervised fine-tuning and reinforcement learning from human feedback.
- Comprehensive evaluations are presented on automated benchmarks as well as human preference evaluations against other models like Mistral.
- The paper discusses the approach to responsible development and deployment, including assessments of potential benefits and risks, as well as mitigations for risks.

**Technical Aspects:**
- The models are trained with a context length of 8,192 tokens.
- Uses techniques like multi-query attention, rotary position embeddings, and gated activation functions.
- The 7B model uses multi-head attention while the 2B model uses multi-query attention.

**Key Features:**
- 7 billion parameters
- Specialized in multilingual NER
- Trained on diverse multilingual data
- Advanced state-of-the-art performance on various benchmarks

**Repository**: [Gemma-7B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/blob/main/Microsoft%20Phi1.5/Finetuning_Gathnex_Phi_1_5.ipynb)

## Conclusion

This project evaluates the performance of the aforementioned LLMs on the NER task for Italian emails. By comparing their all the fine-tuned LLMs models, we aim to identify the most suitable model for this specific task. 

1. Llama3-8B
2. Microsoft Phi1.5

## References 
1. https://arxiv.org/abs/2305.07759
2. https://huggingface.co/unsloth/llama-3-8b-bnb-4bit
