# Named Entity Recognition on Italian Emails

## Overview

This project aims to identify the best Open Large Language Model (LLM) for the Named Entity Recognition (NER) task on Italian emails. The models being compared are:

1. **LLama-2 7B**
2. **Llama-3 8B**
3. **Microsoft Phi1.5**
4. **Mistral 7B**
5. **Mistral 8* 7B**
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

Mistral 7B is designed to balance performance and efficiency. It is a scalable LLM that performs well on various tasks, including NER. The model emphasizes efficient training and inference, making it suitable for deployment in resource-constrained environments.

**Key Features:**
- 7 billion parameters
- Efficient training and inference
- Scalable performance

**Repository**: [Mistral 7B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/tree/main/Mistral7B)

### 5. Mistral 8* 7B

**Research Paper:** [Mistral 8X7B](https://arxiv.org/abs/2401.04088)

Mistral 8* 7B is a variant of Mistral 7B, optimized for enhanced performance in specific applications. This model variant focuses on fine-tuning and optimization for specialized tasks, ensuring higher accuracy and efficiency.

**Key Features:**
- 7 billion parameters
- Enhanced performance for specific tasks
- Specialized fine-tuning

**Repository**: [Mistral8X7B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/blob/main/Mistral8*7B/Mixtral_fine_tuning8X7B.ipynb)

### 6. Gemma 7B

**Research Paper:** [Gemma-7B](https://arxiv.org/abs/2403.08295)

Gemma 7B is specifically designed for multilingual NER tasks. It excels in identifying and categorizing named entities in different languages, making it ideal for the NER task on Italian emails. The model is trained on a large and diverse multilingual dataset to ensure high accuracy and robustness.

**Key Features:**
- 7 billion parameters
- Specialized in multilingual NER
- Trained on diverse multilingual data

**Repository**: [Mistral8X7B-Fine-Tuned](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/blob/main/Microsoft%20Phi1.5/Finetuning_Gathnex_Phi_1_5.ipynb)

## Conclusion

This project evaluates the performance of the aforementioned LLMs on the NER task for Italian emails. By comparing their characteristics, training techniques, and specialized features, we aim to identify the most suitable model for this specific task. The detailed analysis and results of this study will guide the selection of the best model for practical deployment.


## References 
1. https://arxiv.org/abs/2305.07759
2. https://huggingface.co/unsloth/llama-3-8b-bnb-4bit
