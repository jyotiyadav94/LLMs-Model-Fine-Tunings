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

**Repository**: [AI-in-industry-Anomaly-Detection](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/tree/main/Llama2-7B)

### 2. LLaMA-3 8B

**Research Paper:** [LLaMA-3: Enhanced Capabilities](https://arxiv.org/abs/2404.19553)

LLaMA-3, released by Meta, includes significant enhancements over its predecessor, LLaMA-2 & LLaMA-1. It comes in 8 billion and 70 billion parameter variants, offering improved performance and larger context sizes. LLaMA-3's context length has been increased from 4,096 to 8,192 tokens, with potential for further expansion up to 80K. This model has been trained on a massive dataset of 15 trillion tokens, incorporating high-quality non-English data to boost multilingual capabilities. Advanced training techniques, such as instruction fine-tuning, have optimized LLaMA-3 for chat and dialogue applications. Meta emphasizes safety with a comprehensive, multi-level development approach.

**Key Features:**
- 8 billion parameters
- Enhanced context length (up to 8,192 tokens)
- Training on diverse, high-quality data
- Advanced instruction fine-tuning

**Repository**: [AI-in-industry-Anomaly-Detection](https://github.com/jyotiyadav94/LLMs-Model-Fine-Tunings/tree/main/Llama2-7B)

### 3. Microsoft Phi-1.5

**Research Paper:** [Microsoft Phi-1.5: Multilingual Optimization](https://arxiv.org/abs/2401.01945)

Microsoft Phi-1.5 is an open-source LLM optimized for multilingual tasks, including NER in various languages. It focuses on robustness and accuracy in understanding and generating multilingual content. The model incorporates extensive pretraining on a diverse dataset and employs advanced fine-tuning techniques to ensure high performance across different languages and tasks.

**Key Features:**
- Optimized for multilingual tasks
- Robust NER capabilities
- Extensive pretraining and fine-tuning

### 4. Mistral 7B

**Research Paper:** [Mistral: Scalable Language Models](https://arxiv.org/abs/2402.01935)

Mistral 7B is designed to balance performance and efficiency. It is a scalable LLM that performs well on various tasks, including NER. The model emphasizes efficient training and inference, making it suitable for deployment in resource-constrained environments.

**Key Features:**
- 7 billion parameters
- Efficient training and inference
- Scalable performance

### 5. Mistral 8* 7B

**Research Paper:** [Mistral: Scalable Language Models](https://arxiv.org/abs/2402.01935)

Mistral 8* 7B is a variant of Mistral 7B, optimized for enhanced performance in specific applications. This model variant focuses on fine-tuning and optimization for specialized tasks, ensuring higher accuracy and efficiency.

**Key Features:**
- 7 billion parameters
- Enhanced performance for specific tasks
- Specialized fine-tuning

### 6. Gemma 7B

**Research Paper:** [Gemma: Multilingual NER](https://arxiv.org/abs/2403.01995)

Gemma 7B is specifically designed for multilingual NER tasks. It excels in identifying and categorizing named entities in different languages, making it ideal for the NER task on Italian emails. The model is trained on a large and diverse multilingual dataset to ensure high accuracy and robustness.

**Key Features:**
- 7 billion parameters
- Specialized in multilingual NER
- Trained on diverse multilingual data

## Conclusion

This project evaluates the performance of the aforementioned LLMs on the NER task for Italian emails. By comparing their characteristics, training techniques, and specialized features, we aim to identify the most suitable model for this specific task. The detailed analysis and results of this study will guide the selection of the best model for practical deployment.


## References 
1. 
2. https://huggingface.co/unsloth/llama-3-8b-bnb-4bit
