# Project Aim

The main objective of this project is to perform Named Entity Recognition (NER) on Italian emails using the LLAMA 2 7B parameter model, employing the Parameter-Efficient Fine-Tuning (PEFT) technique called LORA.

## Dataset Description
The dataset is supervised and contains words and entity_groups.

### Prompt Generation Function

```python
def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Extract entity from the given input:
### Input:
{data_point["input_text"]}
### Response:
{data_point["output_text"]}"""
```


## Fine-tuning the Meta-Llama/Llama-2-7b-hf model

### Introduction
In recent times, Large Language Models (LLMs) have garnered significant attention and discussion, with ChatGPT being a prominent example. These models, whether open-source or closed-source, continue to evolve, offering advancements in various aspects without necessarily increasing in size. This presents an opportune moment to delve into their functionalities and explore how they can be tailored to suit specific requirements.

we will explore Llama2, a recent addition to the open-source LLM landscape, and demonstrate its application through fine-tuning for a standard Natural Language Processing (NLP) task of entity recognition within text. Initially, we'll provide an overview of large language models, differentiate between open-source and closed-source models, and offer some examples. Subsequently, we'll delve into the specifics of Llama2 and elucidate its distinguishing features. Following this, we'll outline our NLP task and the dataset involved before delving into the coding aspect.

### About Large Language Models (LLMs)
Language models are artificial intelligence systems designed to comprehend and generate human language. LLMs such as GPT-3, ChatGPT, GPT-4, Bard, and others exhibit remarkable versatility, capable of performing a wide array of tasks. However, the quality of their outputs often hinges on the precision of the input prompts provided by users.

These models are trained on extensive textual data sourced from the internet, encompassing a diverse range of written materials spanning from literature to social media posts. With applications across various domains including chatbots, virtual assistants, content creation, and more, LLMs find utility in industries such as customer service, healthcare, finance, and marketing.

Due to their extensive training data, LLMs excel at zero-shot inference and can be fine-tuned to improve performance with minimal additional examples. Zero-shot inference enables models to recognize entities not encountered during training, while few-shot learning involves making predictions for new classes based on limited labeled data provided during inference.

Despite their remarkable text generation capabilities, these models are not without limitations, including issues such as hallucinations and biases, which necessitate consideration when integrating them into production pipelines.

### Closed and Open-source Language Models
Closed-source LLMs are proprietary models utilized by certain companies and are not publicly accessible. The training data for these models is typically proprietary as well, which may raise concerns regarding transparency, bias, and data privacy.

In contrast, open-source projects like GPT-3 are freely available to researchers and developers, trained on publicly accessible datasets, fostering transparency and collaboration.

The choice between closed and open-source LLMs depends on various factors including project goals and the importance of transparency and accessibility.

### About LLama2
Meta's open-source LLM, Llama2, was trained on a vast corpus of 2 trillion tokens sourced from publicly available repositories such as Wikipedia, Common Crawl, and the Gutenberg Project. Offering three different parameter-level model versions (7 billion, 13 billion, and 70 billion parameters) and two types of completion models (Chat-tuned and General), Llama2 provides flexibility and adaptability. In this blog post, we will utilize the general Meta's 7b Llama-2 Huggingface model for fine-tuning, although other versions of Llama2-7b are also available.

About Named Entity Recognition (NER)
Named Entity Recognition (NER) is a crucial component of information extraction, tasked with identifying and categorizing specific entities within unstructured text into predefined groups such as individuals, organizations, locations, and more. NER facilitates a rapid comprehension of the primary concepts or content within lengthy texts.

I will be focusing on fine-tuning Llama2-7b using PEFT techniques on a Colab Notebook, transforming the NER dataset(Containing words & its labels) for NER purposes. 



### Notebook utilities
**1. Load the model and tokenizer.**

* Use the `AutoTokenizer` and `AutoModelForCausalLM` classes from the Transformers library to load the model and tokenizer.
* Set the `use_auth_token` parameter to the authentication token that is provided by Hugging Face.

**2. Prepare the dataset for fine-tuning.**

* Read the CSV file named `Qspot-Sea Final Annotation reviwed by sandeep - combined_df.csv`. This file contains the data that will be used to fine-tune the model.
* Rename the columns of the CSV file to `input_text` and `output_text`.
* Create a function named `generate_peft_config` that takes a model as input and returns a dictionary of configuration parameters.
* The function first creates a `LoraConfig` object with the specified parameters.
* Then, it prepares the model for int8 training, enables input require grads,
* and gets a `peft_model` from the model and `peft_config`.



**3. Fine-tune the model.**

* Define the training arguments using the `TrainingArguments` class.
* The `per_device_train_batch_size` parameter is set to 1, which means that each training process will use one GPU.
* The `gradient_accumulation_steps` parameter is set to 16, which means that the gradients will be accumulated 16 times before being updated.
* The `learning_rate` parameter is set to 4e-05.
* The `logging_steps` parameter is set to 50, which means that training progress will be logged every 50 steps.
* The `optim` parameter is set to `adamw_torch`, which is a type of optimizer that is well-suited for training language models.
* The `evaluation_strategy` parameter is set to `steps`, which means that the model will be evaluated every 50 steps.
* The `save_strategy` parameter is also set to `steps`, which means that the model will be saved every 50 steps.
* The `eval_steps` parameter is set to 50, which is the number of training steps that will be evaluated before the model is saved.
* The `save_steps` parameter is also set to 50.
* The `output_dir` parameter is set to `/content/drive/Shareddrives/dataset/clients/LLama2/`, which is the directory where the training artifacts will be saved.

**4. Evaluate the model.**

* Once the model has been fine-tuned, evaluate its performance on the testing set.
* The `trainer.evaluate()` method is called to evaluate the model.

**5. Save the model.**

* Once the model has been evaluated, save it to disk using the `model.save_pretrained("./finetuned_model")` method.
* You can then load the model and use it for inference.

