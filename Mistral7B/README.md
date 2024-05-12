# Mistral 7B Model Fine-tuning Guide

This guide will walk you through the steps to fine-tune the Mistral 7B model using the provided dataset and inference code.

## Dataset Information

The dataset required for fine-tuning the Mistral 7B model consists of two main files was trained on 703 emails:

1. **df_final_llm (1).csv**: This initial dataset contains preprocessed email entities. Each entry consists of a raw preprocessed email and its corresponding entities in a dictionary format.
2. **final_data.csv**: This file contains the merger of prompt + Input + Output for the fine-tuning process.

## Key Characteristics of Mistral Model

- **Model**: Mistral 8*7B (Llama 2 70B & GPT3.5), Mistral 7B (Llama2-13B)
- **Parameters**: 47B during training & 13B during inference
- **Capabilities**: Strong multilingual capabilities (FR, DE, ES, IT)
- **Long-range Reasoning**: Supports sequences up to 32k tokens

## Requirements

- Access to Kaggle or Google Collab Enterprise
- A powerful GPU is recommended (e.g., V100 or T4 X2)
- Training on 703 emails will take it 3-4 hours However if you are using T4X2 from kaggle it will take 1.5 hours

## Steps to Execute

1. **Dataset Preparation and Fine-tuning Adaptor Layer**:
   - Execute the notebook `mistral_7b_4bit_qlora_fine_tuning.ipynb`. This notebook contains code for dataset preparation and fine-tuning the adaptor layer.

2. **Combine Adapter Layer with Base Model**:
   - `merge_load (1) (1).ipynb` After fine-tuning the adapter layer, you need to combine it with the base model.
   - Load the base model and attach the adapter using `PeftModel`.
   - Run the inference, merge the model weights, and push it to the Hugging Face Hub.

3. **Inference Code**:
   - `inference-ner.ipynb` Utilize the provided inference code to test the fine-tuned model.
   - Analyze the results of the test to evaluate the performance of the model.

## Note on Prompt Engineering

Prompt engineering plays a crucial role in leveraging Mistral's capabilities effectively. You can utilize similar notebook setups for Mistral and modify prompts according to your requirements in the future.

If you encounter any issues or require further clarification, please feel free to reach out.


## P.s 
Please Ignore the count the email count as  9 in the notebook `mistral_7b_4bit_qlora_fine_tuning.ipynb`. since I performed multiple experiments so I didn't save the notebook runs from the 703 emails. Instead I saved the adopter layer to huggingFace and then performed the `merge_load (1) (1).ipynb` on the same model. 


###Inspiration - 
https://www.datacamp.com/tutorial/mistral-7b-tutorial
