# Fine-tune a Llama 2 7B parameters in 4-bit to generate Python Code

A demo on how to fine-tune the new Llama-2 using PEFT, QLoRa and the Huggingface utilities

**This repository is still In progress**

## The dataset

For our tuning process, we will take a [dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) containing about 18,000 examples where the model is asked to build a Python code that solves a given task. 
This is an extraction of the [original dataset](https://huggingface.co/datasets/sahil2801/code_instructions_120k) where only the Python language examples are selected. Each row contains the description of the task to be solved, an example of data input to the task if applicable, and the generated code fragment that solves the task is provided.

## Problem description
Fine tune the pretrained model, Llama 2 7B parameters, using 4-bit quantization to produce a Python coder.

**Note:** This is still in progress and some models maybe need some more experimentation to return good answers.


## The model
[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b)

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety

## Content
**Still In progress**

- Fine-tuning notebook `Llama-2-finetune-qlora-python-coder.ipynb`: In this notebook we fine-tune the model.
- Fine-tuning script `train.py`: An script to run training process.
- Notebook to run the script `run-script-finetune-llama-2-python-coder.ipynb`: An very simple example on how to use NER to search for relevant articles.

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a public GNU License version 3.