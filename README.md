# 👩‍💻 Fine-tune a Llama 2 7B parameters in 4-bit to generate Python Code

**LlaMa-2 7B** model fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the method **QLoRA** in 4-bit with [PEFT](https://github.com/huggingface/peft) library.


## The dataset

For our tuning process, we will take a [dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) containing about 18,000 examples where the model is asked to build a Python code that solves a given task. 
This is an extraction of the [original dataset](https://huggingface.co/datasets/sahil2801/code_instructions_120k) where only the Python language examples are selected. Each row contains the description of the task to be solved, an example of data input to the task if applicable, and the generated code fragment that solves the task is provided.

## Problem description

Our goal is to fine-tune the pretrained model, Llama 2 7B parameters, using 4-bit quantization to produce a Python coder. We will run the training on Google Colab using a A100 to get better performance. But you can try out to run it on a T4 adjusting some parameters to reduce memory consumption like batch size.

**Note:** This is still in progress and some models maybe need some more experimentation to return good answers.


## The base model
[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b)

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety

## Content
**Still In progress**

- Fine-tuning notebook `Llama-2-finetune-qlora-python-coder.ipynb`: In this notebook we fine-tune the model.
- Fine-tuning script `train.py`: An script to run training process.
- Notebook to run the script `run-script-finetune-llama-2-python-coder.ipynb`: An very simple example on how to use NER to search for relevant articles.

### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/llama-2-7b-int4-python-code-20k"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, 
                                             device_map=device_map)

instruction="Write a Python function to display the first and last elements of a list."
input=""

prompt = f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the Task.

### Task:
{instruction}

### Input:
{input}

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.3)

print(f"Prompt:\n{prompt}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

```
### Citation

```
@misc {edumunozsala_2023,
	author       = { {Eduardo Muñoz} },
	title        = { llama-2-7b-int4-python-coder },
	year         = 2023,
	url          = { https://huggingface.co/edumunozsala/llama-2-7b-int4-python-18k-alpaca },
	publisher    = { Hugging Face }
}
```
## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a public GNU License version 3.