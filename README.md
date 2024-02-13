# 👩‍💻 Fine-tune Llama 2-like models to generate Python Code

## Llama-2 7B
**LlaMa-2 7B** model fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the method **QLoRA** in 4-bit with [PEFT](https://github.com/huggingface/peft) and bitsandbytes library.

Aditionally, we include a **GPTQ quantized version** of the model, **LlaMa-2 7B 4-bit GPTQ** using Auto-GPTQ integrated with Hugging Face transformers.
The quantization parameters for the GPTQ algo are:
- 4-bit quantization
- Group size is 128
- Dataset C4
- Decreasing activation is False

## TinyLlaMa 1.1B
We have also finetuned a TinyLlama 1.1 model using the **Axolot** library:

**TinyLlaMa 1.1B** fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the **Axolot** library in 4-bit with [PEFT](https://github.com/huggingface/peft) library.


## The dataset

For our tuning process, we will take a [dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) containing about 18,000 examples where the model is asked to build a Python code that solves a given task. 
This is an extraction of the [original dataset](https://huggingface.co/datasets/sahil2801/code_instructions_120k) where only the Python language examples are selected. Each row contains the description of the task to be solved, an example of data input to the task if applicable, and the generated code fragment that solves the task is provided.

## Problem description

Our goal is to fine-tune the pretrained model, Llama 2 7B parameters, using 4-bit quantization to produce a Python coder. We will run the training on Google Colab using a A100 to get better performance. But you can try out to run it on a T4 adjusting some parameters to reduce memory consumption like batch size.

Once the model is fine-tuned, we apply the GPTQ quantization to get a new model with a better inference time.

**Note:** This is still in progress and some models may be included in the future. 

## The base models
[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b)

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety

[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)

The [TinyLlama project](https://github.com/jzhang38/TinyLlama) aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, they can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀.

They adopted exactly the same architecture and tokenizer as Llama 2. This means TinyLlama can be plugged and played in many open-source projects built upon Llama. Besides, TinyLlama is compact with only 1.1B parameters. This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint.


## Quantization

A quick definition extracted from a great article in Medium by Benjamin Marie ["GPTQ or bitsandbytes: Which Quantization Method to Use for LLMs — Examples with Llama 2"](https://medium.com/towards-data-science/gptq-or-bitsandbytes-which-quantization-method-to-use-for-llms-examples-with-llama-2-f79bc03046dc) (Only for Medium subscribers)

*"GPTQ (Frantar et al., 2023) was first applied to models ready to deploy. In other words, once the model is fully fine-tuned, GPTQ will be applied to reduce its size. GPTQ can lower the weight precision to 4-bit or 3-bit. 
In practice, GPTQ is mainly used for 4-bit quantization. 3-bit has been shown very unstable (Dettmers and Zettlemoyer, 2023). It quantizes without loading the entire model into memory. Instead, GPTQ loads and quantizes the LLM module by module. 
Quantization also requires a small sample of data for calibration which can take more than one hour on a consumer GPU."*

## Content
**Still In progress**

- Fine-tuning notebook `Llama-2-finetune-qlora-python-coder.ipynb`: In this notebook we fine-tune the model.
- Fine-tuning script `train.py`: An script to run training process.
- Notebook to run the script `run-script-finetune-llama-2-python-coder.ipynb`: An very simple example on how to use NER to search for relevant articles.
- Quantization notebook `Quantize-Llama-2-7B-python-coder-GPTQ.ipynb`: In this notebook we quantize the model with GPTQ.
- Finetuning TinyLlama 1.1B with Axolot in notebook `Finetuning_TinyLlama_with_Axolot.pynb`.

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