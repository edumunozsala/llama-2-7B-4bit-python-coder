---
tags:
- llama-2
- gptq
- quantization
- code
- llama-2
model-index:
- name: Llama-2-7b-4bit-GPTQ-python-coder
  results: []
license: gpl-3.0
language:
- code
datasets:
- iamtarun/python_code_instructions_18k_alpaca
pipeline_tag: text-generation
library_name: transformers
---


# LlaMa 2 7b 4-bit GPTQ Python Coder 👩‍💻 

This model is the **GPTQ Quantization of my Llama 2 7B 4-bit Python Coder**. The base model link is [here](https://huggingface.co/edumunozsala/llama-2-7b-int4-python-code-20k)

The quantization parameters for the GPTQ algo are:
- 4-bit quantization
- Group size is 128
- Dataset C4
- Decreasing activation is False


## Model Description

[Llama 2 7B 4-bit Python Coder](https://huggingface.co/edumunozsala/llama-2-7b-int4-python-code-20k) is a fine-tuned version of the Llama 2 7B model using QLoRa in 4-bit with [PEFT](https://github.com/huggingface/peft) library and bitsandbytes.


## Quantization

A quick definition extracted from a great article in Medium by Benjamin Marie ["GPTQ or bitsandbytes: Which Quantization Method to Use for LLMs — Examples with Llama 2"](https://medium.com/towards-data-science/gptq-or-bitsandbytes-which-quantization-method-to-use-for-llms-examples-with-llama-2-f79bc03046dc) (Only for Medium subscribers)

*"GPTQ (Frantar et al., 2023) was first applied to models ready to deploy. In other words, once the model is fully fine-tuned, GPTQ will be applied to reduce its size. GPTQ can lower the weight precision to 4-bit or 3-bit. 
In practice, GPTQ is mainly used for 4-bit quantization. 3-bit has been shown very unstable (Dettmers and Zettlemoyer, 2023). It quantizes without loading the entire model into memory. Instead, GPTQ loads and quantizes the LLM module by module. 
Quantization also requires a small sample of data for calibration which can take more than one hour on a consumer GPU."*



### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/llama-2-7b-int4-GPTQ-python-code-20k"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

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
outputs = model.generate(input_ids=input_ids, max_new_tokens=128, do_sample=True, top_p=0.9,temperature=0.3)

print(f"Prompt:\n{prompt}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

```

### Citation

```
@misc {edumunozsala_2023,
	author       = { {Eduardo Muñoz} },
	title        = { llama-2-7b-int4-GPTQ-python-coder },
	year         = 2023,
	url          = { https://huggingface.co/edumunozsala/llama-2-7b-int4-GPTQ-python-code-20k },
	publisher    = { Hugging Face }
}
```