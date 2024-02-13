---
tags:
- axolot
- code
- coding
- Tinyllama
- axolot
model-index:
- name: TinyLlama-1431k-python-coder
  results: []
license: apache-2.0
language:
- code
datasets:
- iamtarun/python_code_instructions_18k_alpaca
pipeline_tag: text-generation
---


# TinyLlaMa 1.1B 1431k 4-bit Python Coder 👩‍💻 

**TinyLlaMa 1.1B** fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the **Axolot** library in 4-bit with [PEFT](https://github.com/huggingface/peft) library.

## Pretrained description

[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)

The [TinyLlama project](https://github.com/jzhang38/TinyLlama) aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, they can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀.

They adopted exactly the same architecture and tokenizer as Llama 2. This means TinyLlama can be plugged and played in many open-source projects built upon Llama. Besides, TinyLlama is compact with only 1.1B parameters. This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint.

## Training data

[python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)

The dataset contains problem descriptions and code in python language. This dataset is taken from sahil2801/code_instructions_120k, which adds a prompt column in alpaca style.

### Training hyperparameters

The following `axolot` configuration was used during training:

- load_in_8bit: false
- load_in_4bit: true
- strict: false

- datasets:
    - path: iamtarun/python_code_instructions_18k_alpaca
      type: alpaca
- dataset_prepared_path:
- val_set_size: 0.05
- output_dir: ./qlora-out

- adapter: qlora
- sequence_len: 1096
- sample_packing: true
- pad_to_sequence_len: true
- lora_r: 32
- lora_alpha: 16
- lora_dropout: 0.05
- lora_target_modules:
- lora_target_linear: true
- lora_fan_in_fan_out:
- gradient_accumulation_steps: 1
- micro_batch_size: 1
- num_epochs: 2
- max_steps:
- optimizer: paged_adamw_32bit
- lr_scheduler: cosine
- learning_rate: 0.0002
- train_on_inputs: false
- group_by_length: false
- bf16: false
- fp16: true
- tf32: false
- gradient_checkpointing: true
- logging_steps: 10
- flash_attention: false
- warmup_steps: 10
- weight_decay: 0.0

### Framework versions
- torch=="2.1.2"
- flash-attn=="2.5.0"
- deepspeed=="0.13.1"
- axolotl=="0.4.0"


### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/TinyLlama-1431k-python-coder"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, 
                                             device_map="auto")

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
	title        = { TinyLlama-1431k-python-coder },
	year         = 2024,
	url          = { https://huggingface.co/edumunozsala/TinyLlama-1431k-python-coder },
	publisher    = { Hugging Face }
}
```