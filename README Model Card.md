---
tags:
- generated_from_trainer
- code
- coding
- llama-2
model-index:
- name: Llama-2-7b-4bit-python-coder
  results: []
license: apache-2.0
language:
- code
datasets:
- iamtarun/python_code_instructions_18k_alpaca
pipeline_tag: text-generation
---


# LlaMa 2 7b 4-bit Python Coder 👩‍💻 

**LlaMa-2 7b** fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the method **QLoRA** in 4-bit with [PEFT](https://github.com/huggingface/peft) library.

## Pretrained description

[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b)

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety

## Training data

[python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)

The dataset contains problem descriptions and code in python language. This dataset is taken from sahil2801/code_instructions_120k, which adds a prompt column in alpaca style.

### Training hyperparameters

The following `bitsandbytes` quantization config was used during training:
- load_in_8bit: False
- load_in_4bit: True
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: nf4
- bnb_4bit_use_double_quant: False
- bnb_4bit_compute_dtype: float16

**SFTTrainer arguments**
```py
    # Number of training epochs
    num_train_epochs = 1
    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = True
    # Batch size per GPU for training
    per_device_train_batch_size = 4
    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1
    # Enable gradient checkpointing
    gradient_checkpointing = True
    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3
    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4
    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001
    # Optimizer to use
    optim = "paged_adamw_32bit"
    # Learning rate schedule
    lr_scheduler_type = "cosine" #"constant"
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03
```
### Framework versions
- PEFT 0.4.0

### Training metrics
```
{'loss': 1.044, 'learning_rate': 3.571428571428572e-05, 'epoch': 0.01}
{'loss': 0.8413, 'learning_rate': 7.142857142857143e-05, 'epoch': 0.01}
{'loss': 0.7299, 'learning_rate': 0.00010714285714285715, 'epoch': 0.02}
{'loss': 0.6593, 'learning_rate': 0.00014285714285714287, 'epoch': 0.02}
{'loss': 0.6309, 'learning_rate': 0.0001785714285714286, 'epoch': 0.03}
{'loss': 0.5916, 'learning_rate': 0.00019999757708974043, 'epoch': 0.03}
{'loss': 0.5861, 'learning_rate': 0.00019997032069768138, 'epoch': 0.04}
{'loss': 0.6118, 'learning_rate': 0.0001999127875580558, 'epoch': 0.04}
{'loss': 0.5928, 'learning_rate': 0.00019982499509519857, 'epoch': 0.05}
{'loss': 0.5978, 'learning_rate': 0.00019970696989770335, 'epoch': 0.05}
{'loss': 0.5791, 'learning_rate': 0.0001995587477103701, 'epoch': 0.06}
{'loss': 0.6054, 'learning_rate': 0.00019938037342337933, 'epoch': 0.06}
{'loss': 0.5864, 'learning_rate': 0.00019917190105869708, 'epoch': 0.07}
{'loss': 0.6159, 'learning_rate': 0.0001989333937537136, 'epoch': 0.08}
{'loss': 0.583, 'learning_rate': 0.00019866492374212205, 'epoch': 0.08}
{'loss': 0.6066, 'learning_rate': 0.00019836657233204182, 'epoch': 0.09}
{'loss': 0.5934, 'learning_rate': 0.00019803842988139374, 'epoch': 0.09}
{'loss': 0.5836, 'learning_rate': 0.00019768059577053473, 'epoch': 0.1}
{'loss': 0.6021, 'learning_rate': 0.00019729317837215943, 'epoch': 0.1}
{'loss': 0.5659, 'learning_rate': 0.00019687629501847898, 'epoch': 0.11}
{'loss': 0.5754, 'learning_rate': 0.00019643007196568606, 'epoch': 0.11}
{'loss': 0.5936, 'learning_rate': 0.000195954644355717, 'epoch': 0.12}
```

### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/llama-2-7b-int4-python-code-20k"

tokenizer = AutoTokenizer.from_pretrained(hf_model_repo)

model = AutoModelForCausalLM.from_pretrained(hf_model_repo, load_in_4bit=True, torch_dtype=torch.float16, 
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
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.5)

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