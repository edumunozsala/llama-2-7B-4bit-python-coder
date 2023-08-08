from datasets import load_dataset
from random import randrange

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM

from trl import SFTTrainer

from huggingface_hub import login
from dotenv import load_dotenv
import os

import argparse

if __name__ == "__main__":
    #Parse the arguments
    parser = argparse.ArgumentParser()
    # Model "bertin-project/bertin-roberta-base-spanish"
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--split", type=str, default="train[:10%]")
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--trained-model-name", type=str, required=True)
    parser.add_argument('--bf16', action='store_true')
     
    args= parser.parse_args()
   
    # The model that you want to train from the Hugging Face hub
    model_id = args.model_name 
    # The instruction dataset to use
    dataset_name = args.dataset
    # Dataset split
    dataset_split= args.split 
    # Fine-tuned model name
    new_model = args.trained_model_name #"llama-2-7b-int4-python-18k-alpaca"
    # Huggingface repository
    hf_model_repo= args.hf_repo
    # Load the entire model on the GPU 0
    device_map = {"": 0}
    
    ################################################################################
    # bitsandbytes parameters
    ################################################################################
    # Activate 4-bit precision base model loading
    use_4bit = True
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # Activate nested quantization for 4-bit base models (double quantization)
    use_double_nested_quant = False

    ################################################################################
    # QLoRA parameters
    ################################################################################
    # LoRA attention dimension
    lora_r = 64
    # Alpha parameter for LoRA scaling
    lora_alpha = 16
    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    # Output directory where the model predictions and checkpoints will be stored
    output_dir = new_model
    # Number of training epochs
    num_train_epochs = args.epochs #1
    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = args.bf16
    # Batch size per GPU for training
    per_device_train_batch_size = 4
    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 2 # 2
    # Enable gradient checkpointing
    gradient_checkpointing = True
    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3
    # Initial learning rate (AdamW optimizer)
    learning_rate = args.lr #2e-4 #1e-5
    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001
    # Optimizer to use
    optim = "paged_adamw_32bit"
    # Learning rate schedule
    lr_scheduler_type = "cosine" #"constant"
    # Number of training steps (overrides num_train_epochs)
    max_steps = -1
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03
    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = False #True
    # Save checkpoint every X updates steps
    save_steps = 0
    # Log every X updates steps
    logging_steps = 25
    # Disable tqdm progress bars, 
    disable_tqdm=True
    
    ################################################################################
    # SFTTrainerparameters
    ################################################################################
    # Maximum sequence length to use
    # max sequence length for model and packing of the dataset
    max_seq_length = 2048 # None
    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = True #False

    # Load the enviroment variables
    load_dotenv()
    # Login to the Hugging Face Hub
    login(token=os.getenv("HF_HUB_TOKEN"))

    # Load dataset from the hub
    dataset = load_dataset(dataset_name, split=dataset_split)
    # Show dataset size
    print(f"dataset size: {len(dataset)}")
    # Show an example
    print(dataset[randrange(len(dataset))])
    # Format the instruction using a data sample
    def format_instruction(sample):
        return f"""### Instruction:
    Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

    ### Task:
    {sample['instruction']}

    ### Input:
    {sample['input']}

    ### Response:
    {sample['output']}
    """
    # Print an instruction
    print(format_instruction(dataset[randrange(len(dataset))]))
    # Configure BitsandBytes parameters
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_use_double_quant=use_double_nested_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map=device_map)
    model.config.pretraining_tp = 1

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
    )

    # prepare model for training (Not in @maximelabonne)
    #model = prepare_model_for_kbit_training(model)
    #model = get_peft_model(model, peft_config)
    
    # Define the training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        #save_steps=save_steps,
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        #tf32=True,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        #max_steps=max_steps,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        disable_tqdm=disable_tqdm,
        report_to="tensorboard",
        seed=42
    )
    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=packing,
        formatting_func=format_instruction,
        args=args,
    )
    # train
    print("Start Training")
    trainer.train() # there will not be a progress bar since tqdm is disabled
    print("End Training")
    
    # save model
    trainer.save_model()
    print("Model saved")
    
    # Empty VRAM to free up resources
    del model
    del trainer
    import gc
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache() # PyTorch thing
    gc.collect()
    
    # Load the saved model
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,    
    )

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained("merged_model",safe_serialization=True)
    tokenizer.save_pretrained("merged_model")

    # push merged model to the hub
    merged_model.push_to_hub(hf_model_repo)
    tokenizer.push_to_hub(hf_model_repo)
