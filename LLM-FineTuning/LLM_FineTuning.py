from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch, json, os, gc, subprocess, sys

# ----------------- 1. Replace the model and configure 4-bit quantization -----------------
model_name = "Qwen1.5-7B"
print(f"Lode Model: {model_name}")

# Configure 4-bit quantization to accommodate low video memory
bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config,device_map="auto",trust_remote_code=True,use_cache=False)
model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
print(f"Tokenizer Type: {type(tokenizer).__name__}")
print(f"Model Type: {type(model).__name__}")

# ----------------- 2. Load and format data -----------------
def load_and_format_data(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Constract the prompt format
                formatted_text = (f"### Instruction: {data['instruction']}\n"f"### Input: {data['input']}\n"f"### Response: {data['output']}")
                texts.append(formatted_text)  
            except json.JSONDecodeError as e:
                print(f"Skip Invalid JSON line: {line[:50]}... Error: {e}")
    
    print(f"\n=== Data statistics ===")
    print(f"Total sample size: {len(texts)}")
    if texts:
        print(f"First Sample Length: {len(texts[0])} character")
        print(f"First Sample Preview: {texts[0][:100]}...")
    
    return Dataset.from_dict({"text": texts})

# Lode Data Set
train_dataset = load_and_format_data("train_data.jsonl")

# ----------------- 3. Tokenize function -----------------
def tokenize_function(examples):
    input_ids_list = []; labels_list = []

    for text in examples["text"]:
        # Split prompt/response
        prefix, response = text.split("### Response:", 1)
        prefix_ids = tokenizer(prefix + "### Response:", add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        response_ids = tokenizer(response,add_special_tokens=False,truncation=True,max_length=512)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = prefix_ids + response_ids
        labels = [-100] * len(prefix_ids) + response_ids
        input_ids_list.append(input_ids[:512])
        labels_list.append(labels[:512])

    return {"input_ids": input_ids_list, "labels": labels_list}

# Apply tokenization
tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

# ----------------- 4. Configure LoRA -----------------
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],lora_dropout=0.1,bias="none")

model = get_peft_model(model, lora_config); model.print_trainable_parameters()
model.enable_input_require_grads(); model.config.use_cache = False; model.train()

# ----------------- 5. Configure training parameters -----------------
training_args = TrainingArguments(output_dir="./lora_results",per_device_train_batch_size=1,
    gradient_accumulation_steps=8,num_train_epochs=3,learning_rate=2e-4,logging_steps=20,
    save_steps=200,fp16=False,gradient_checkpointing=True,optim="paged_adamw_8bit",
    warmup_steps=50,logging_dir="./logs",report_to="none",remove_unused_columns=False,save_total_limit=2,)

# ----------------- 6. Create trainer and train -----------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
trainer = Trainer(model=model,args=training_args,train_dataset=tokenized_dataset,data_collator=data_collator,)
trainer.train()

# ----------------- 7. Save the LoRA adapter -----------------
model.save_pretrained("./my_custom_lora_adapter")
tokenizer.save_pretrained("./my_custom_lora_adapter")
print("LoRA adapter saved to ./my_custom_lora_adapter")

# ----------------- 8. Combine LoRA weight -----------------
# Clear the memory
del model, trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Reload the base model for merging
base_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,
    device_map=None,low_cpu_mem_usage=False,trust_remote_code=True,)
lora_model = PeftModel.from_pretrained(base_model, "./my_custom_lora_adapter")
merged_model = lora_model.merge_and_unload()

# ----------------- 9. Save the complete model -----------------
merged_model_dir = "./merged_full_model"
print(f"LoRA adapter saved to {merged_model_dir} ...")

if os.path.exists(merged_model_dir):
    import shutil
    shutil.rmtree(merged_model_dir)

merged_model.save_pretrained(merged_model_dir, max_shard_size="2GB")
tokenizer.save_pretrained(merged_model_dir)

# ----------------- 10. Convert to GGUF format(Using llama.cpp) -----------------
def convert_to_gguf_python(hf_model_path, output_name, quant_type="q8_0"):
    convert_script = "./llama.cpp/convert_hf_to_gguf.py"
    
    if not os.path.exists(convert_script):
        print(f"Error: Can't find convert script {convert_script}")
        print("Please use git clone https://github.com/ggerganov/llama.cpp")
        return False

    output_file = f"./{output_name}.{quant_type}.gguf"
    cmd = [sys.executable,convert_script,hf_model_path,"--outtype", quant_type,"--outfile", output_file,]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, errors='ignore')
        print(f"GGUF file saved to: {output_file}")

        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024 / 1024
            print(f"File size: {file_size:.2f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed! Error details:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

# Select the quantification level
quantization_type = "q8_0"

print(f"Prepare to convert the model to {quantization_type} format...")
print(f"Model Catalogue: {merged_model_dir}")

convert_to_gguf_python(hf_model_path=merged_model_dir,output_name="finetuned_model",quant_type=quantization_type)