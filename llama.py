#!/usr/bin/env python3
import time
import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "/home/sparkleholic/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# ** Prompt Evaluation**
start_prompt = time.time()
inputs = pipe.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
prompt_tokens = inputs.shape[1]
end_prompt = time.time()

prompt_eval_time = end_prompt - start_prompt
prompt_tokens_per_sec = prompt_tokens / prompt_eval_time if prompt_eval_time > 0 else float("inf")

# ** Generate Evaluation**
start_generate = time.time()
outputs = pipe(
    messages,
    max_new_tokens=256,
    return_full_text=True
)
end_generate = time.time()

# Extract generated text
generated_text = outputs[0]["generated_text"]

# Count tokens safely
try:
    # Convert to string if not already
    text_to_tokenize = str(generated_text)
    tokens = pipe.tokenizer(text_to_tokenize, return_tensors="pt")
    generated_tokens = len(tokens['input_ids'][0])
except Exception as e:
    print(f"Error counting tokens: {e}")
    generated_tokens = 0

generate_eval_time = end_generate - start_generate
generate_tokens_per_sec = generated_tokens / generate_eval_time if generate_eval_time > 0 else float("inf")

# **📌 Print Results**
print(f"\n[Prompt Evaluation] {prompt_tokens} tokens processed in {prompt_eval_time:.3f} sec ({prompt_tokens_per_sec:.2f} tokens/sec)")
print(f"[Generate Evaluation] {generated_tokens} tokens generated in {generate_eval_time:.3f} sec ({generate_tokens_per_sec:.2f} tokens/sec)")
print("\n[Model Output]:")
print(generated_text)