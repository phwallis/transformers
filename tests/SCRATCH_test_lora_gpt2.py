import os, json, sys
from pathlib import Path
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelWithLMHead, AutoModelForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# instantiate pretrained model with lora params
model = GPT2LMHeadModel.from_pretrained('gpt2')
for n, p in model.named_parameters():
	print(n)

text = "this is a test"
input_tensor = tokenizer.encode(text, return_tensors='pt')
outputs = model(input_tensor)
print(outputs.logits.shape)

