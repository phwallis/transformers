import os, json, sys
from pathlib import Path
import numpy as np
from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# instantiate pretrained model with lora params
model = AutoModelForCausalLM.from_pretrained('gpt2')
print('CHECKING FOR LoRA MODEL PARAMS')
for n, p in model.named_parameters():
    if 'adapter' in n:
	    print(n, p.shape)

text = 'this is a test'

print('testing model forward...')
print(f'input text: {text}')
input_tensor = tokenizer.encode(text, return_tensors='pt')
outputs = model(input_tensor)
print(f'logits output shape (should be [1, seq_len, vocab_size]): {outputs.logits.shape}')

