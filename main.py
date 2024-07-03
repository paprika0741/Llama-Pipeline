# from transformers import AutoTokenizer,LlamaTokenizer 
# from transformers import LlamaForCausalLM
# from transformers import LlamaConfig
import torch
from src.llama.modeling_llama import LlamaForCausalLM
from src.llama.tokenization_llama import LlamaTokenizer
from src.llama.modeling_llama import LlamaConfig
import random
import numpy as np

def set_all_seeds(seed_value):
    """设置所有随机数种子"""
    random.seed(seed_value) # Python random module
    np.random.seed(seed_value) # NumPy
    torch.manual_seed(seed_value) # PyTorch CPU
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_all_seeds(0)


model_dir = "../llama-2-7b/llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
config = LlamaConfig.from_pretrained(model_dir)
load_in_8bit = False

# LlamaForCausalLMPP.from_pretrained(
#                                      pretrained_model_name_or_path= None,
#                                               config=config, 
#                                               state_dict = stage_state_dict, 
#                                               use_safetensors=False ,
#                                               torch_dtype=torch.float16,
#         )
pretrained_dict1 = torch.load("../llama-2-7b/llama-2-7b-chat-hf/pytorch_model-00001-of-00002.bin")
pretrained_dict2 = torch.load("../llama-2-7b/llama-2-7b-chat-hf/pytorch_model-00002-of-00002.bin")
pretrained_dict = {**pretrained_dict1, **pretrained_dict2}
with torch.device("cuda"):
    model =  LlamaForCausalLM.from_pretrained(  pretrained_model_name_or_path= None, 
                                              torch_dtype=torch.float16,
                                              use_safetensors=False, 
                                              config=config, 
                                              state_dict = pretrained_dict)
model.eval()
if not load_in_8bit:
    model.to("cuda")
print(model)
input_text = "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda")
print("inputs shape:", inputs["input_ids"].shape)
outputs = model (  input_ids=  inputs["input_ids"])
logits = outputs.logits
print("logits shape:", logits.shape)
torch.save(logits, "logits_main.pt"  )
pred = model.generate(**inputs, max_length=60, temperature=0.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
max_gpu_memory = torch.cuda.max_memory_allocated("cuda")

print(f"Maximum GPU memory used: {max_gpu_memory / (1024 ** 2):.2f} MB")
####################################################
print("greedy search")
from transformers import (
            AutoTokenizer,
           AutoModelForCausalLM,
           LogitsProcessorList,
           MinLengthLogitsProcessor,
TemperatureLogitsWarper,
          StoppingCriteriaList,
            MaxLengthCriteria,
         )
past_key_values = None

input_text = "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three parts."
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda")
input_ids = inputs["input_ids"]
print("Prefiling...")
outputs = model (  input_ids = input_ids, past_key_values = past_key_values)
logits = outputs.logits
past_key_values = outputs.past_key_values
next_token_logits = logits[:, -1, :]
logits_processor = LogitsProcessorList(
      [
           TemperatureLogitsWarper( 0.1),
         ]
        )
next_tokens_scores =  logits_processor( inputs["input_ids"], next_token_logits)
next_tokens = torch.argmax(next_tokens_scores, dim=-1)
next_tokens = next_tokens.unsqueeze(-1)
input_ids = torch.cat([input_ids, next_tokens], dim=-1)

print("Decoding...")
for idx in range( 10):
    outputs = model (  input_ids= next_tokens , 
                     past_key_values = past_key_values)
    past_key_values = outputs.past_key_values
    logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    next_tokens_scores =  logits_processor(input_ids , next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    next_tokens = next_tokens.unsqueeze(-1)
    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
print(tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
