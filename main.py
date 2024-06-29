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
pred = model.generate(**inputs, max_length=51, temperature=0.1)
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
input_text = "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three parts."
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda")
print("inputs shape:", inputs["input_ids"].shape)
outputs = model (  input_ids=  inputs["input_ids"])
logits = outputs.logits
next_token_logits = logits[:, -1, :]
print("logits shape:", logits.shape)
print("next_token_logits:", next_token_logits.shape)
logits_processor = LogitsProcessorList(
      [
           TemperatureLogitsWarper( 0.1),
         ]
        )
next_tokens_scores =  logits_processor( inputs["input_ids"], next_token_logits)
next_tokens = torch.argmax(next_tokens_scores, dim=-1)
print("next_tokens:", next_tokens.shape)
print(tokenizer.decode(next_tokens.cpu()[0], skip_special_tokens=True))



 