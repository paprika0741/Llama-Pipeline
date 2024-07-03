import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import os
import torch.distributed as dist
from src.llamapipe.config import LlamaConfig
from src.llamapipe.llama_pipe_model import StageModel
from transformers import LlamaTokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
import random
import numpy as np
import torch
import os
def  initialize_distributed(config):
    print("Initializing process group...")
    torch.distributed.init_process_group(
        backend=config.distributed_backend,
        init_method=config.init_method,
        world_size=args.world,
        rank=args.rank,
    )
    print("Initialization of process group complete!")
def parse_args():
    parser = argparse.ArgumentParser(description='rank: device id')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world', default=2, type=int)
    parser.add_argument('--config_file',default="./src/llamapipe/config.json",type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config =   LlamaConfig.from_pretrained(args.config_file )
    initialize_distributed(config)
    config.update_pp_stage_config(args)
    # config.print_config()
    tokenizer = LlamaTokenizer.from_pretrained(config.model_dir)
    model = StageModel(config).to("cuda")
    input_text =  "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three parts."
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to("cuda")
    bs,seq_len = inputs["input_ids"].shape
    past_key_values = None

    print("prefiling")
    if  config.is_first_stage :
        outputs = model (  input_ids=  inputs["input_ids"])
    else:
        recv_tensor = torch.zeros( bs,  seq_len,  config.hidden_size, dtype=torch.float16)
        dist.recv(tensor=recv_tensor, src= config.pre_rank) 
        hidden_states = recv_tensor.to("cuda")
        outputs = model   (input_ids=None, inputs_embeds=hidden_states) 

    if not config.is_last_stage:
        assert isinstance(outputs, BaseModelOutputWithPast)
        hidden_states = outputs.last_hidden_state
        # send to next stage
        send_tensor = hidden_states.cpu()
        dist.send(tensor= send_tensor, dst= config.next_rank)
    else:
        assert isinstance(outputs, CausalLMOutputWithPast)
        logits = outputs.logits
        next_tokens = model.decode_next_token(logits)
        print(model.tokenizer.decode(next_tokens.cpu()[0], skip_special_tokens=True))
    past_key_values = outputs.past_key_values
    print("decoding")
    for idx in range(  10):
        if config.is_last_stage:
            send_tensor = next_tokens.cpu()
            dist.send(tensor= send_tensor, dst=  0)
        if config.is_first_stage:
            recv_tensor = torch.zeros( 1, dtype=torch.int64) # int
            dist.recv(tensor=recv_tensor, src= config.total_stage-1) 
            input_ids = recv_tensor.to("cuda")
            input_ids = input_ids.unsqueeze(1)
        if  config.is_first_stage :
            outputs = model (  input_ids=input_ids,past_key_values=past_key_values) 
        else:
            recv_tensor = torch.zeros( bs,  1,  config.hidden_size, dtype=torch.float16)
            dist.recv(tensor=recv_tensor, src= config.pre_rank) 
            hidden_states = recv_tensor.to("cuda")
            outputs = model   (input_ids=None, inputs_embeds=hidden_states,past_key_values=past_key_values) 
        # update kv cache
        past_key_values = outputs.past_key_values
        # send to next stage
        if not config.is_last_stage:
            assert isinstance(outputs, BaseModelOutputWithPast)
            hidden_states = outputs.last_hidden_state
            send_tensor = hidden_states.cpu()
            dist.send(tensor= send_tensor, dst= config.next_rank)
        else:
            assert isinstance(outputs, CausalLMOutputWithPast)
            logits = outputs.logits
            next_tokens = model.decode_next_token(logits)
            # print(next_tokens.shape)
            print(model.tokenizer.decode(next_tokens.cpu()[0], skip_special_tokens=True))
            if next_tokens is not None:
                if model.tokenizer.eos_token_id in next_tokens :
                    print("\n")
                    print("finish decoding")
                    break
    max_memory = torch.cuda.max_memory_allocated(device=  "cuda")
    print("Max memory:  {} ( {} MB ) ".format( max_memory , max_memory /(1024*1024) ))    
    
