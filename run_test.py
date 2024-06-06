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
    config.print_config()
    tokenizer = LlamaTokenizer.from_pretrained(config.model_dir)
    model = StageModel(config).to("cuda")
    input_text = "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is #divided into three"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to("cuda")
    bs,seq_len = inputs["input_ids"].shape
    print("model:", model)
    if  config.is_first_stage :
        outputs = model (  input_ids=  inputs["input_ids"])
    else:
        # receive from previous stage
        recv_tensor = torch.zeros( bs,  seq_len,  config.hidden_size, dtype=torch.float16)
        print( "receive from previous stage", config.pre_rank)
        dist.recv(tensor=recv_tensor, src= config.pre_rank) 
        hidden_states = recv_tensor.to("cuda")
        outputs = model   (input_ids=None, inputs_embeds=hidden_states) 
    # send to next stage
    if not config.is_last_stage:
        assert isinstance(outputs, BaseModelOutputWithPast)
        hidden_states = outputs.last_hidden_state
        # send to next stage
        print("send to next stage", config.next_rank)
        send_tensor = hidden_states.cpu()
        dist.send(tensor= send_tensor, dst= config.next_rank)
    else:
        assert isinstance(outputs, CausalLMOutputWithPast)
        logits = outputs.logits
        torch.save(logits, "logits{}.pt".format( str(config.stage_num_hidden_layers_list)))
        print(input_text)
        model.decode_next_token(logits)