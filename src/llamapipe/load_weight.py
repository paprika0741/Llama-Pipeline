import torch
import os
import re
def get_pretrained_stae_dict(weight_dir,total_layer_num):
    pretrained_dict =  {}
    all_files = os.listdir(weight_dir)
    weight_file_list =  [os.path.join(weight_dir, f) for f in all_files if f.endswith('.bin')]
    for weigt_file in weight_file_list:
        pretrained_dict1 = torch.load(weigt_file)
        pretrained_dict = {**pretrained_dict, **pretrained_dict1}
    print("len(pretrained_dict):", len(pretrained_dict))
    not_layer_dict = {k: v for k, v in pretrained_dict.items() if  "model.layers" not in k}   
    print("len(not_layer_dict)", len(not_layer_dict))
    layer_dicts = []# 每一层的layer的权重
    print("total_layer_num:", total_layer_num)
    for i in range(total_layer_num):
        layer_name = f'model.layers.{i}'
        layer_dict = {k: v for k, v in pretrained_dict.items() if ".".join( k.split(".")[:3]) == layer_name} #TODO:这里可能不同模型的key不一样
        layer_dicts.append(layer_dict)
    print("len(layer_dicts)", len(layer_dicts))
    print("len(layer_dicts[0])", len(layer_dicts[0]))
    return not_layer_dict,layer_dicts
def get_stage_state_dict( weight_dir,stage_layer_num_list, rank):
    not_layer_dict,layer_dicts = get_pretrained_stae_dict(weight_dir, sum(stage_layer_num_list))
    print("len(not_layer_dict)", len(not_layer_dict))
    print("len(layer_dicts)", len(layer_dicts))
    stage_state_dict = not_layer_dict
    if rank == 0: # 序号是0开始
        left= 0
        right = stage_layer_num_list[0]
        print( "left:", left, "right:", right)
        for i in range( left, right):
            print("i:", i,end=" ")
            stage_state_dict.update(layer_dicts[i])
        print("len(stage_state_dict)", len(stage_state_dict))
        return stage_state_dict
    else:
        print(  "stage_layer_num_list", stage_layer_num_list)
        print("rank", rank)
        left = sum(stage_layer_num_list[:rank ])
        right = sum(stage_layer_num_list[ :rank+1]  )
        print( "left:", left, "right:", right)
        # 取出对应layer范围的权重
        for i in range( left, right):
            print("i:", i,end=" ")
            stage_state_dict.update(layer_dicts[i])
        print("len(stage_state_dict)", len(stage_state_dict))
        # 修改layer编号
        new_dict = {}
        for k,v in stage_state_dict.items():
            match = re.search(r'layers\.(\d+)\.', k) # 原本权重layer.x.
            if match==None: # 不是layer.x的权重，直接拷贝
                new_dict[k] = v
            else:
                old_index = int(match.group(1))
                new_index = old_index - left
                # print(  old_index,  "->", new_index)
                new_name = re.sub(r'layers\.(\d+)\.', f'layers.{new_index}.', k) # layer.old. -> layer.new. 修改layer编号
                new_dict[new_name] = v
        return new_dict