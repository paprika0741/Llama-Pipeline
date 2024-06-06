# Llama-Pipeline

`run_test.py`: pipeline 推理出下一个词

```
 CUDA_VISIBLE_DEVICES=1  python  run_test.py  --rank 0 --world 2 --config_file ./src/llamapipe/config.json
 CUDA_VISIBLE_DEVICES=1  python  run_test.py  --rank 1 --world 2 --config_file ./src/llamapipe/config.json

```

- `config.json`:

  - `stage_num_hidden_layers_list`: the number of hidden layers for each stage of the model.
  - `model_dir`: directory of model checkpoint
