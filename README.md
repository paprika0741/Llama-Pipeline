# Llama-Pipeline

`run_test.py`: pipeline 推理出后续 n 个词

```
 CUDA_VISIBLE_DEVICES=1  python  run_test.py  --rank 0 --world 2 --config_file ./src/llamapipe/config.json
 CUDA_VISIBLE_DEVICES=1  python  run_test.py  --rank 1 --world 2 --config_file ./src/llamapipe/config.json

```

- `config.json`:

  - `stage_num_hidden_layers_list`: the number of hidden layers for each stage of the model.
  - `model_dir`: directory of model checkpoint

## Quantization

`config.json` 参数

- load_in_4bit
- load_in_8bit

split = `[16,16]`

| data type | stage 0 | stage 1 |
| :-------- | :-----: | ------: |
| fp16      | 7294 MB | 7294 MB |
| int8      | 4311 MB | 4311 MB |
| int4      | 3118 MB | 3117 MB |
