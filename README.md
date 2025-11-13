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

- `load_in_4bit`
- `load_in_8bit`

split = `[16,16]`

最大内存(模型参数内存)
| data type | stage 0 | stage 1 |
| :-------- | :----------------: | ----------------: |
| fp16 | 7046 MB ( 6554 MB) | 7044 MB (6554 MB) |
| int8 | 4061 MB (3466 MB) | 4061 MB (3466 MB) |
| int4 | 2868 MB (1922 MB) | 2867 MB(1922 MB) |

split = `[8,8,8,8]`

| data type | stage 0     | stage 1     | stage 2     | stage 3     |
| --------- | ----------- | ----------- | ----------- | ----------- |
| fp16      | 3788 (3466) | 3536 (3216) | 3536 (3216) | 3786 (3466) |
| int8      | 2292 (1947) | 2041 (1697) | 2041 (1697) | 2292 (1947) |
| int4      | 1761 (1254) | 1507 (1003) | 1507 (1003) | 1759 (1254) |
