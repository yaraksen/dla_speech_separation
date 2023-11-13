# Speech separation project
### Aksenov Yaroslav

## Installation guide

Download [model](https://disk.yandex.ru/d/8QZKtK3w6-0row) to the folder ```final_model```

```shell
pip install -r ./requirements.txt
```

## Launching guide

#### Generating data
Create folder ```data/datasets/speech_separation/```, where the dataset will be created using command:
   ```shell
   python src/mixer/generate_ss_data.py
   ```

#### Testing:
   ```shell
   python test.py \
      -c final_model/test_config.json \
      -r final_model/model_best.pth \
      -t TEST_DATASET_PATH \
      -o test_result.json
   ```

#### Training step 1:
   ```shell
   python train.py \
      -c src/train_config_step1.json \
      -wk "YOUR_WANDB_API_KEY"
   ```

#### Training step 2:
   ```shell
   python train.py \
      -c src/train_config_step2.json \
      -wk "YOUR_WANDB_API_KEY" \
      -p "PATH TO THE LATEST CHECKPOINT FROM TRAIN 1"
   ```

#### Training step 3:
   ```shell
   python train.py \
      -c src/train_config_step3.json \
      -wk "YOUR_WANDB_API_KEY" \
      -p "PATH TO THE LATEST CHECKPOINT FROM TRAIN 2"
   ```
