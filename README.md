# Position Bias in LLM Summarization

## Pegasus

Main code is in script.py
Command to run script:


```
python script.py --dataset dataset_name --batch_size batch-size --device device# --model model_name

```
- Replace dataset_name with either cnn, xsum, news, reddit 
- Replace batch-size with number
- Replace device# with GPU
- Replace model_name with either the huggingface model name or the local finetuned model name

## BART

Main code is in script.py
Command to run script:


```
python script.py --dataset dataset_name --batch_size batch-size --device device# --model model_name

```
- Replace dataset_name with either cnn, xsum, news, reddit 
- Replace batch-size with number
- Replace device# with GPU
- Replace model_name with either the huggingface model name or the local finetuned model name

## Dolly-v2-7B

Inference the LLM with simply running data_collection.py.

Main code is in script.py
Command to run script:


```
python script.py --dataset dataset_name --batch_size batch-size --device device# 

```
- Replace dataset_name with either cnn, xsum, news, reddit 
- Replace batch-size with number
- Replace device# with GPU


## ChatGPT 3.5-T

Inference the LLM with running inference.ipynb
You will need to add a azure-configuration.json that has the Azure OpenAI endpoints.

Main code is in script.py
Command to run script:

```
python script.py --dataset dataset_name --batch_size batch-size --device device# 

```
- Replace dataset_name with either cnn, xsum, news, reddit 
- Replace batch-size with number
- Replace device# with GPU


## Llama-13B-chat

You will need to download and copy the Llama folder from Meta with the weights to the directory first.
Inference can be done by running the file inference_13b_chat.py with the following command.

```
torchrun --nproc_per_node 2 inference_13b_chat.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 2000 --max_batch_size 4

```

Main code is in script.py
Command to run script:

```
python script.py --dataset dataset_name --batch_size batch-size --device device# 

```
- Replace dataset_name with either cnn, xsum, news, reddit 
- Replace batch-size with number
- Replace device# with GPU


