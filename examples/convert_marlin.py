from transformers import AutoTokenizer
import gptqmodel
from gptqmodel import GPTQModel, QuantizeConfig
import shutil
import random
import numpy as np
import torch
import os
import argparse
from datasets import load_dataset
## Install GPTQModel
# https://github.com/ModelCloud/GPTQModel

## usage:
## convert Raw/Uncompressed model to Marlin-compatible format (4-bit GPTQ, 128-group, desc_act=False)
## python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin-Compatible

# run DeepSeek-R1-Distill-Qwen-14B inference using fast marlin kernel
# candle-vllm will repack the model weights to Marlin format
## cargo run --release --features cuda -- --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin-Compatible/ qwen2 --quant gptq --penalty 1.0 --temperature 0.

# If you have Marlin-format model, run it with (--quant marlin)
## cargo run --release --features cuda -- --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin/ qwen2 --quant marlin --penalty 1.0 --temperature 0.

def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


@torch.no_grad()
def calculate_avg_ppl(model, tokenizer):
    from gptqmodel.utils import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    # average ppl
    avg = sum(all) / len(all)

    return avg

def main(args):
    pretrained_model_dir = args.src #path to original model (un-quantized model)
    # saving path, save as gptq (4-bit quantization) model if needed 
    #(you may skip the quantization step if you have GPTQ model)
    quantized_model_dir = args.dst
    quantize_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit (candle-vllm now only support 4-bit quantization for marlin)
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    if pretrained_model_dir[-1] != "/":
        pretrained_model_dir += "/"
    if quantized_model_dir[-1] != "/":
        quantized_model_dir += "/"

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = GPTQModel.load(pretrained_model_dir, quantize_config)
    model.resize_token_embeddings(len(tokenizer))
    traindataset = get_wikitext2(tokenizer, args.samples, args.seqlen)
    # # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(traindataset)
    # save quantized model
    model.save(quantized_model_dir)

    # load quantized model, currently only support cpu or single gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(quantized_model_dir, device=device)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("test is", return_tensors="pt").to(device))[0]))

    print(f"Quantized Model {quantized_model_dir} avg PPL is {calculate_avg_ppl(model, tokenizer)}")
    shutil.copy2(pretrained_model_dir + "/tokenizer.json", quantized_model_dir)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Transform uncompressed safetensors weights to 4-bit marlin-compatible format.")
    parser.add_argument(
        "--src", 
        type=str, 
        required=True, 
        help="Path to the source safetensors file."
    )
    parser.add_argument(
        "--dst", 
        type=str, 
        required=True, 
        help="Path to save the transformed safetensors file."
    )

    parser.add_argument('--samples', default=512, type=int, help="Number of samples for calibration.")
    parser.add_argument('--seqlen', default=1024, type=int, help="Sample sequence length for calibration.")

    args = parser.parse_args()
    if os.path.exists(args.src):
        if not os.path.exists(args.dst):
            os.makedirs(args.dst)
        main(args)
    else:
        print("Source folder not exists: ", args.src)
