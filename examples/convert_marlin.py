from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig #install latest autogptq
import shutil

#pipeline to marlin format: pretrained model (f16/bf16/f32 format) -> gptq (4-bit quantization) -> gptq marlin

#change the following paths
pretrained_model_dir = "/home/mistral_7b/" #path to original model (un-quantized model)
# saving path, save as gptq (4-bit quantization) model if needed 
#(you may skip the quantization step if you have GPTQ model)
quantized_model_dir = "/home/mistral_7b-int4/" 
save_path = "/home/mistral_7b-int4-Marlin/" # final saving path, save as gptq marlin model

def main():
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit (candle-vllm now only support 4-bit quantization for marlin)
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    
    # # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    # # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    #must specify "use_marlin=True" to save marlin format model
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_marlin=True, use_safetensors=True)
    print(model.state_dict().keys())

    model.save_pretrained(save_path)

    #if everything works fine, the target folder should contain the quantized marlin model called "gptq_model-4bit-128g.safetensors"
    #candle-vllm only support "model.safeternsors" for single-file model or "model.safetensors.index.json" for chunked model
    shutil.move(save_path + "gptq_model-4bit-128g.safetensors", save_path + "model.safetensors")
    #we also need tokenizer.json
    shutil.copy2(pretrained_model_dir + "tokenizer.json", save_path)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
