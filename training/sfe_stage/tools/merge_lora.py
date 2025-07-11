import argparse

import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer

# export PYTHONPATH="${PYTHONPATH}:/DATA/workshop/personal/InternVL-main/internvl_chat/"
# python tools/merge_lora.py 
# /mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/finetuning/InternVL/internvl_chat/output/internvl_chat_v2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full
# /mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/finetuning/InternVL/internvl_chat/output/merged_model/internvl2_8b_v1


argparse = argparse.ArgumentParser()
argparse.add_argument('input_path', type=str, help='Path to the input model')
# /mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/finetuning/InternVL/internvl_chat/output/internvl_chat_v2/internvl2_8b_dynamic_res_2nd_finetune_lora_v2
argparse.add_argument('output_path', type=str, help='Path to the output model')
# /mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/finetuning/InternVL/internvl_chat/output/merged_model/internvl2_8b_v2
args = argparse.parse_args()

print('Loading model...')
model = InternVLChatModel.from_pretrained(
    args.input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

if model.config.use_backbone_lora:
    model.vision_model.merge_and_unload()
    model.vision_model = model.vision_model.model
    model.config.use_backbone_lora = 0
if model.config.use_llm_lora:
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.model
    model.config.use_llm_lora = 0

print('Saving model...')
model.save_pretrained(args.output_path)
print('Saving tokenizer...')
tokenizer.save_pretrained(args.output_path)
print('Done!')
