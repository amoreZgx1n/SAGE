set -x

GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-4}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

cd workplace/internVL2/mpo/InternVL/internvl_chat
source /opt/conda/bin/activate

conda activate environment/internvl2
echo "Python path: $(which python)" >> "workplace/shell/train_log.txt"s
which python


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='output/internvl_chat_mpo_v2/internvl2_8b_mpo'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 64
# batch size per gpu: ~4
# gradient accumulation steps: 1
# total batch size: ~256
# epoch: 8
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=0.0.0.0 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  edpo/train/internvl_chat_dpo.py \
  --model_name_or_path "output/merged_model/internvl2_8b" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/adqa_mpo.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --save_steps 100 \
  --save_total_limit 100 \
  --learning_rate 5e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 1024 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  --loss_type sigmoid,bco_pair \
  --sigmoid_loss_weight 0.8 \
  --bco_pair_loss_weight 0.2 \
  --rpo_alpha 1 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
