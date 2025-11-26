CUDA_DEVICES='0,1,2,3'
CUDA='0,1,2,3'

# Set CUDA_HOME for DeepSpeed
export CUDA_HOME=/cm/shared/ebtree/software/CUDA/12.4.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ./train/dpo || exit
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES deepspeed --include localhost:$CUDA train_dpo_2stages.py \
    --model_name_or_path /home/m.ismail/MMed-RAG/checkpoints \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path /home/m.ismail/MMed-RAG/data/training/alignment/radiology/radiology_vqa_iu.json \
    --image_folder /home/m.ismail/MMed-RAG/iu_xray/iu_xray/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/m.ismail/MMed-RAG/logs_alignment \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --tf32 False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \



