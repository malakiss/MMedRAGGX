cd ./train/open_clip/src || exit

CUDA_VISIBLE_DEVICES=4 python ./retrieve_clip_report.py \
    --img_root /home/m.ismail/MMed-RAG/iu_xray/iu_xray/images \
    --train_json /home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json \
    --eval_json /home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_val_iu.json \
    --model_name_or_path hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --checkpoint_path /home/m.ismail/MMed-RAG/train/open_clip/src/logs/2025_11_20-20_37_06-model_hf-hub:thaottn-OpenCLIP-resnet50-CC12M-lr_0.0001-b_64-j_2-p_amp/checkpoints/epoch_360.pt \
    --output_path /home/m.ismail/MMed-RAG/output_rag.json \
    --eval_type "test" \
    --fixed_k 10 \
    # --clip_threshold 1.5 \

