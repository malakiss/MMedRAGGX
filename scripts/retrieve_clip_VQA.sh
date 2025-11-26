cd ./train/open_clip/src || exit
CUDA_VISIBLE_DEVICES=1 python ./retrieve_clip_VQA.py \
    --img_root /home/m.ismail/MMed-RAG/iu_xray/iu_xray/images \
    --train_json /home/m.ismail/MMed-RAG/data/training/retriever/radiology/radiology_train.json \
    --eval_json /home/m.ismail/MMed-RAG/data/training/retriever/radiology/radiology_val.json \
    --model_name_or_path hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --checkpoint_path /home/m.ismail/MMed-RAG/train/open_clip/src/logs/2025_11_20-20_37_06-model_hf-hub:thaottn-OpenCLIP-resnet50-CC12M-lr_0.0001-b_64-j_2-p_amp/checkpoints/epoch_360.pt \
    --output_path /home/m.ismail/MMed-RAG/output_rag.json \
    --fixed_k 10 \
    # --clip_threshold 1.5 \
    # --eval_type $eval_type


