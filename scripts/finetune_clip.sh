export CUDA_VISIBLE_DEVICES="0"

cd ./train/open_clip/src

# harvard dataset
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH torchrun --nproc_per_node=1 --master_port=12347 -m training.main \
    --model hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --train-data /home/m.ismail/MMed-RAG/data/training/retriever/radiology/radiology_train.json \
    --dataset-type IUXray \
    --img_root /home/m.ismail/MMed-RAG/iu_xray/iu_xray/images \
    --batch-size 64 \
    --precision amp \
    --workers 2 \
    --lr 0.0001 \
    --epochs 360 \
    --val-data /home/m.ismail/MMed-RAG/data/training/retriever/radiology/radiology_val.json \
    --val-frequency 10 \
    --report-to tensorboard \

