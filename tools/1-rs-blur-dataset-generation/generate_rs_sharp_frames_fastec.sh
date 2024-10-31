export PATH="/home/luyunfan/miniconda3/bin/":$PATH

which python

python tools/1-rs-blur-dataset-generation/generate_rs_blur_frames.py \
    --dataset_path="./dataset/2-Fastec-Simulated/Train/" \
    --blur_accumulate_frames=2 \
    --blur_accumulate_step=130 \
    --dataset="Fastec"

python tools/1-rs-blur-dataset-generation/generate_rs_blur_frames.py \
    --dataset_path="./dataset/2-Fastec-Simulated/Test/" \
    --blur_accumulate_frames=2 \
    --blur_accumulate_step=130 \
    --dataset="Fastec"
