python tools/1-rs-blur-dataset-generation/generate_rs_blur_frames.py \
    --dataset_path="./dataset/2-Fastec-Simulated/Train/" \
    --blur_accumulate_frames=260 \
    --blur_accumulate_step=260 \
    --dataset="Fastec"

python tools/1-rs-blur-dataset-generation/generate_rs_blur_frames.py \
    --dataset_path="./dataset/2-Fastec-Simulated/Test/" \
    --blur_accumulate_frames=260 \
    --blur_accumulate_step=260 \
    --dataset="Fastec"
