CUDA_VISIBLE_DEVICES=0  python inference.py \
    --config-dir configs/dinoseg.py \
    --checkpoint-dir output/SEG_1/model_final.pth \
    --inference-dir infer_output/SEG_1