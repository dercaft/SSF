source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=5  python -m torch.distributed.launch --nproc_per_node=1  --master_port=10455 \
    train.py /data/datasets/FGVC/flowers102  --dataset oxford_flowers --num-classes 102 --val-split val --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 200 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--model-ema --model-ema-decay 0.999  \
	--output ${OUTPUT_PATH}/vit_base_patch16_224_in21k/fgvc/oxford_flowers/ssf \
	--amp --tuning-mode ssf --pretrained  