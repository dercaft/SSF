source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=4  python -m torch.distributed.launch --nproc_per_node=1  --master_port=10454 \
    pruning/retrain_lora.py /data/datasets/FGVC/flowers102  --dataset oxford_flowers --num-classes 102 --val-split val --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 200 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/fgvc/oxford_flowers/pruning_retrain_lora \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 2e-4\
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/fgvc/oxford_flowers/flowers_full_98.91_177.pth.tar \
    --lora-rank 12
	# --model-ema --model-ema-decay 0.9  \
