source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=3  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10453  \
	pruning/retrain_lora.py ${FGVC_PATH} --dataset cub2011 --num-classes 200 --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 200 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/fgvc/cub2011/pruning_retrain_lora \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-4\
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/fgvc/cub2011/cub_full_83.91_162.pth.tar \
    --lora-rank 12
	# --model-ema --model-ema-decay 0.9  \
