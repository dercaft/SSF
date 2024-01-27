source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=0 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10450  \
	pruning/train_reg+rec_auto.py ${VTAB_PATH}/sun397  --dataset sun397 --num-classes 397 --no-aug --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 350 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/sun397/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 7e-5  \
    --rec 0.8   \
    --model-path /mnt/SFT_store/zxy/vit_base_patch16_224_in21k/vtab/sun397/baseline-53.30-95.pth.tar \
    --ratio5-epochs 30 \
    --pin-mem
	# --model-ema --model-ema-decay 0.9  \
