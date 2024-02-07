source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=1  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10451  \
	train.py ${VTAB_PATH}/sun397  --dataset sun397 --num-classes 397  --no-aug --direct-resize  --model vit_small_patch16_224_in21k  \
    --batch-size 128 --epochs 200 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_small_patch16_224_in21k/vtab/sun397/ssf \
	--amp --tuning-mode ssf --pretrained  \
	--pin-mem \