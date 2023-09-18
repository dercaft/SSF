
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=0  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10450  \
	trainm.py ${VTAB_PATH}/clevr_count  --dataset clevr_count --num-classes 8  --no-aug  --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/clevr_count/ssf \
	--amp --tuning-mode ssf --pretrained  \