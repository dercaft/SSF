
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	train.py ${VTAB_PATH}/eurosat  --dataset eurosat --num-classes 10  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 250 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 3e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/eurosat/ssf \
	--amp --tuning-mode ssf --pretrained  
