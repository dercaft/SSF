source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/retrain_lora.py ${VTAB_PATH}/diabetic_retinopathy  --dataset diabetic_retinopathy --num-classes 5 --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 -vb 128 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/diabetic_retinopathy/pruning_retrain_lora \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 5e-3\
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/vtab/diabetic_retinopathy/full-73.60.pth.tar \
    --lora-rank 2  
	# --model-ema --model-ema-decay 0.9  \
