source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=5 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10455  \
	pruning/train_reg+rec22.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-4  \
    --rec 0.5  \
    --model-path /mnt/SFT_store/zxy/vit_base_patch16_224_in21k/vtab/cifar_100/baseline-ssf-71.28.pth.tar \
    --ratio5-epochs 30 \
    --pin-mem
	# --model-ema --model-ema-decay 0.9  \
