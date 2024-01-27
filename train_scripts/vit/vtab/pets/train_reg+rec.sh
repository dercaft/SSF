source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10451  \
	pruning/train_reg+rec.py ${VTAB_PATH}/oxford_iiit_pet  --dataset pets --num-classes 37 --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/pets/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-4  \
    --rec 0.3  \
    --model-path /mnt/SFT_store/zxy/vit_base_patch16_224_in21k/vtab/pets/baeline-91.96-155.pth.tar   \
    --ratio5-epochs 10 \
    --pin-mem
	# --model-ema --model-ema-decay 0.9  \
