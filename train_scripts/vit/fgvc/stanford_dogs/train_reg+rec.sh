source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=3  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10453 \
    pruning/train_reg+rec_auto.py ${FGVC_PATH}/stanford_dogs --dataset stanford_dogs --num-classes 120 --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2.5e-4 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/fgvc/stanford_dogs/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1.2e-4  \
    --rec 0.95   \
    --model-path /mnt/SFT_store/zxy/vit_base_patch16_224_in21k/fgvc/stanford_dogs/dogs_baseline_89.13_127.pth.tar    \
    --pin-mem \
    --ratio5-epochs 20
	# --model-ema --model-ema-decay 0.9  \