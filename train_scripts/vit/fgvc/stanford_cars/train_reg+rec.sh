
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=6  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10456  \
	pruning/train_reg+rec.py ${FGVC_PATH}/cars --dataset stanford_cars --num-classes 196 --val-split val  --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/fgvc/stanford_cars/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 4.25e-4  \
    --rec 0.9   \
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/fgvc/stanford_cars/cars_baseline_94.60_283.pth.tar
	# --model-ema --model-ema-decay 0.9  \
