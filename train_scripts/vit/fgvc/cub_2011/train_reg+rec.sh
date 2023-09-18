
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=1  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=10451  \
	pruning/train_reg+rec.py ${FGVC_PATH} --dataset cub2011 --num-classes 200 --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/fgvc/cub2011/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 2.3e-4  \
    --rec 0.9  \
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/fgvc/cub2011/cub_baseline_89.16_117.tar   
	# --model-ema --model-ema-decay 0.9  \
