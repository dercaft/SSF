source DATA_PATH.sh
# get the path of the current script

FILE_PATH=$(cd "$(dirname "$0")"; pwd)
EPOCH=1
CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=$2  --master_port=$3  \
	pruning/train_reg_retrain_conv.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 256 --epochs $EPOCH \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 1 --cooldown-epochs 1 \
    --lr 1e-2 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning_retrain_${EPOCH}_only_scale \
	--amp --tuning-mode ssfmerge --pretrained --seed 1  \
	--reg 1e-5\
    # --no-save
	# --model-ema --model-ema-decay 0.9  \
