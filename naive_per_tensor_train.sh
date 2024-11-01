#!/bin/bash

# Set the environment variable if needed
export OMP_NUM_THREADS=1

DIR=output/naive_4bit_per_tensor_forward_deit_small/train/seed${SEED}

mkdir -p ${DIR}

# Run the distributed training
python3 ./naive_per_tensor_main_myDDP.py \
--model qt_deit_small_patch16_224 \
--abits 4 \
--wbits 4 \
--gbits 4 \
--qdtype int8 \
--epochs 90 \
--warmup-epochs 0 \
--weight-decay 0.05 \
--batch-size 110 \
--data-path /data/ILSVRC2012 \
--lr 5e-4 
# > ${DIR}/output.log 2>&1 &

    # --distillation-type hard \
    # --teacher-model vit_deit_tiny_distilled_patch16_224 \
    # --opt fusedlamb


# #!/bin/bash

# # Set the environment variable if needed
# export OMP_NUM_THREADS=1

# # Run the distributed training with nohup
# nohup python -m torch.distributed.launch \
#     --master_port=12345 \
#     --nproc_per_node=4 \
#     --use_env main.py \
#     --model fourbits_deit_tiny_patch16_224 \
#     --epochs 300 \
#     --warmup-epochs 0 \
#     --weight-decay 0. \
#     --batch-size 128 \
#     --data-path /mnt/lustre/share/images/ \
#     --lr 3e-4 \
#     --no-repeated-aug \
#     --output_dir ./dist_4bit_tiny_lamb_3e-4_300_512 \
#     --distillation-type hard \
#     --teacher-model vit_deit_tiny_distilled_patch16_224 \
#     --opt fusedlamb > output.log 2>&1 &
