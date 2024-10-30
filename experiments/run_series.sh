#/bin/bash

NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_CompetLearn_unsupervised" --config="experiments/CompetLearn_unsupervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_NCA_CompetLearn_pretrained_semi_supervised" --config="experiments/NCA_CompetLearn_pretrained_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_NCA_CompetLearn_semi_supervised" --config="experiments/NCA_CompetLearn_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_NCA_CompetLearn_unsupervised" --config="experiments/NCA_CompetLearn_unsupervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_NCA_supervised_pretrained" --config="experiments/NCA_supervised_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_ProxyNCA_semi_supervised" --config="experiments/ProxyNCA_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_ProxyNCA_unsupervised" --config="experiments/ProxyNCA_unsupervised.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_CompetLearn_unsupervised" --config="experiments/CompetLearn_unsupervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_NCA_CompetLearn_pretrained_semi_supervised" --config="experiments/NCA_CompetLearn_pretrained_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_NCA_CompetLearn_semi_supervised" --config="experiments/NCA_CompetLearn_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_NCA_CompetLearn_unsupervised" --config="experiments/NCA_CompetLearn_unsupervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_NCA_supervised_pretrained" --config="experiments/NCA_supervised_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_ProxyNCA_semi_supervised" --config="experiments/ProxyNCA_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_ProxyNCA_unsupervised" --config="experiments/ProxyNCA_unsupervised.json"

NCCL_IB_DISABLE=1 python train.py --dataset=sop --gpu-id=6 --log-filename="sop_CompetLearn_unsupervised" --config="experiments/CompetLearn_unsupervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=sop --gpu-id=6 --log-filename="sop_NCA_CompetLearn_pretrained_semi_supervised" --config="experiments/NCA_CompetLearn_pretrained_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=sop --gpu-id=6 --log-filename="sop_NCA_CompetLearn_semi_supervised" --config="experiments/NCA_CompetLearn_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=sop --gpu-id=6 --log-filename="sop_NCA_CompetLearn_unsupervised" --config="experiments/NCA_CompetLearn_unsupervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=sop --gpu-id=6 --log-filename="sop_NCA_supervised_pretrained" --config="experiments/NCA_supervised_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=sop --gpu-id=6 --log-filename="sop_ProxyNCA_semi_supervised" --config="experiments/ProxyNCA_semi_supervised.json"
NCCL_IB_DISABLE=1 python train.py --dataset=sop --gpu-id=6 --log-filename="sop_ProxyNCA_unsupervised" --config="experiments/ProxyNCA_unsupervised.json"


