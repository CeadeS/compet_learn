#/bin/bash

NCCL_IB_DISABLE=1 python train.py --dataset=mnist --gpu-id=6 --log-filename="mnist_regularization_pretrained" --config="experiments/regularization_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=mnist --gpu-id=6 --log-filename="mnist_classification_pretrained" --config="experiments/classification_pretrained.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cifar10 --gpu-id=6 --log-filename="cifar10_regularization_pretrained" --config="experiments/regularization_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cifar10 --gpu-id=6 --log-filename="cifar10_classification_pretrained" --config="experiments/classification_pretrained.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cifar100 --gpu-id=6 --log-filename="cifar100_regularization_pretrained" --config="experiments/regularization_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cifar100 --gpu-id=6 --log-filename="cifar100_classification_pretrained" --config="experiments/classification_pretrained.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_regularization_pretrained" --config="experiments/regularization_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_classification_pretrained" --config="experiments/classification_pretrained.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_regularization_pretrained" --config="experiments/regularization_pretrained.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_classification_pretrained" --config="experiments/classification_pretrained.json"

NCCL_IB_DISABLE=1 python train.py --dataset=mnist --gpu-id=6 --log-filename="mnist_regularization" --config="experiments/regularization.json"
NCCL_IB_DISABLE=1 python train.py --dataset=mnist --gpu-id=6 --log-filename="mnist_classification" --config="experiments/classification.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cifar10 --gpu-id=6 --log-filename="cifar10_regularization" --config="experiments/regularization.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cifar10 --gpu-id=6 --log-filename="cifar10_classification" --config="experiments/classification.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cifar100 --gpu-id=6 --log-filename="cifar100_regularization" --config="experiments/regularization.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cifar100 --gpu-id=6 --log-filename="cifar100_classification" --config="experiments/classification.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_regularization" --config="experiments/regularization.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cub --gpu-id=6 --log-filename="cub_classification" --config="experiments/classification.json"

NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_regularization" --config="experiments/regularization.json"
NCCL_IB_DISABLE=1 python train.py --dataset=cars --gpu-id=6 --log-filename="cars_classification" --config="experiments/classification.json"

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


