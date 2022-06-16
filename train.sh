#!/bin/bash
lang="Hindi"
workspace="./logs"
feats_base_path="./features/"
mode="train"
csv_path="./annotations/"

CUDA_VISIBLE_DEVICES=0 python main.py \
--feats_base_path=$feats_base_path \
--workspace=$workspace \
--lang=$lang \
--mode=$mode \
--csv_path=$csv_path
