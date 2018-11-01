#!/usr/bin/env bash

cut -f 3 gold_singular_swap.tsv > gold_singular_swap.src
cut -f 4 gold_singular_swap.tsv > gold_singular_swap.tgt

cut -f 3 elmo_singular_swap.tsv > elmo_singular_swap.src
cut -f 4 elmo_singular_swap.tsv > elmo_singular_swap.tgt

allennlp elmo --top \
    --options-file ../data/elmo_2x4096_512_2048cnn_2xhighway_options.json \
    --weight-file ../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
    --batch-size 20 \
    --cuda-device 0 \
    gold_singular_swap.src gold_singular_swap.src.h5py

allennlp elmo --top \
    --options-file ../data/elmo_2x4096_512_2048cnn_2xhighway_options.json \
    --weight-file ../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
    --batch-size 20 \
    --cuda-device 0 \
    gold_singular_swap.tgt gold_singular_swap.tgt.h5py

allennlp elmo --top \
    --options-file ../data/elmo_2x4096_512_2048cnn_2xhighway_options.json \
    --weight-file ../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
    --batch-size 20 \
    --cuda-device 0 \
    elmo_singular_swap.src elmo_singular_swap.src.h5py

allennlp elmo --top \
    --options-file ../data/elmo_2x4096_512_2048cnn_2xhighway_options.json \
    --weight-file ../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
    --batch-size 20 \
    --cuda-device 0 \
    elmo_singular_swap.tgt elmo_singular_swap.tgt.h5py