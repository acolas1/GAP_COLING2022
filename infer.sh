#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cli_gap.py \
        --do_predict \
        --output_dir out/event_type_e_r \
        --train_file ../data/eventNarrative/processed/train/train \
        --predict_file ../data/eventNarrative/processed/test/test \
        --tokenizer_path ../pretrained_LM/bart-base \
        --dataset eventNarrative \
	--entity_entity \
        --entity_relation \
	--type_encoding \
	--max_node_length 60 \
        --predict_batch_size 16 \
        --max_input_length 256 \
        --max_output_length 512 \
        --append_another_bos \
        --num_beams 5 \
	--length_penalty 5 \
        --prefix test_event_type_e_r_
