#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python cli_gap.py \
         --do_train \
         --output_dir out/webnlg_type_e_r_2 \
         --train_file ../data/webnlg/train \
         --predict_file ../data/webnlg/val \
         --model_path ../pretrained_LM/bart-base \
         --tokenizer_path ../pretrained_LM/bart-base \
         --dataset webnlg \
         --entity_entity \
         --entity_relation \
         --type_encoding \
         --max_node_length 50 \
         --train_batch_size 16 \
         --predict_batch_size 16 \
         --max_input_length 256 \
         --max_output_length 128 \
         --append_another_bos \
         --learning_rate 2e-5 \
         --num_train_epochs 40 \
         --warmup_steps 1600 \
         --eval_period 500 \
         --num_beams 5
        
#CUDA_VISIBLE_DEVICES=0,1,2,3 python cli_gap.py \
#        --do_train \
#        --output_dir out/event_type_e_r \
#        --train_file ../data/eventNarrative/processed/train/train \
#        --predict_file ../data/eventNarrative/processed/val/val \
#        --model_path ../pretrained_LM/bart-base \
#        --tokenizer_path ../pretrained_LM/bart-base \
#        --dataset eventNarrative \
#        --entity_entity \
#        --entity_relation \
#        --type_encoding \
#        --max_node_length 60 \
#        --train_batch_size 16 \
#        --predict_batch_size 16 \
#        --max_input_length 256 \
#        --max_output_length 128 \
#        --append_another_bos \
#        --learning_rate 2e-5 \
#        --num_train_epochs 40 \
#        --warmup_steps 1600 \
#        --eval_period 5000 \
#        --num_beams 5 \
#        ==length_penalty 5 \
