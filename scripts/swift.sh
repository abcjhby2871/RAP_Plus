# after you install the swift package, you can use the swift command to run swift code

VIDEO_SEGMENTS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type dpo \
      --model OpenGVLab/InternVL2-2B \
      --dataset tmp/mmrlhf_v1_video.jsonl  tmp/mmrlhf_v1_image.jsonl \
      --beta 0.2 \
      --rpo_alpha $rpo_alpha \
      --train_type full \
      --deepspeed zero2 \
      --torch_dtype bfloat16 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 32 \
      --learning_rate $learning_rate \
      --freeze_vit true \
      --eval_steps 1000 \
      --save_steps 1000 \
      --save_total_limit 5 \
      --logging_steps 5 \
      --max_length 32768 \
      --output_dir $output_dir \
      --warmup_ratio 0.05 \
      --dataloader_num_workers 4 \
      --report_to wandb 