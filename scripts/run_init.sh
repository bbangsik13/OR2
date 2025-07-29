#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3dgstream

ITERATION=15000
DATA=data/n3v
DECAY=100
LR=1e-4
LAMBDA=0.01


GPU=2
for SCENE in {coffee_martini_wo_cam13,cook_spinach,cut_roasted_beef}
do
for i in {00..04}
do
OUTPUT=3dgs_init
LOGS=logs/first_frame
mkdir -p $LOGS
EXP=${SCENE}_${i}
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "python train.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --eval --sh_degree 3 --lambda_dssim 0.2 --use_offset \
 --save_iteration $ITERATION --images images_2 \
 --iterations $ITERATION --densify_until_iter $((ITERATION / 2)) --port $((6754+GPU)) \
 --densify_grad_threshold  0.0002 --position_lr_max_steps $ITERATION \
 --offset_lr_init $LR --offset_lr_final $(echo "$LR" | awk -v decay="$DECAY" '{print $1 / decay}') \
 --lambda_offset $LAMBDA --offset_lr_max_steps $((ITERATION / 1)) --lambda_opacity 0.01 && \
 python render.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --iteration $ITERATION --skip_train  && \
 python metrics.py -m $OUTPUT/$EXP" \
  > $LOGS/${EXP}.out 2>&1
done
done &

GPU=3
for SCENE in {coffee_martini_wo_cam13,cook_spinach,cut_roasted_beef}
do
for i in {05..09}
do
OUTPUT=3dgs_init
LOGS=logs/first_frame
mkdir -p $LOGS
EXP=${SCENE}_${i}
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "python train.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --eval --sh_degree 3 --lambda_dssim 0.2 --use_offset \
 --save_iteration $ITERATION --images images_2 \
 --iterations $ITERATION --densify_until_iter $((ITERATION / 2)) --port $((6754+GPU)) \
 --densify_grad_threshold  0.0002 --position_lr_max_steps $ITERATION \
 --offset_lr_init $LR --offset_lr_final $(echo "$LR" | awk -v decay="$DECAY" '{print $1 / decay}') \
 --lambda_offset $LAMBDA --offset_lr_max_steps $((ITERATION / 1)) --lambda_opacity 0.01 && \
 python render.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --iteration $ITERATION --skip_train  && \
 python metrics.py -m $OUTPUT/$EXP" \
  > $LOGS/${EXP}.out 2>&1
done
done &

GPU=1
for SCENE in {flame_salmon_frag1,flame_steak,sear_steak}
do
for i in {00..04}
do
OUTPUT=3dgs_init
LOGS=logs/first_frame
mkdir -p $LOGS
EXP=${SCENE}_${i}
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "python train.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --eval --sh_degree 3 --lambda_dssim 0.2 --use_offset \
 --save_iteration $ITERATION --images images_2 \
 --iterations $ITERATION --densify_until_iter $((ITERATION / 2)) --port $((6754+GPU)) \
 --densify_grad_threshold  0.0002 --position_lr_max_steps $ITERATION \
 --offset_lr_init $LR --offset_lr_final $(echo "$LR" | awk -v decay="$DECAY" '{print $1 / decay}') \
 --lambda_offset $LAMBDA --offset_lr_max_steps $((ITERATION / 1)) --lambda_opacity 0.01 && \
 python render.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --iteration $ITERATION --skip_train  && \
 python metrics.py -m $OUTPUT/$EXP" \
  > $LOGS/${EXP}.out 2>&1
done
done &

GPU=0
for SCENE in {flame_salmon_frag1,flame_steak,sear_steak}
do
for i in {05..09}
do
OUTPUT=3dgs_init
LOGS=logs/first_frame
mkdir -p $LOGS
EXP=${SCENE}_${i}
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "python train.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --eval --sh_degree 3 --lambda_dssim 0.2 --use_offset \
 --save_iteration $ITERATION --images images_2 \
 --iterations $ITERATION --densify_until_iter $((ITERATION / 2)) --port $((6754+GPU)) \
 --densify_grad_threshold  0.0002 --position_lr_max_steps $ITERATION \
 --offset_lr_init $LR --offset_lr_final $(echo "$LR" | awk -v decay="$DECAY" '{print $1 / decay}') \
 --lambda_offset $LAMBDA --offset_lr_max_steps $((ITERATION / 1)) --lambda_opacity 0.01 && \
 python render.py -s $DATA/$SCENE/frame000000 \
 -m $OUTPUT/$EXP --iteration $ITERATION --skip_train  && \
 python metrics.py -m $OUTPUT/$EXP" \
  > $LOGS/${EXP}.out 2>&1
done
done &