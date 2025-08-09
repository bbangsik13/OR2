source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3dgstream

DATA=data/n3v

INIT_PATH=3dgs_init_best
LR=5e-5
DECAY=100
LAMBDA=0.01

GPU=0
for SCENE in {coffee_martini_wo_cam13,cook_spinach,cut_roasted_beef,flame_salmon_frag1,flame_steak,sear_steak}
do
OUTPUT=outputs
LOGS=logs/sequential
if [ -f $OUTPUT/$SCENE/results.json ]
then
continue
fi
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=${GPU} nohup bash -c "python train_frames.py \
 --read_config --config_path configs/cfg_args.json \
 -o $OUTPUT/$SCENE --sh_degree 3 --use_offset \
 -m $INIT_PATH/$SCENE \
 -v $DATA/$SCENE \
 --image images_2 --first_load_iteration 15_000 \
 --ntc_path ntc/${SCENE}_ntc_params_F_4.pth --frame_end 300 --port 6126 \
 --offset_lr_init $LR --offset_lr_final $(echo "$LR" | awk -v decay="$DECAY" '{print $1 / decay}') --lambda_offset $LAMBDA && \
 python metrics_frames.py \
 -m $OUTPUT/$SCENE \
 -i $INIT_PATH/$SCENE \
 -s $DATA/$SCENE \
 -c mask/n3v/$SCENE/images_2/cam00.png && \
 cp $INIT_PATH/$SCENE/test/ours_15000/renders/00000.png $OUTPUT/$SCENE/test/rendering2/ && \
 ffmpeg -y -framerate 30 -pattern_type glob -i \"$OUTPUT/$SCENE/test/rendering2/*.png\" -r 30 $OUTPUT/$SCENE/${SCENE}.mp4" \
 > $LOGS/${SCENE}.out 2>&1
done &
