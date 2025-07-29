#!/bin/bash
ROOT_DIR=data/n3v
FRAG_DURATION=10
FPS=30

scene="flame_salmon_1"
for mp4_file in $ROOT_DIR/$scene/cam*.mp4; do
    cam_name=$(basename $mp4_file) 
    # for frag in {1..4}; do
    for frag in 1; do
        start_time=$(( (frag - 1) * FRAG_DURATION ))
        num_frame=$(( FRAG_DURATION * FPS ))
        out_dir=$ROOT_DIR/flame_salmon_frag${frag}
        mkdir -p $out_dir
        echo "Splitting $cam_name (frag $frag, start=${start_time}s)"
        ffmpeg -loglevel error -y -ss $start_time -i $mp4_file -vframes $num_frame $out_dir/$cam_name
    done
done
rm -r $ROOT_DIR/$scene

mv $ROOT_DIR/coffee_martini $ROOT_DIR/coffee_martini_wo_cam13

# video to frame
for scene_dir in $ROOT_DIR/*; do
    scene_name=$(basename $scene_dir)
    for mp4_file in $scene_dir/cam*.mp4; do
        cam_name=$(basename $mp4_file .mp4)

        if [[ $scene_name == "coffee_martini" && $cam_name == "cam13" ]]
        then
            echo Skipping $scene_name/$cam_name
            rm $mp4_file
            continue
        fi

        out_dir=$scene_dir/$cam_name
        mkdir -p $out_dir

        echo Extracting $cam_name to $out_dir
        ffmpeg -loglevel error -y -i $mp4_file -start_number 0 $out_dir/frame%06d.png
        rm $mp4_file
    done
done

frame 이동
for scene_dir in $ROOT_DIR/*; do
    scene_name=$(basename $scene_dir)
    for cam_dir in $scene_dir/cam*; do
        cam_name=$(basename $cam_dir)
        echo Moving $scene_name/$cam_name
        for frame_path in $cam_dir/frame*.png; do
            frame_name=$(basename $frame_path)
            frame_id=${frame_name%.*} 
            images_dir=$scene_dir/$frame_id
            mkdir -p $images_dir
            mv $frame_path $images_dir/${cam_name}.png
        done
        rm -r $cam_dir
    done
done

for scene_dir in $ROOT_DIR/*; do
    mkdir -p $scene_dir/frame000000/inputs
    cp $scene_dir/frame000000/*.png $scene_dir/frame000000/inputs/
done