DATA_PATH=data/n3v
GPU=0

for i in $DATA_PATH/*
do
CUDA_VISIBLE_DEVICES=$GPU python scripts/convert.py -s $i/frame000000 --resize
CUDA_VISIBLE_DEVICES=$GPU python scripts/copy_cams.py --source $i/frame000000 --scene $i
CUDA_VISIBLE_DEVICES=$GPU python scripts/convert_frames.py -s $i --resize
done