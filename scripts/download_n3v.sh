DATA_PATH=data # change here to your personal dataset path
mkdir -p $DATA_PATH
WORKDIR_PATH=$(pwd)

cd $DATA_PATH
mkdir -p $DATA_PATH/n3v
cd $DATA_PATH/n3v

for SCENE in {coffee_martini.zip,cook_spinach.zip,cut_roasted_beef.zip,flame_salmon_1_split.z01,flame_salmon_1_split.z02,flame_salmon_1_split.z03,flame_salmon_1_split.zip,flame_steak.zip,sear_steak.zip}
do
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/$SCENE
done

zip -F flame_salmon_1_split.zip --out flame_salmon_1.zip

rm flame_salmon_1_split*

for i in *.zip
do
unzip $i
done

rm *.zip

cd $WORKDIR_PATH

# mkdir data
# ln -r -s $DATA_PATH/n3v data/