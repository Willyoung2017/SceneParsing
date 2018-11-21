TEST_IMG_DIR=./img_data
MODEL_PATH=baseline-resnet101_dilated8-ppm_bilinear_deepsup
RESULT_PATH=./output

ENCODER=$MODEL_PATH/encoder_epoch_25.pth
DECODER=$MODEL_PATH/decoder_epoch_25.pth

if [ ! -e $MODEL_PATH ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
if [ ! -e $TEST_IMG ]; then
  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/images/validation/$TEST_IMG
fi
source activate pytorch
python test_dir.py \
  --model_path $MODEL_PATH \
  --test_img_dir $TEST_IMG_DIR \
  --arch_encoder resnet101_dilated8 \
  --arch_decoder ppm_bilinear_deepsup \
  --fc_dim 2048 \
  --result $RESULT_PATH
