echo "Downloading Pascal VOC 2007"
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

echo "Extracting Pascal VOC 2007"
tar xf VOCtrainval_06-Nov-2007.tar

mkdir -p network/data
mkdir network/data/pascal_voc/

mv VOCdevkit network/data/pascal_voc/

echo "Done"
