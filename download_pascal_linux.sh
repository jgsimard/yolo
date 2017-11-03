echo "Downloading Pascal VOC 2007"
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

echo "Extracting Pascal VOC 2007"
tar xf VOCtrainval_06-Nov-2007.tar

if [!-d "$data"]; then
	mkdir network/data;
fi
mkdir network/data/pascal_voc/
mv VOCdevkit network/data/pascal_voc/

echo "Done"
