echo "Downloading weights_JG"
wget https://www.dropbox.com/s/7uyw95l2qgebvoe/weights.tar

echo "Extracting weights_JG"
tar xf weights.tar

mkdir -p network/data
mkdir -p network/data/weights

mv weights_JG.data-00000-of-00001 weights_JG.index weights_JG.meta network/data/weights

echo "Done"
