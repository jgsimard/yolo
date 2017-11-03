echo "Downloading weights_JG"
wget https://www.dropbox.com/s/oit8og2t2on0j5t/weights_JG.zip?dl=0

echo "Extracting weights_JG"
tar unzip weights_JG.zip

if [!-d "$data"]; then
	mkdir network/data;
fi
mkdir network/data/weights_JG
mv weights_JG.data-00000-of-00001 weights_JG.index weights_JG.meta network/data/weights_JG

echo "Done"
