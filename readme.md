# Yolo by Jean-Gabriel Simard

Yolo (You only look once, Unifed, Real Time Object Detection) implementated in Python 3 with Tensorflow.
Based on a paper by Joseph Redmon (https://pjreddie.com/media/files/papers/yolo_1.pdf)

## Getting Started


### Prerequisites

Install the following : Python 3, Tensorflow, Opencv

If on linux :
In terminal, in the folder yolo run this
```
chmod u+x download_pascal.sh download_weights
./download_weights
```
If on Windows:
Download the weights with one of the links, extract the file and put the result in the folder and put them network\data\weights folder.
 
Link : https://www.dropbox.com/s/oit8og2t2on0j5t/weights_JG.zip?dl=0 or https://www.dropbox.com/s/7uyw95l2qgebvoe/weights.tar

### Training
On linux, before the first training session, run this command in terminal in the folder yolo
```
./download_pascal.sh
```
The file training.py trains the network.

```
python training.py -gpu
```

"-gpu" is a flag to use the gpu if one is present 



### Inference
Before the first time, only run the weights shell scripts

The file inference.py does inference on images and video. 
```
python inference.py -test_img -gpu
```
* Flags
    * "-gpu"  : use the gpu if one is present 
    * "-test_img" and "-test_video" : run the network on files in the test folder
    * "-file_path FILE_PATH" : give the path to a file and the network will process that file
    * "-save" : saves the results
    * "-h" : gives information on the flags
    

## Results

Cats

![alt text](https://user-images.githubusercontent.com/6108674/32355280-0bff0c7a-c02e-11e7-9de2-d66435a712f7.jpg)

Dog and boy

![alt text](https://user-images.githubusercontent.com/6108674/32355505-5f5b2506-c02f-11e7-95ff-e1c0708c272e.jpg)

Dog and boy

![alt text](https://user-images.githubusercontent.com/6108674/32355508-61e44406-c02f-11e7-957e-c03e413c65d9.jpg)

## Built With

* [Tensorflow](https://www.tensorflow.org/) - An open-source software library for Machine Intelligence

* [OpenCV](https://opencv.org/) - OpenCV is a library of programming functions mainly aimed at real-time computer vision


## Authors

* **Jean-Gabriel Simard**
