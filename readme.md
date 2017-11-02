# Yolo by Jean-Gabriel Simard

Yolo (You only look once, Unifed, Real Time Object Detection) implementated in Python 3 with Tensorflow.
Based on a paper by Joseph Redmon (https://pjreddie.com/media/files/papers/yolo_1.pdf)

## Getting Started


### Prerequisites

Download the weights and put them in network\data\weights 
Link : https://www.dropbox.com/s/oit8og2t2on0j5t/weights_JG.zip?dl=0



### Training


```
python training.py --gpu
```

 --gpu is optional



### Inference

```
python inference.py --test_img
```

## Results

Cats

![alt text](https://user-images.githubusercontent.com/6108674/32355280-0bff0c7a-c02e-11e7-9de2-d66435a712f7.jpg)

Dog and boy

![alt text](https://user-images.githubusercontent.com/6108674/32355505-5f5b2506-c02f-11e7-95ff-e1c0708c272e.jpg)

Dog and boy

![alt text](https://user-images.githubusercontent.com/6108674/32355508-61e44406-c02f-11e7-957e-c03e413c65d9.jpg)

## Built With

* [Tensorflow](https://www.tensorflow.org/) - An open-source software library for Machine Intelligence


## Authors

* **Jean-Gabriel Simard**


