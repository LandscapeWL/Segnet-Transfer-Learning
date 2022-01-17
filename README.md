# Segnet-Transfer-Learning
This project is based on the Python language, the Keras deep learning framework, migration learning techniques and the tensorboard module to build a SegNet neural network, documenting the training process.

![](http://r.photo.store.qq.com/psc?/V51wK6B50SnpHF0Ql90V120XkX2YMvAu/TmEUgtj9EK6.7V8ajmQrEJ9AjC6X2pH4kgCjg9q34.uUJ5OfEqUmMCBS.*keM*QqiKWPFMLk1y6SUMjCfBhwyGlcVkXGg0KyJ1z27KR5kDw!/r "")

# Installation
tensorflow-gpu==1.13.1

keras==2.1.5

# Run project
You can use our trained deep learning models for semantic segmentation prediction of city street images.

1.Place your prepared street view images in the ***img*** folder.

2.Change the image read path in ***<predict.py>***.

3.Run ***<predict.py>*** for semantic segmentation of street view image.

4.The results are saved in the ***imgout*** folder.

# Download the cityscapes dataset
The Cityscapes is an open data set that can be downloaded from the official website.
(https://www.cityscapes-dataset.com/)

# Train your own neural network

1.Prepare the cityscapes training dataset and modify the path to the dataset in the train file.

2.Open the ***<train.py>*** and change the training parameters, which I have as follows:

```
batch_size = 2

Transfer training:

epoch = 20

optimizer = Adam(lr=1e-3)

Global training:

epoch = 40

optimizer = Adam(lr=1e-4)
```

3.Run the ***<train.py>***.

4.The output neural network weights will be saved in the ***logs*** folder.

# Neural Network Accuracy

Use tensorboard record the loss function and accuracy of the training process.

Transfer training stage stored in ***Tensorboard1*** folder.

Global training stage stored in ***Tensorboard2*** folder.

Use command line ```cmd``` and ```tensorboard --logidr=Tensorboard1``` ```tensorboard --logidr=Tensorboard2``` get url to open chrome visualisation of the training process.

Following is an example of a visual file of our team training process:
![](url'')
