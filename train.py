from nets.segnet import resnet50_segnet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb

NCLASSES = 19
HEIGHT = 416
WIDTH = 416

#Fill the image with a square
def letterbox_image(image, size, type):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.NEAREST)
    # if (type == "jpg"):
    #     new_image = Image.new('RGB', size, (0, 0, 0))
    # elif (type == "png"):
    #     new_image = Image.new('RGB', size, (0, 0, 0))
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh

#Get randomly changed images
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
def get_random_data(image, label, input_shape, jitter=.1, hue=.1, sat=1.1, val=1.1):

    h, w = input_shape

    # resize image
    rand_jit1 = rand(1-jitter,1+jitter)
    rand_jit2 = rand(1-jitter,1+jitter)
    new_ar = w/h * rand_jit1/rand_jit2
    scale = rand(.7, 1.3)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.NEAREST)
    label = label.resize((nw,nh), Image.NEAREST)
    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (0,0,0))
    new_label = Image.new('RGB', (w,h), (0,0,0))
    new_image.paste(image, (dx, dy))
    new_label.paste(label, (dx, dy))
    image = new_image
    label = new_label
    # flip image or not
    flip = rand()<.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x)
    return image_data,label

def generate_arrays_from_file(lines,batch_size):
    # Get total length
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # Get a batch_size of data
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # Reading images from files
            # please change to your train path
            img = Image.open(r"E:\202006Segmentaion\cityscapesScripts-master\leftImg8bit\train" + '/' + name)
            img, _, _ = letterbox_image(img, (WIDTH, HEIGHT), "jpg")

            # img = img.resize((WIDTH,HEIGHT))
            # img = np.array(img)
            # img = img/255
            # X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # Reading images from files
            # please change to your train path
            label = Image.open(r"E:\202006Segmentaion\cityscapesScripts-master\gtFine\train" + '/' + name)
            label, _, _ = letterbox_image(label, (HEIGHT, WIDTH), "png")

            img, label = get_random_data(img, label, [WIDTH, HEIGHT])

            X_train.append(img)

            label = label.resize((int(WIDTH/2), int(HEIGHT/2)))
            label = np.array(label)
            seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))

            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (label[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            # Read a cycle and start again
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

#Validation Set Generator
def generate_arrays_from_file2(lines2,batch_size):
    # Get total length
    n = len(lines2)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # Get a batch_size of data
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines2)
            name = lines2[i].split(';')[0]
            # Reading images from files
            # please change to your val path
            img = Image.open(r"E:\202006Segmentaion\cityscapesScripts-master\leftImg8bit\val" + '/' + name)
            img, _, _ = letterbox_image(img, (WIDTH, HEIGHT), "jpg")

            # img = img.resize((WIDTH,HEIGHT))
            # img = np.array(img)
            # img = img/255
            # X_train.append(img)

            name = (lines2[i].split(';')[1]).replace("\n", "")
            # Reading images from files
            # please change to your val path
            label = Image.open(r"E:\202006Segmentaion\cityscapesScripts-master\gtFine\val" + '/' + name)
            label, _, _ = letterbox_image(label, (HEIGHT, WIDTH), "png")

            img, label = get_random_data(img, label, [WIDTH, HEIGHT])

            # img = img.resize((WIDTH, HEIGHT))
            # img = np.array(img)
            # img = img / 255
            X_train.append(img)

            label = label.resize((int(WIDTH/2),int(HEIGHT/2)))
            label = np.array(label)
            seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (label[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            # Read a cycle and start again
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true,y_pred)
    return loss

if __name__ == "__main__":
    log_dir = "logs/"
    # get model
    model = resnet50_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)

    # pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    # weights_path = keras.utils.get_file( pretrained_url.split("/")[-1] , pretrained_url  )
    weights_path = r"model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model.load_weights(weights_path,by_name=True)

    # model.summary()
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)

    # Open the txt of the dataset
    with open(r"E:\202006Segmentaion\1.make_dataset\read_data\train_data0.txt", "r") as f:
        lines = f.readlines()

    # Open the txt of the dataset
    with open(r"E:\202006Segmentaion\1.make_dataset\read_data\val_data0.txt", "r") as f:
        lines2 = f.readlines()

    # Break up the rows, this txt is mainly used to help read the data for training
    # Disrupted data is better for training
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.shuffle(lines2)
    np.random.seed(None)

    # 90% for training and 10% for estimation.
    # num_val = int(len(lines)*0.1)
    # num_train = len(lines) - num_val
    lines_len = int(len(lines))
    lines2_len = int(len(lines2))

    # The way it is saved, once in 3 generations
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc',
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=1
                                )
    # The way the learning rate drops, acc three times without dropping then drop the learning rate and continue training
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc',
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # Whether or not to stop early, when val_loss doesn't drop means the model is basically trained and can be stopped
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    #Neural network before layer 142 for freezing
    trainable_layer = 142
    for i in range(trainable_layer):
        model.layers[i].trainable = False
    print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))
    
    # Cross-entropy
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(lines_len, lines2_len, batch_size))

    # add tensorboard modules
    Tensorboard1 = TensorBoard(log_dir="./Tensorboard1", histogram_freq=0)

    # Start training
    model.fit_generator(generate_arrays_from_file(lines, batch_size),
            steps_per_epoch=max(1, lines_len//batch_size),

            validation_data=generate_arrays_from_file2(lines2, batch_size),
            validation_steps=max(1, lines2_len//batch_size),
            epochs=20,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr,early_stopping,Tensorboard1])
    model.save_weights(log_dir+'middle1.h5')

    #============================================================================
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    # Cross-entropy
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-4),
            metrics = ['accuracy'])

    # add tensorboard modules
    Tensorboard2 = TensorBoard(log_dir="./Tensorboard2", histogram_freq=0)

    # Start training
    model.fit_generator(generate_arrays_from_file(lines, batch_size),
            steps_per_epoch=max(1, lines_len//batch_size),

            validation_data=generate_arrays_from_file2(lines2, batch_size),
            validation_steps=max(1, lines2_len//batch_size),
            epochs=50,
            initial_epoch=20,
            callbacks=[checkpoint_period, reduce_lr,early_stopping,Tensorboard2])

    model.save_weights(log_dir+'last1.h5')
