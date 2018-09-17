from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
import numpy as np

import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

#DEFINING PATH
path_in = '/Users/handing/Desktop/tracce_128/'#input path
path_train = path_in + 'train/'#folder with training data
path_validation = path_in + 'validate/'#folder with validation data
path_out = '/Users/handing/Desktop/cnn_out/multiConv/'#output path
path_weights = path_out + 'weights/'#folder for weight model saving

if not os.path.exists(path_out):
    os.makedirs(path_out)
if not os.path.exists(path_weights):
    os.makedirs(path_weights)

#Creating log .txt
log_path = path_out + 'log_book.txt'

log_text=open(log_path, "w")
log_text.write("CONVOLUTIONAL NEURAL NETWORK\n\n")
log_text.close()

#MULTICONV
#input: number of convolutional and pooling layers
def MultiConv(conv_layers, pool_layers,
              conv_size=64, pool_size=2,
              image_shape=128, epochs=40, batch_size=64,
              lr=0.0001, decay = True):

    print("MultiConv: ConvLayers "+str(conv_layer)+", ConvSize "+str(conv_size)+", PoolLayers "+str(pool_layers)+", PoolSize "+str(pool_size)+"\n\n")
    log_text=open(log_path, "a")
    log_text.write("MultiConv: ConvLayers "+str(conv_layer)+", ConvSize "+str(conv_size)+", PoolLayers "+str(pool_layers)+", PoolSize "+str(pool_size)+"\n\n")
    log_text.close()


    #IMAGE PREPROCESSING
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator( #data_format is channel_last, as default
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,#https://en.wikipedia.org/wiki/Shear_mapping
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)


    #MODEL
    model = Sequential()
    model.add(Conv2D(conv_size, (3, 3), activation='relu', input_shape=(image_shape, image_shape, 1)))

    for i in range(conv_layer-1):
        model.add(Conv2D(32, (3, 3), activation='relu'))
        
    for i in range(pool_layer):
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    #optimizer(also RMSprop can be used)
    if decay:
        opt = optimizers.Adam(lr=lr, decay=0.005) #lr standard is .001
    else:
        opt = optimizers.Adam(lr=lr) #lr standard is .001

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    #print model structure
    print(model.summary())

    with open(log_path,'a') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


    #BATCHES GENERATION
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            path_train,  # this is the target directory
            color_mode='grayscale',
            target_size=(image_shape, image_shape),
            batch_size=batch_size,
            class_mode='binary') # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            path_validation,
            color_mode='grayscale',
            target_size=(image_shape, image_shape),
            batch_size=batch_size,
            class_mode='binary')

    # this is a similar generator, for test data
    test_generator = test_datagen.flow_from_directory(
            path_validation,
            color_mode='grayscale',
            target_size=(image_shape, image_shape),
            batch_size=1,
            class_mode='binary',
            shuffle=False)


    #TRAINING
    hist = model.fit_generator(
            train_generator,
            steps_per_epoch=1600 // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=400 // batch_size)
    #saving model parameters
    model.save_weights(str(path_weights)+'weights-MultiConv-ConvLayers'+str(conv_layer)+'-ConvSize'+str(conv_size)+'-PoolLayers'+str(pool_layers)+'-PoolSize'+str(pool_size)+'.h5')


    #EVALUATE
    score, acc = model.evaluate_generator(test_generator)
    print('Test score:', score)
    print('Test accuracy:', acc)

    #compute auc
    true_classes = test_generator.classes
    predictions = model.predict_generator(generator = test_generator)
    auc_keras = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))
    print('first auc score ', auc_keras)

    log_text=open(log_path, "a")
    log_text.write('Test score: '+str(score)+'\n')
    log_text.write('Test accuracy: '+str(acc)+'\n')
    log_text.write('Test AUC: '+str(auc_keras)+'\n')
    log_text.close()


    #PRINT GRAPHICS
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(hist.history['loss'],'r',linewidth=3.0)
    plt.plot(hist.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig(str(path_out)+"loss_curves-MultiConv"+str(conv_layer)+"MultiPool"+str(pool_layer)+"-poolSize"+str(pool_size)+"-lrate "+str(lr)+"-decay"+str(decay)+'.png')

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(hist.history['acc'],'r',linewidth=3.0)
    plt.plot(hist.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Binary Accuracy Curves',fontsize=16)
    plt.savefig(str(path_out)+"accuracy_curves-MultiConv"+str(conv_layer)+"MultiPool"+str(pool_layer)+"-pool"+str(pool_size)+"-lrate "+str(lr)+"-decay"+str(decay)+'.png')
