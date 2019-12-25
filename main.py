# example of semi-supervised gan for mnist with comparison to a standalone classifier
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from matplotlib import pyplot
from keras import backend
from keras.models import load_model
import os 
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

#backend.set_image_dim_ordering('th')
backend.set_image_dim_ordering('tf')


# loading the MNIST dataset - offline.
# Input: Folder - A folder that contains the /mnist_database/
def load_mnist(folder='.'):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data_tr = np.fromfile( folder + "/mnist_database/" + 'train' + '-images.idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data_tr[:nMetaDataBytes].tobytes(), intType )
    data_tr = data_tr[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels_tr = np.fromfile( folder + "/mnist_database/" + 'train' + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]
    
    
    data_te = np.fromfile( folder + "/mnist_database/" + 't10k' + '-images.idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data_te[:nMetaDataBytes].tobytes(), intType )
    data_te = data_te[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels_te = np.fromfile( folder + "/mnist_database/" + 't10k' + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]
    
    return (data_tr,labels_tr),(data_te,labels_te)


# custom activation function
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result
 
# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(28,28,1), n_classes=10):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output layer nodes
    fe = Dense(n_classes)(fe)
    # supervised output
    c_out_layer = Activation('softmax')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    # unsupervised output
    d_out_layer = Lambda(custom_activation)(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return d_model, c_model



# Define the standalone classifier model
def define_standalone_calssifier(in_shape=(28,28,1), n_classes=10):
    # image input
    in_image = Input(shape=in_shape)
    # Conv Layer and downsample
    fe = Conv2D(32, (3,3))(in_image)
    fe = LeakyReLU(alpha=0)(fe)
    fe = MaxPooling2D(pool_size = (2,2))(fe)
   # Conv Layer and downsample
    fe = Conv2D(32, (3,3))(fe)
    fe = LeakyReLU(alpha=0)(fe)
    fe = MaxPooling2D(pool_size = (2,2))(fe)
    # Conv Layer and downsample
    fe = Conv2D(64, (3,3))(fe)
    fe = LeakyReLU(alpha=0)(fe)
    fe = MaxPooling2D(pool_size = (2,2))(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dense(64)(fe)
    fe = LeakyReLU(alpha=0)(fe)
    fe = Dropout(0.5)(fe)
    # output layer nodes
    fe = Dense(n_classes)(fe)
    # supervised output
    c_out_layer = Activation('sigmoid')(fe)
    # define and compile standalone classifier model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return c_model



# Define the standalone generator model
def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model
 
# Define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
 
# Load the data. Envelope function for the load_mnist function
def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = load_mnist()
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5   
    #X = np.transpose(X,[0,3,1,2])
    print(X.shape, trainy.shape)
    return [X, trainy]
 
# select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=100, n_classes=10):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        # get all images for this class
        X_with_class = X[y == i]
        # choose random instances
        ix = randint(0, len(X_with_class), n_per_class)
        # add to list
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)
 
# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
        # choose random instances
    
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    
    # generate class labels
    y = ones((n_samples, 1))


    return [X, labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y
 
# generate samples, save as a plot and save the best weights
def summarize_performance(step, g_model, c_model, latent_dim, dataset,max_acc_c,folder, n_samples=100):
    
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = folder+'\generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # evaluate the classifier model
    X, y = dataset
    _, acc = c_model.evaluate(X, y, verbose=0)
    if acc>max_acc_c:
        print('Best acc so far! best_acc_c = %.3f%%' % (acc * 100))
        max_acc_c = acc
        c_model.save(folder+'\c_model_best.h5')
    else:
        print('Best acc did not improve! best_acc_c = %.3f%%' % (max_acc_c * 100))
        
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    # save the generator model
    filename2 = folder+'\g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    # save the classifier model
    filename3 = folder+'\c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))   
    return max_acc_c   


# evaluate and save the standalone classifier model
'''
def summarize_performance_standalone_c(step, c_model, dataset,max_acc_c,folder, n_samples=6000,n_classes=10):
    # evaluate the classifier model
    if n_samples==-1:
        X, y = dataset
    else:
        X, y = select_supervised_samples(dataset, n_samples, n_classes)
        
    _, acc = c_model.evaluate(X, y, verbose=0)
    if acc>max_acc_c:
        print('Best acc so far! best_acc_c = %.3f%%' % (acc * 100))
        max_acc_c = acc
        c_model.save(folder+'\c_model_best.h5')
    else:
        print('Best acc did not improve! best_acc_c = %.3f%%' % (max_acc_c * 100))
        
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    filename3 = folder+'\c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s' % (filename3))   
    return max_acc_c   
'''
#this is the old training function, no augmentations here
'''
# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim,folder,n_samples=100, n_epochs=20, n_batch=100):
    max_acc_c =0
    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset,n_samples)
    print(X_sup.shape, y_sup.shape)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        if (i+1)%100 ==0:
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            max_acc_c= summarize_performance(i, g_model, c_model, latent_dim, dataset,max_acc_c,folder)
 '''

    
 
#this is the new training function, includes augmentations for the classifier
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim,folder,n_samples=100, n_epochs=20, n_batch=100,with_aug=False):
    max_acc_c =0
    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset,n_samples)
    print(X_sup.shape, y_sup.shape)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    
    if(with_aug):
        train_datagen = ImageDataGenerator(
                rotation_range = 20,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                shear_range=0.2,
                zoom_range=0.2)
    else:
        train_datagen = ImageDataGenerator(
                horizontal_flip=False)  
        
        
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs, get one 
    for i in range(n_epochs):
        print("Epoch {}/{}".format(i,n_epochs))
        flow_iter = train_datagen.flow(X_sup, y_sup,half_batch) #this is an enumerator for random half-batches
        with tqdm(total = bat_per_epo,position = 0, leave=True) as progress_bar:
            for j,[Xsup_real, ysup_real] in enumerate(flow_iter):
                if j>=bat_per_epo:
                    break #the enumerator itself will run forever, stop on expected number of batches
                progress_bar.update(1)
                c_model.fit(Xsup_real, ysup_real,verbose=0)
                # update unsupervised discriminator (d)
                [X_real, _], y_real = generate_real_samples(dataset, half_batch) #generate real, but not labeled samples for the discriminator
                d_model.train_on_batch(X_real, y_real)
                X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
                d_model.train_on_batch(X_fake, y_fake)
                # update generator (g)
                X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
                gan_model.train_on_batch(X_gan, y_gan)



        max_acc_c= summarize_performance(i, g_model, c_model, latent_dim, dataset,max_acc_c,folder)
 
    
    
# train the standalone classifier
def train_stand_alone_c(c_model, dataset, folder, n_epochs=20, n_batch=32, n_samples=100,n_classes=10,with_aug=True):
    max_acc_c = 0
    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset,n_samples,n_classes)
    print(X_sup.shape, y_sup.shape)
    # calculate the number of batches per training epoch
    bat_per_epo = int(X_sup.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs
    
    test_datagen = ImageDataGenerator()
    if(with_aug):
        train_datagen = ImageDataGenerator(
                rotation_range = 20,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                shear_range=0.2,
                zoom_range=0.2)
    else:
        train_datagen = ImageDataGenerator(
                horizontal_flip=False)
    
    train_dgi = train_datagen.flow(X_sup, y_sup,n_batch)
    X_all, y_all = dataset 
    test_dgi = test_datagen.flow(X_all, y_all,64)
    print(test_dgi.n)
    csv_logger = CSVLogger(folder +'/training.log')
    check_point = ModelCheckpoint(folder+"/c_model_best.h5",monitor = "val_acc", verbose =1, save_best_only = True,mode = "max") 
    c_model.fit_generator(train_dgi,
            steps_per_epoch=bat_per_epo,
            epochs=n_epochs,
            validation_data = test_dgi,
            validation_steps = test_dgi.n//(4*64),
            callbacks = [csv_logger, check_point])
            

#evaluate classifier on the entire MNIST dataset. The input folder should contain
#   a "c_model_best.h5" weights file, this file is saved during training 
def evalClassifier(folder):
    # load the model
    model = load_model(folder+'/c_model_best.h5')
    # load the dataset
    (trainX, trainy), (testX, testy) = load_mnist()
    # expand to 3d, e.g. add channels
    trainX = expand_dims(trainX, axis=-1)
    testX = expand_dims(testX, axis=-1)
    # convert from ints to floats
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    # scale from [0,255] to [-1,1]
    trainX = (trainX - 127.5) / 127.5
    testX = (testX - 127.5) / 127.5
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    print('Train Accuracy: %.3f%%' % (train_acc * 100))
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Test Accuracy: %.3f%%' % (test_acc * 100))



############################# configuration params ############################
# size of the latent space
latent_dim = 100
#number of classes
n_classes=10
#number of labled samples
#arr_n_samples = [50,100,150,200,500,700,1000,2000,3000,4000,5000]
arr_n_samples = [7000]
 #batch sizes (for standalone classifier)
#n_batches = [8,8,8,16,16,32,32,32,32,64,64] 
n_batches = [16] 

stand_alone_flag = False #Change this for SGAN/STANDALONE
augmentation_flag = True #Change this for AUG/NO_AUG
###############################################################################


if stand_alone_flag:
    for i in range(np.size(arr_n_samples ,0)):
        if augmentation_flag:
            folder = "testStandAlone_with_aug"+str(arr_n_samples[i])
        else:
            folder = "testStandAlone_without_aug"+str(arr_n_samples[i])
        if not os.path.exists("./"+folder+"/"):
                        os.makedirs("./"+folder+"/")
        
       
        # create the standalone classifier
        c_model = define_standalone_calssifier(n_classes=n_classes)
    
        # load image data
        dataset = load_real_samples()
        # train model
        n_batch = n_batches[i]
        train_stand_alone_c(c_model, dataset, folder, n_samples=arr_n_samples[i],n_epochs=200, n_batch=n_batch,n_classes=n_classes,with_aug=augmentation_flag)

else:   
    for i in range(np.size(arr_n_samples ,0)):
        if augmentation_flag:
            folder = "testSGAN_with_aug"+str(arr_n_samples[i])
        else:
            folder = "testSGAN_without_aug"+str(arr_n_samples[i])
        
        if not os.path.exists("./"+folder+"/"):
                        os.makedirs("./"+folder+"/")
        
        # create the discriminator models
        d_model, c_model = define_discriminator()
        # create the generator
        g_model = define_generator(latent_dim)
        # create the gan
        gan_model = define_gan(g_model, d_model)
        # load image data
        dataset = load_real_samples()
        # train model
        train(g_model, d_model, c_model, gan_model, dataset, latent_dim,folder,n_samples=arr_n_samples[i],with_aug = augmentation_flag)


