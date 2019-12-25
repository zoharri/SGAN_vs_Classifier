import matplotlib.pyplot as plt
import glob 
import numpy as np
from keras.models import load_model
from numpy import expand_dims
from matplotlib import pyplot
import pickle
import os 
from keras import backend
backend.set_image_dim_ordering('tf')

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


def evalClassifier(folder,trainX,trainy,testX,testy):
    # load the model
    model = load_model(folder+'/c_model_best.h5')
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
    return test_acc

#this is for the sort function
def get_fir(val):
    return val[0]
if __name__ == "__main__":
    
    
    graph_SGAN = []
    graph_SA = []
    graph_SANA = []
    graph_SGAN_A = []
     # load the dataset
    (trainX, trainy), (testX, testy) = load_mnist()
    
    dir_names = glob.glob("test*")
    name_of_pickle_file = "acc_for_graph.pkl"
    
    for dir_name in dir_names:
        print(dir_name)
        num_of_reals = int("".join([ch for ch in dir_name if ch.isdigit()]))
        if os.path.exists("./{}/{}".format(dir_name,name_of_pickle_file)):
            with open("./{}/{}".format(dir_name,name_of_pickle_file),"rb") as f:
                te_acc = pickle.load(f)
        else:
            te_acc = evalClassifier(dir_name,trainX,trainy,testX,testy)
            with open("./{}/{}".format(dir_name,name_of_pickle_file),"wb") as f:
                pickle.dump(te_acc,f)
        if("testStandAlone_without_aug" in dir_name):
            graph_SANA.append((num_of_reals,te_acc))
        else:
            if("testStandAlone_with_aug" in dir_name):
                graph_SA.append((num_of_reals,te_acc))
            else:
                if("testSGAN_with_aug" in dir_name):
                    graph_SGAN_A.append((num_of_reals,te_acc))
                else:
                    graph_SGAN.append((num_of_reals,te_acc))
    
    graph_SA.sort(key=get_fir)
    graph_SGAN.sort(key=get_fir)
    graph_SA = np.array(graph_SA)
    graph_SGAN = np.array(graph_SGAN)
    graph_SANA.sort(key = get_fir)
    graph_SANA = np.array(graph_SANA)
    
    graph_SGAN_A.sort(key = get_fir)
    graph_SGAN_A = np.array(graph_SGAN_A)
    
    diff_A = graph_SGAN_A[:,1] - graph_SA[:,1]
    diff_NA = graph_SGAN[:,1] - graph_SANA[:,1]
    
    pyplot.figure()
    plt.title('model accuracy (with augmentation)')
    plt.ylabel('accuracy')
    plt.xlabel('number of supervised samples')
    pyplot.plot(graph_SGAN_A[:,0],graph_SGAN_A[:,1],'-o',c='r')
    pyplot.plot(graph_SA[:,0],graph_SA[:,1],'-s',c='g')
    
    
    plt.legend(['SGAN With Augmentation','Stand Alone Classifier With Augmentation'],loc = "lower right")
    
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor')
    
    plt.savefig("model_acc_with_aug")
    
    
    pyplot.figure()
    plt.title('defference between SGAN and StandAlone Classifier (with augmentation)')
    plt.ylabel('accuracy diff')
    plt.xlabel('number of supervised samples')
    pyplot.plot(graph_SA[:,0],diff_A,'-X',c='b')
    
    
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor')
    
    plt.savefig("diff_with_aug")
    
    
    
    pyplot.figure()
    plt.title('model accuracy (without augmentation)')
    plt.ylabel('accuracy')
    plt.xlabel('number of supervised samples')
    pyplot.plot(graph_SGAN[:,0],graph_SGAN[:,1],'-o',c='r')
    pyplot.plot(graph_SANA[:,0],graph_SANA[:,1],'-s',c='g')
    
    plt.legend(['SGAN','Stand Alone Classifier'],loc = "lower right")
    
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor')
    
    plt.savefig("model_acc_without_aug")
            
            
    pyplot.figure()
    plt.title('defference between SGAN and StandAlone Classifier (without augmentation)')
    plt.ylabel('accuracy diff')
    plt.xlabel('number of supervised samples')
    pyplot.plot(graph_SA[:,0],diff_NA,'-X',c='b')
    
    
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor')
    
    plt.savefig("diff_without_aug")


