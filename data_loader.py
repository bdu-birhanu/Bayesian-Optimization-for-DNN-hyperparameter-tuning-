
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
num_class=310
maxlen=46

def load_dataset():
    '''
    This function loads the training and test datset ( we have three differnt test sets)
    and returns the following arguments. you may download the image from 
    http://www.dfki.uni-kl.de/~belay/ and store in the same directory
    train_imagei --> training text-line images
    train_texi -->  Ground truth of training data
    test_imagep----> test set of printed text-line image with a power-geez font 
    test_imagepg----> test set of synthetic text-line image with power-geez font
    test_imagevg----> test set of  synthetic text-line with visual geez font
    test_textp----> Ground truth for printed text-line image with a power-geez font 
    test_textpg----> Ground truth for synthetic text-line image with power-geez font
    test_textvg----> Ground truth for synthetic text-line with Visualgeez font

we recommend you to run this code with full dataset directly if you computer have >=32 GB RAM
Otherwise, you need to write  your own Data-generator code ( will do it soon).
to check how it works you could use the give sample text line image.

   '''
    x_train_real_resize = np.load('./hist_all_resize/x_train_real_resize.npy' ,allow_pickle=True)
    x_train_synth_resize =np.load('./hist_all_resize/x_train_synth_dist.npy',allow_pickle=True)
    y_train_real_resize= np.load('./hist_all_resize/y_train_real_resize.npy', allow_pickle=True)
    y_train_synth_resize = np.load('./hist_all_resize/y_train_synth_dist.npy', allow_pickle=True)

    #x_val=np.load('./hist_all_resize/x_val_all.npy', allow_pickle=True)
    #y_val = np.load('./hist_all_resize/y_val_all.npy', allow_pickle=True)



    return x_train_real_resize, y_train_real_resize, x_train_synth_resize,  y_train_synth_resize #, x_val,y_val
    #return x_train_real_resize,  y_train_real_resize
#the following two functions are employed for test sets and trainsets separetly just for simplcity

def preprocess_train_val_test_data():
    ''' 
    input: a 2D shape text-line image (h,w)
    output:  returns 3D shape image format (h,w,1)

    Plus this function randomly splits the training and validation set
    This function also computes list of length for both training and validation images and GT
      '''
    #training
    x_tr=load_dataset()
    x_train_real_resize=x_tr[0]
    y_train_real_resize = x_tr[1]
    x_train_synth_resize=x_tr[2]
    y_train_synth_resize=x_tr[3]
    x_val=x_tr[4]
    y_val=x_tr[5]



    x_train_all=np.append(x_train_real_resize,x_train_synth_resize,axis=0)
    y_train_all=np.append(y_train_real_resize,y_train_synth_resize,axis=0)

    #x_train, x_val, y_train, y_val = train_test_split(x_train_real_resize, y_train_real_resize, test_size=0.1)
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.1)
    # np.save('./hist_all_resize/x_train_all_rand_wfirstsynth', x_train)
    # np.save('./hist_all_resize/y_train_all_rand_wfirstsynth', y_train)
    # np.save('./hist_all_resize/x_val_all_wfirstsynth', x_val)
    # np.save('./hist_all_resize/y_val_all_wfirstsynth', y_val)

    # reshape the size of the image from 3D to 4D so as to make the input size is similar with it.
    x_train_r = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)  # [samplesize,32,128,1]
    x_val_r = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    y_train = y_train
    y_val = y_val

    nb_train = len(x_train_r)
    nb_val = len(x_val_r)


    x_train_len = np.array([len(x_train_r[i]) + 59 for i in range(nb_train)])
    # the +59 here is just to make the size ((2*46-1=91) then 48+43=91)of the image equal to the input of LSTM
    x_val_len = np.array([len(x_val_r[i]) + 59 for i in range(nb_val)])  # the +59 here is just to make the size of the image equal to the out put of LSTM
    y_train_len = np.array([len(y_train[i]) for i in range(nb_train)])
    y_val_len = np.array([len(y_val[i]) for i in range(nb_val)])


    return x_train_r, y_train, x_train_len, y_train_len, x_val, y_val, x_val_len, y_val_len
'''
all set of text images and GT
'''
train=preprocess_train_val_test_data()
x_train=train[0]
y_train=train[1]
x_train_length=train[2]
y_train_length=train[3]

x_val=train[4]
y_val=train[5]
x_val_length=train[6]
y_val_length=train[7]


print("data_loading is compeletd")
print("===============================")
print(str(len(x_train))+ " train image and "+ str(len(y_train))+" labels")
print(str(len(x_val))+ "valid image and "+ str(len(y_val))+ "labels")

