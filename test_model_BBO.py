#from keras.models import load_model
from keras import backend as K
#import matplotlib.pyplot as plt
import editdistance
import tensorflow as tf
import numpy as np

'''
from data_loader import x_test_rand, y_test_rand, x_test_18th,y_test_18th


The following program returns the CER of printed text-line image onl,y and then
 you can follow the same steps for the synthetic images
'''
x_test_rand= np.load('./hist_all_resize/x_test_rand_resize.npy', allow_pickle=True)
#x_test_18th = np.load('./hist_all_resize/x_test_rand_resize.npy', allow_pickle=True)
x_test_18th = np.load('./hist_all_resize/x_test_18th_resize.npy', allow_pickle=True)
y_test_rand = np.load('./hist_all_resize/y_test_rand_resize.npy', allow_pickle=True)
y_test_18th = np.load('./hist_all_resize/y_test_18th_resize.npy', allow_pickle=True)

print(x_test_rand.shape)
print(y_test_rand.shape)
print(x_test_18th.shape)
print(y_test_18th.shape)
model= tf.keras.models.load_model(
    './model/tr12_25.hdf5',
    custom_objects={'Functional':tf.keras.models.Model})

y_pred=model.predict(x_test_rand)

#the CTC decoer
y_decode = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])

for i in range(5):
    # print the first 10 predictions
    print("Prediction :", [j for j in y_decode[i] if j!=-1], " -- Label : ", y_test_rand[i])

#=========== compute editdistance and returne CER ====================================

true=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_test_rand)):
    x=[j for j in y_test_rand[i] if j!=0]
    true.append(x)

pred=[]
# to stor the predicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
for i in range(len(y_decode)):
    x=[j for j in y_decode[i] if j not in(0,-1)]
    pred.append(x)

cer=0
for(i,j) in zip(true,pred):
    x=editdistance.eval(i,j)
    cer=cer+x
err=cer
x=0
for i in range(len(true)):
    x=x+len(true[i])
totalchar=x
cerp=(float(err)/totalchar)*100
print("the CER of random test set:")
print(cerp)

#his= np.load('./model/history_tr12_25.npy', allow_pickle=True)

y_pred_18th=model.predict(x_test_18th)

#the CTC decoer
y_decode_18th = K.get_value(K.ctc_decode(y_pred_18th[:, :, :], input_length=np.ones(y_pred_18th.shape[0]) * y_pred_18th.shape[1])[0][0])

for i in range(3):
    # print the first 10 predictions
    print("Prediction :", [j for j in y_decode_18th[i] if j!=-1], " -- Label : ", y_test_18th[i])

#=========== compute editdistance and returne CER ====================================

true_18=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_test_18th)):
    x=[j for j in y_test_18th[i] if j!=0]
    true_18.append(x)

pred_18=[]
# to stor the pdicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
for i in range(len(y_decode_18th)):
    x=[j for j in y_decode_18th[i] if j not in(0,-1)]
    pred_18.append(x)

cer_18=0
for(i,j) in zip(true_18,pred_18):
    x=editdistance.eval(i,j)
    cer_18=cer_18+x

err_18=cer_18

x_18=0
for i in range(len(true_18)):
    x_18=x_18+len(true_18[i])

totalchar_18=x_18

cerp_18=(float(err_18)/totalchar_18)*100
print("The CER of 18th century book is:")
print(cerp_18)

#print("history of the un_optimized trained model")
#print(his)

print("===================================================================================================")
print( "================================================================================================")

model_opt= tf.keras.models.load_model(
    './model/tr1_25_BBO_615k.hdf5',
    custom_objects={'Functional':tf.keras.models.Model})

his_opt= np.load('./model/tr1_history1_25_BBO_615k.npy', allow_pickle=True)
y_pred_opt=model_opt.predict(x_test_rand)

#the CTC decoer
y_decode_opt = K.get_value(K.ctc_decode(y_pred_opt[:, :, :], input_length=np.ones(y_pred_opt.shape[0]) * y_pred_opt.shape[1])[0][0])

for i in range(5):
    # print the first 10 predictions
    print("Prediction :", [j for j in y_decode_opt[i] if j!=-1], " -- Label : ", y_test_rand[i])

#=========== compute editdistance and returne CER ====================================

true_opt=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_test_rand)):
    x=[j for j in y_test_rand[i] if j!=0]
    true_opt.append(x)

pred_opt=[]
# to stor the predicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
for i in range(len(y_decode_opt)):
    x=[j for j in y_decode_opt[i] if j not in(0,-1)]
    pred_opt.append(x)

cer_opt=0
for(i,j) in zip(true_opt,pred_opt):
    x=editdistance.eval(i,j)
    cer_opt=cer_opt+x
err_opt=cer_opt

x_opt=0
for i in range(len(true_opt)):
    x_opt=x_opt+len(true_opt[i])
totalchar_opt=x_opt
cerp_opt=(float(err_opt)/totalchar_opt)*100
print("the CER of random test set_optimized_6375:")
print(cerp_opt)




y_pred_18th_opt=model_opt.predict(x_test_18th)

#the CTC decoer
y_decode_18th_opt = K.get_value(K.ctc_decode(y_pred_18th_opt[:, :, :], input_length=np.ones(y_pred_18th_opt.shape[0]) * y_pred_18th_opt.shape[1])[0][0])

for i in range(3):
    # print the first 10 predictions
    print("Prediction :", [j for j in y_decode_18th_opt[i] if j!=-1], " -- Label : ", y_test_18th[i])

#=========== compute editdistance and returne CER ====================================

true_18_opt=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_test_18th)):
    x=[j for j in y_test_18th[i] if j!=0]
    true_18_opt.append(x)

pred_18_opt=[]
# to stor the pdicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
for i in range(len(y_decode_18th_opt)):
    x=[j for j in y_decode_18th_opt[i] if j not in(0,-1)]
    pred_18_opt.append(x)

cer_18_opt=0
for(i,j) in zip(true_18_opt,pred_18_opt):
    x=editdistance.eval(i,j)
    cer_18_opt=cer_18_opt+x

err_18_opt=cer_18_opt

x_18_opt=0
for i in range(len(true_18_opt)):
    x_18_opt=x_18_opt +len(true_18_opt[i])

totalchar_18_opt=x_18_opt

cerp_18_opt=(float(err_18_opt)/totalchar_18_opt)*100
print("The CER of 18th century book is optimized:")
print(cerp_18_opt)

print("history of the optimized trained model_15k")
print(his_opt)