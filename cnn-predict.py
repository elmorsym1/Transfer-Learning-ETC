#!/usr/bin/env python3

# The full CNN code!
####################
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(1)
import time
import numpy as np
from numpy import savetxt
import os
import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras import backend as K  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D, Dense, Flatten
from tensorflow.keras.layers import Dropout, SpatialDropout3D, Activation, BatchNormalization
from tensorflow.keras.layers import Input, concatenate, add
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
# from tensorflow.keras.layers.experimental import preprocessing
from contextlib import redirect_stdout
# from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# Visualization for results
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom



# Model: COM_65248_0.2_adam_MAPE_100_3dir_relu_InsNet_nobias_2B_BSS_KLS_BrSS_ELS_C1_C2_65K_1_USE
# S5 RAW - 200  to 9600  - dataset_S5_len_910_size_100_bal_200_100_9600_len_2055.npz  - MARE_95 = 11.8%
# S5 BAL - 200  to 9600  - dataset_S5_len_910_size_100_bal_200_100_9600_len_3746.npz  - MARE_95 = 12.73%  (40 pps)
# S5 RAW - 9600 to 20000 - dataset_S5_len_910_size_100_bal_9600_100_20000_len_485.npz - MARE_95 = 20.75
# S1 RAW - 200  to 9600  - dataset_S1_len_1700_size_100_bal_200_100_9600_len_4130.npz - MARE_95 = 15%
# C2 Bal - 200  to 9600  - dataset_C2_len_402_size_100_bal_200_100_9600_len_863.npz   - MARE_95 = 27%
# BP RAW - 200  to 9600  - dataset_BP_len_116_size_100_raw_5500_100_7700_len_348.npz  - MARE_95 = 40% (under estimating)
# BP RAW - 200  to 9600  - dataset_BP_no_scale_len_401_size_100_bal_4000_100_8700_len_1203.npz - MARE_95 = 60% (over estimating)
# BP RAW - 200  to 9600  - dataset_BP_no_scale_f_len_401_size_100_bal_6000_100_13100_len_1203.npz - MARE_95 = 

# COM RAW - 9600 - 20000 - dataset_BSS_KLS_BrSS_DSS_ELS_C1_C2_len_19005_size_100_bal_9600_100_20000_len_3326 - MARE_95 = 25.4%

#------------------

# Model: COM_60048_0.2_adam_MAPE_75_3dir_relu_InsNet_nobias_2B_BSS_KLS_BrSS_DSS_ELS_C1_60K_1_USE
# C2 BAL - 200 to 9600 - dataset_C2_len_402_size_100_bal_200_100_9600_len_863.npz   - MARE_95 = 45%

t = time.time()
# os.chdir('D:/Canada/University/PhD/Research/Programs/Python/CNN/codes')

# Choose GPU to use (8 available) - skip 0 (busy)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU') #gpus[4:8]


LT = 0  #mD
UT = 15000 #mD
validation_split = 0.2
test_split = 0.2
reduce_data = "false"   #true/false
trial = '_1'
folder_name = "networks/"+"0 - COM_65248_0.2_adam_MAPE_100_3dir_relu_InsNet_nobias_2B_BSS_KLS_BrSS_ELS_C1_C2_65K_1_USE_BEST" #update
# Options:
# tragets [porosity, eff. Porosity, kx, ky, kz, Resolution]
data_type = 'BSS_US_1'  # Not working update "save_name" manually
data = np.load("datasets/balanced/dataset_BSS_US_len_27_size_100_bal_800_100_13500_len_81.npz", allow_pickle=True) #update
target = "3dir"
print(data.files)
x_test = data['samples']
y_test = data['k']
casenames = data['casenames']
direction = data['direction']
porosity = data['porosity']
eff_porosity = data['eff_porosity']
rock_type = data['rock_type']
k_min = data['k_min']
k_int = data['k_int']
k_max = data['k_max']
AR = data['AR']
DOA = data['DOA']
del data



info = np.zeros((len(y_test), 9), dtype="float64")
info[:, 0] = casenames
info[:, 1] = porosity
info[:, 2] = eff_porosity
info[:, 3] = rock_type
info[:, 4] = k_min
info[:, 5] = k_int
info[:, 6] = k_max
info[:, 7] = AR
info[:, 8] = DOA

data_len = len(y_test)



# Reduce data
if reduce_data == "true":
  indices = np.arange(0, data_len)
  indi_rem_1 = np.array(np.where(y_test < LT), dtype=int)
  indi_rem_2 = np.array(np.where(y_test > UT), dtype=int)
  indices_remove = np.append(indi_rem_1, indi_rem_2)
  del indi_rem_1, indi_rem_2
  indices_reduced = np.delete(indices, indices_remove, axis=0)
  x_test = x_test[indices_reduced, :, :, :]
  y_test = y_test[indices_reduced]
  casenames = casenames[indices_reduced]
  direction = direction[indices_reduced]
  info = info[indices_reduced, :]
  data_len = len(y_test)





print("Data length = "+str(len(y_test))+" samples")


# Reshape the images from (28, 28) to (28, 28, 1)
x_test = np.expand_dims(x_test, axis=4)



opt = "adam" #defult LR=0.001
O = "adam"
loss = 'mean_absolute_percentage_error'
L = 'MAPE'
metrics = 'mean_absolute_percentage_error'
m = 'MAPE'
use_bias = "False"

# 2 branches
# function for creating an inception block
def inception_module(layer_in, f1, f2, strides):
	  # 7x7 conv
    conv7 = layer_in
    # conv3 = Conv3D(f1, 1, strides=1, padding='same', use_bias="False", activation='relu')(conv3)
    conv7 = Conv3D(f1, 7, strides=strides, padding='same', use_bias="False")(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(f1, 7, strides=strides, padding='same', use_bias="False")(conv7)
    conv7 = Activation('relu')(conv7)
	  # 15x15 conv
    conv15 = layer_in
    # conv10 = Conv3D(f3_in, 1, strides=1, padding='same', use_bias="False", activation='relu')(conv10)
    conv15 = Conv3D(f2, 15, strides=strides, padding='same', use_bias="False")(conv15)
    conv15 = Activation('relu')(conv15)
    conv15 = Conv3D(f2, 15, strides=strides, padding='same', use_bias="False")(conv15)
    conv15 = Activation('relu')(conv15)
	  # 3x3 max pooling
    # skip = Conv3D(f1, 1, strides=strides, padding='same', use_bias="False", activation='relu')(layer_in)
    # if strides == 2:
    #    skip = AveragePooling3D(pool_size=2, strides=2, padding='same')(skip)
	  # concatenate filters, assumes filters/channels last
    # layer_out = concatenate([conv3, conv5, conv10, skip], axis=-1) # with skip connnection
    layer_out = concatenate([conv7, conv15], axis=-1) # without skip connnection
    return layer_out

input_shape=(100, 100, 100, 1)
# Use strategy for Multiple GPUs
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  # define model input
  input_image = Input(shape=(100, 100, 100, 1))
  # add inception module
  layer = inception_module(input_image, 16, 16, strides=2)
  layer = BatchNormalization()(layer)
  # layer = SpatialDropout3D(0.1)(layer)
  layer = Conv3D(16, 2, strides=2, padding='same', use_bias="False")(layer) #best result
  # layer = AveragePooling3D(pool_size=2, strides=2, padding='same')(layer) #moderate
  # layer = MaxPooling3D(pool_size=2, strides=2, padding='same')(layer)     #worse
  # add Conv3D block
  layer = Conv3D(32, 5, strides=1, padding='same', use_bias="False")(layer)
  layer = Activation('relu')(layer)
  # layer = BatchNormalization()(layer)
  layer = Conv3D(32, 5, strides=1, padding='same', use_bias="False")(layer)
  layer = Activation('relu')(layer)
  layer = BatchNormalization()(layer)
  layer = SpatialDropout3D(0.1)(layer)
  layer = Conv3D(32, 2, strides=2, padding='same', use_bias="False")(layer)
  # layer = AveragePooling3D(pool_size=2, strides=2, padding='same')(layer)
  # layer = MaxPooling3D(pool_size=2, strides=2, padding='same')(layer)
  
  # FCL part
  layer = Flatten()(layer)
  layer = Dense(128, use_bias=use_bias, activation='relu')(layer)
  layer = Dropout(0.1)(layer)
  layer = Dense(64, use_bias=use_bias, activation='relu')(layer)
  layer = Dense(1)(layer)



  # create model
  model = Model(inputs=input_image, outputs=layer)

  model.compile(optimizer=opt, loss=loss, metrics=[metrics])
  
  # Save the model to disk.
  save_name_1 = 'networks/predictions/'+data_type+'_'
  save_name_2 = str(data_len)
  save_name_3 = trial
  save_name = save_name_1 + save_name_2 + save_name_3
  os.mkdir(save_name)
  del save_name_1, save_name_2, save_name_3




  checkpoint_filepath = folder_name+'/checkpoint'
  # Load the model from disk later using:
  # model = tf.keras.models.load_model('saved_model.pb', compile=False)
  model.load_weights(checkpoint_filepath)
  
  # model.compile(optimizer='adam', loss=loss, metrics=[metrics])
  # history = np.load(folder_name+'/history.npy', allow_pickle='TRUE').item()
  print(model.summary())


  # Evaluate the model on the test data using `evaluate`
  print("Evaluate on test data")
  results = model.evaluate(x_test, y_test, batch_size=32)
  print("test loss(" + L + "), test(" + m + "):", results)
  t0 = time.time()
  predict_test = model.predict(x_test)
  t1 = time.time()
  inference_time = (t1-t0)/x_test.shape[0]
  print("Inference time = "+str(np.round(inference_time*1000, 1))+ "[ms]/sample")



y_test_mD = y_test
predict_test_mD = predict_test

    
y_test_mD_PE = np.zeros((len(predict_test)), dtype="float64")
for i in range(0, len(predict_test)):
    y_test_mD_PE[i] = np.abs(predict_test_mD[i]-y_test_mD[i])/y_test_mD[i]
MAPE_mD = np.mean(y_test_mD_PE)

UT_90 = np.percentile(y_test_mD_PE, 95)
indices_remove = np.array(np.where(y_test_mD_PE >= UT_90))
y_test_mD_PE_90 = np.delete(y_test_mD_PE, indices_remove)
y_test_mD_90 = np.delete(y_test_mD, indices_remove)
predict_test_mD_90 = np.delete(predict_test_mD, indices_remove)
MAPE_90_mD = np.mean(np.delete(y_test_mD_PE, indices_remove))

MAPE_75_mD = np.mean(np.delete(y_test_mD_PE, np.array(np.where(y_test_mD_PE >= np.percentile(y_test_mD_PE, 75)))))
MAPE_50_mD = np.mean(np.delete(y_test_mD_PE, np.array(np.where(y_test_mD_PE >= np.percentile(y_test_mD_PE, 50)))))

###########################################################
y_test_mD_error = predict_test_mD[:, 0] - y_test_mD
Y_summary = np.zeros((len(y_test), 14), dtype="float64")
Y_summary[:, 0] = casenames
Y_summary[:, 1] = info[:, 0]
Y_summary[:, 2] = info[:, 1]
Y_summary[:, 3] = info[:, 2]
Y_summary[:, 4] = info[:, 3]
Y_summary[:, 5] = info[:, 4]
Y_summary[:, 6] = info[:, 5]
Y_summary[:, 7] = info[:, 6]
Y_summary[:, 8] = info[:, 7]
Y_summary[:, 9] = info[:, 8]
Y_summary[:, 10] = y_test_mD
Y_summary[:, 11] = predict_test_mD[:, 0]
Y_summary[:, 12] = y_test_mD_error
Y_summary[:, 13] = y_test_mD_PE
###########################################################



print("Total test samples: "+str(len(y_test)))
print("MAPE < 95th percentile [mD]:" + str(np.round(MAPE_90_mD*100, 2))+" %")
time = np.round((time.time() - t)/60,1)

Y_summary_red = np.delete(Y_summary, indices_remove, axis=0)


# save results
savetxt(save_name+'/results_summary.csv', Y_summary, delimiter=',')
savetxt(save_name+'/results_summary_dir.csv', direction, delimiter=',', fmt="%s")

# Plot Predictions

# min = np.min([np.min(predict_test), np.min(y_test), np.min(predict_train), np.min(y_train)])*0.95
# max = np.max([np.max(predict_test), np.max(y_test), np.max(predict_train), np.max(y_train)])*1.05

# min = min([np.min(y_test), np.min(y_train)])*0.9
# max = max([np.max(y_test), np.max(y_train)])*1.1

min = LT
max = UT
plt.plot([min, max], [min, max], c='b', linewidth=1, alpha=0.9)
plt.scatter(y_test_mD, predict_test_mD, c='r', s=1.5)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlim(min, max)
# plt.ylim(min, max)
tl1 = "Model Predictions ("+target+")\n"
tl2_1 = "Test Resuts\n"
tl2_2 = "(MAPE, mD): "+str(np.round(MAPE_mD*100, 2))+" %\n"
tl4 = "MAPE < 95th percentile [mD]:" + str(np.round(MAPE_90_mD*100, 2))+" %"
plt.title(tl1+tl2_1+tl2_2+tl4)
# plt.title('Model Predictions ('+target+")\n Test Resuts ("+m+"): " +
#           str(np.round(results[1], 3))+" [log], (MAPE): " +str(np.round(MAPE_mD, 3))+" [D]")
plt.ylabel('Predicted Permeability')
plt.xlabel('True Permeability')
plt.legend(['True', 'Predictions (Test)'], loc='upper left')
plt.savefig(save_name+'/predictions_graph_all.png', dpi=300)
# plt.show()
plt.clf()

# Plot Predictions < 90th percentile
plt.plot([min, max], [min, max], c='b', linewidth=1, alpha=0.9)
plt.scatter(y_test_mD_90, predict_test_mD_90, c='r', s=1.5)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlim(min, max)
# plt.ylim(min, max)
tl1 = "Model Predictions ("+target+")\n"
tl2_1 = "Test Resuts\n"
tl2_2 = " (MAPE, mD): "+str(np.round(MAPE_mD*100, 2))+" %\n"
tl4 = "MAPE < 95th percentile [mD]:" + str(np.round(MAPE_90_mD*100, 2))+" %"
plt.title(tl1+tl2_1+tl2_2+tl4)
# plt.title('Model Predictions ('+target+")\n Test Resuts ("+m+"): " +
#           str(np.round(results[1], 3))+" [log], (MAPE): " +str(np.round(MAPE_mD, 3))+" [D]")
plt.ylabel('Predicted Permeability')
plt.xlabel('True Permeability')
plt.legend(['True', 'Predictions (Test)'], loc='upper left')
plt.savefig(save_name+'/predictions_graph_90.png', dpi=300)
# plt.show()
plt.clf()




UT_90 = np.percentile(y_test_mD_PE, 95)
UT_75 = np.percentile(y_test_mD_PE, 75)
UT_50 = np.percentile(y_test_mD_PE, 50)
# Plot Predictions Error (All)
max = np.max([np.max(y_test_mD_PE)*1.2, 0.5])
plt.scatter(y_test_mD, y_test_mD_PE, s=1)
legend1 = plt.legend(['Predictions (Test)'], loc='upper left')
plt.axhline(UT_90, color='r', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_75, color='b', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_50, color='g', linestyle='dashed', linewidth=1, alpha=0.8)
legend2 = plt.legend(['95% Percentile', '75% Percentile', '50% Percentile'], loc='upper right')
plt.gca().add_artist(legend1)
plt.ylim(0, max)
tl1 = "Predictions Error Distribution"
plt.title(tl1)
plt.ylabel('error')
plt.xlabel('True Permeability [mD]')
plt.savefig(save_name+'/predictions_error_all_graph.png', dpi=300)
# plt.show()
plt.clf()

# Plot Predictions Error  < 90th percentile
max = np.max([np.max(y_test_mD_PE_90)*1.5, 0.5])
plt.scatter(y_test_mD_90, y_test_mD_PE_90, s=1, label='Predictions (Test)')
legend1 = plt.legend(loc='upper left')
plt.axhline(UT_90, color='r', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_75, color='b', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_50, color='g', linestyle='dashed', linewidth=1, alpha=0.8)
legend2 = plt.legend(['90% Percentile', '75% Percentile', '50% Percentile'], loc='upper right')
plt.gca().add_artist(legend1)
plt.ylim(0, max)
tl1 = "Predictions Error Distribution < 90th percentile"
plt.title(tl1)
plt.ylabel('error')
plt.xlabel('True Permeability [mD]')
plt.annotate('Mean error= '+str(np.round(MAPE_90_mD, 3)), (np.max(y_test_mD)*0.75,UT_90), textcoords="offset points", xytext=(0,-10), ha='left', size=9, weight='bold')
plt.annotate('Mean error= '+str(np.round(MAPE_75_mD, 3)), (np.max(y_test_mD)*0.75,UT_75), textcoords="offset points", xytext=(0,-10), ha='left', size=9, weight='bold')
plt.annotate('Mean error= '+str(np.round(MAPE_50_mD, 3)), (np.max(y_test_mD)*0.75,UT_50), textcoords="offset points", xytext=(0,-10), ha='left', size=9, weight='bold')
plt.savefig(save_name+'/predictions_error_90_graph.png', dpi=300)
# plt.show()
plt.clf()



# Draw Box Plot
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.boxplot.html
plt.boxplot(np.abs(y_test_mD_error), sym='b.', showmeans=True)
tl1 = "Permeability Prediction Error (<95th percentile)\n"
tl2 = "mean absolute error: "+str(int(np.mean(np.abs(y_test_mD_error))))+" [mD]"
plt.title(tl1+tl2)
labels = ['3D CNN']
ticks = [1]
plt.xticks(ticks, labels, rotation='horizontal')
# plt.xlabel('3D CNN')
plt.ylabel('Error [mD]')
plt.savefig(save_name+'/box_plot_mD.png', dpi=300)
# plt.show()
plt.clf()

# Draw Box Plot
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.boxplot.html
plt.boxplot(np.abs(y_test_mD_PE_90*100), sym='b.', showmeans=True)
tl1 = "Permeability Prediction Error (<95th percentile)\n"
tl2 = "mean error: "+str(np.round(MAPE_90_mD*100, 2))+" %"
plt.title(tl1+tl2)
labels = ['3D CNN']
ticks = [1]
plt.xticks(ticks, labels, rotation='horizontal')
plt.ylabel('Error (%)')
plt.savefig(save_name+'/box_plot_error_ratio.png', dpi=300)
# plt.show()
plt.clf()



# Visualize data (Histogarm)
for (x,y) in ([y_test_mD, "Test"], [y_test_mD_PE*100, "Test Error(%)"], [y_test_mD_PE_90*100, "Test Error (<90th percentile)"] ):
  bins = int(np.round((np.max(x) - np.min(x))/100))
  title = " Samples Histogram, 100 mD step"
  title_2 = "Permeability Value [mD]"
  if bins < 10:
    bins = int(np.round((np.max(x) - np.min(x))/5))
    title = " Histogram"
    title_2 = "Error(%)"
  plt.hist(x, bins=bins, color='b', alpha=0.8)  # arguments are passed to np.histogram
  plt.axvline(x.mean(), color='r', linestyle='dashed', linewidth=1)
  plt.axvline(np.median(x), color='k', linestyle='dashed', linewidth=1)
  plt.ylabel('Count')
  plt.xlabel(title_2)
  plt.legend(['Mean', 'Median'],
             loc='upper right')
  plt.title(str(y)+title)
  plt.savefig(save_name+'/histogram_'+str(LT)+'_'+str(UT)+'_'+str(y)+'_'+str(len(x))+'.png', dpi=300)
  # plt.show()
  plt.clf()