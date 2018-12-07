
# coding: utf-8

# # Machine Learing Project to predict Digital Ad Fraud Nov-16-2018

# In[2]:


from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Activation
from keras.utils import normalize
import keras.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## 1. Import the dataset

# In[3]:


#dataset = pd.read_csv("/Users/cmallavarapu/Documents/Chiran/MSDS/ML/MLProject/train_sample.csv")
dataset = pd.read_csv("/Users/bujji/Documents/MS_SMU/Sem_5_ML/MLProject/train_sample.csv")
#dataset = pd.read_csv("/Users/chiranjeevimallavarapu/Documents/Chiran/MS_SMU/ML/MLProject/train_sample.csv")


# In[4]:


dataset.head()


# ### As our goal is to find when a click is fraudulent we will make the response as 1 in those cases

# In[5]:


dataset['response'] = 1-dataset['is_attributed']


# ### 1.1 Field definitions

# ####                    ip:  
#                 ip address of click.
# 
# ####                   app: 
#                 app id for marketing.
# 
# ####                device: 
#                 device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# 
# ####                    os: 
#                 os version id of user mobile phone
# 
# ####               channel: 
#                 channel id of mobile ad publisher
# 
# ####            click_time: 
#                 timestamp of click (UTC)
# 
# ####       attributed_time: 
#                 if user download the app for after clicking an ad, this is the time of the app download
# 
# ####         is_attributed: 
#                 the target that is to be predicted, indicating the app was downloaded
#                 
# ####        response 
#                 if a click is fraud it will have a value of 1, otherwise 0

# In[6]:


len(dataset)


# ### 1.2 Adjust 'click_time' date value to pandas date and calculate time delta from min time

# In[7]:


dataset['click_time'] = pd.to_datetime(dataset['click_time'])


# In[8]:


sttime = pd.Timestamp(min(dataset['click_time']))


# In[9]:


sttime


# In[10]:


dataset['delta'] = (dataset['click_time']-sttime).astype('timedelta64[s]')


# In[11]:


dataset.head()


# ##  2. Explore the data

# In[12]:


import seaborn as sns
sns.set(style="ticks")


# In[13]:


sns.pairplot(dataset, hue="device")


# ### 2.1 Lets plot response Vs app

# In[14]:


np.random.seed(19680801)
fig = plt.figure(figsize=(8,6))

N = 50
#colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

colors = dataset['device']

# c=colors,
scatter=plt.scatter(dataset['app'], dataset['response'], s=100,alpha=0.4, c = colors,cmap='cool')
fig.suptitle('Scatter Plot with Devices as colorbar', fontsize=30)
plt.xlabel('app', fontsize=30)
plt.ylabel('response', fontsize=30)


plt.colorbar(scatter)
plt.show()


# ### 2.2 Lets plot response vs device

# In[15]:


np.random.seed(19680801)
fig = plt.figure(figsize=(8,6))

N = 50
#colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

colors = dataset['app']

# c=colors,
scatter=plt.scatter(dataset['device'], dataset['response'], s=100,alpha=0.40, c = colors,cmap='winter')
fig.suptitle('Scatter Plot with app as colorbar', fontsize=30)
plt.xlabel('device', fontsize=30)
plt.ylabel('response', fontsize=30)


plt.colorbar(scatter)
plt.show()


# ### 2.3 Lets plot response vs OS

# In[16]:


np.random.seed(19680801)
fig = plt.figure(figsize=(8,6))

N = 50
#colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

colors = dataset['device']

# c=colors,
scatter=plt.scatter(dataset['os'], dataset['response'], s=100,alpha=0.4, c = colors,cmap='summer')
fig.suptitle('Scatter Plot with Devices as colorbar', fontsize=30)
plt.xlabel('os', fontsize=30)
plt.ylabel('response', fontsize=30)


plt.colorbar(scatter)
plt.show()


# ##  3. Lets model the data using Deep Neural Networks

# In[334]:


# Slice the dataset to get the features (X) and Response(Y)


# In[335]:


X_imb = np.array(dataset)[:,[0,1,2,3,4,9]]
Y_imb = np.array(dataset)[:,8]


# In[336]:


print ("Rows of fraud clicks",len(Y_imb[Y_imb == 1]))
print ("Rows of genuine clicks",len(Y_imb[Y_imb == 0]))


# ### 3.1 Balancing the dataset
# 
# We know that the nature of this data (Fraud Detection) is such that very little percentage of response variable can be considered attributed (someone installed an APP) and major portion of data is considered fraudulent. Although the main focus is to predict the fraud data it is also essential to not mis classify the genuine clicks as fraud. 
# 
# Because our data is imbalanced we will some sampling techniques mentioned below.
# 
# We will use the technique of combining over and under sampling i.e Class to perform over-sampling using SMOTE and cleaning using ENN (Edited nearest neighbours). This will result in a more balanced class of response(app installed or not after clicking an ad)

# In[337]:


from collections import Counter
from imblearn.combine import SMOTEENN
sme = SMOTEENN(random_state=42)
X, Y = sme.fit_resample(X_imb, Y_imb)


# In[338]:


print ("Total rows of feature data before balancing is", len(X_imb), "with fraud clicks",len(Y_imb[Y_imb==1]),"and genuine clicks",len(Y_imb[Y_imb==0]), "after SMOTEENN total rows of feature data is" ,len(X), "with fraud clicks of",len(Y[Y==1]),"and genuine clicks of",len(Y[Y==0]), ",a more balanced dataset")


# ### 3.2 Classification of response
# 
# Now lets convert our response variable to a class variable so we can feed to DNN

# In[401]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# ### 3.3 Model with simple feed forward DNN

# In[340]:


model = Sequential()

model.add(Dense(8, input_dim=6, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(12, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
snn=model.fit(X, dummy_y, epochs=20, batch_size=1000)
scores = model.evaluate(X, dummy_y)
print("hi", "\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# #### This resulted a good accuracy but we are not sure if it is overfitting as we did not cross validate

# ### 3.4 Model with CNN along with Cross Validation

# In[341]:


from sklearn.model_selection import train_test_split


# In[402]:


X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)


# In[343]:


import string
import random
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, SpatialDropout1D, Conv1D,Conv2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D


# #### Now the CNN is architected per below

# Convolutional Layer
# 
# Pooling Layer
# 
# Normalization Layer
# 
# Fully-Connected Layer

# In[344]:


# Standard off the shelf 
def create_model(i_shape):
    #refactored to allow for num_labels
    model = Sequential()
    
    # CNN : 16 layers, 3 filters 
    model.add(Conv1D(16, 3, input_shape=i_shape,padding='same', use_bias='False'))
    # Notice the Batch normalization that is critical for the data
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout1D(0.2))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D())

    #model.add(Dense(32, input_dim=6, activation='relu'))
    #model.add(SpatialDropout1D(0.2))
    #model.add(Dropout(0.25))
    
    # CNN: 64 layers, 3 filters 
    model.add(Conv1D(32, 3, padding='same',  use_bias='False'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D())
    #model.add(Flatten())
    
     # CNN: 16 layers, 3 filters 
    model.add(Conv1D(16, 3, padding='same',  use_bias='False'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    #model.add(MaxPooling1D())

    # Fully connected layer 
    model.add(Dense(10, use_bias='False'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    
    model.add(Dense(2, activation='sigmoid'))
    
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer= 'adam', metrics=['accuracy'])
    model.summary()
    return model


# In[403]:


X_train = X_train.reshape(-1, 6, 1)
X_test= X_test.reshape(-1, 6, 1)


# In[404]:


X_train.shape


# In[405]:


X_test.shape


# #### Construct and fit the model

# In[348]:


clf = create_model(X_train.shape[1:])


# In[349]:


cnn= clf.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=20000)


# In[350]:


X = X.reshape(-1, 6, 1)


# In[351]:


X.shape


# In[352]:


scores = clf.evaluate(X, dummy_y)
print("hi", "\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[354]:


fig = plt.figure(figsize=(16,8))
plt.plot(cnn.history['acc'],'r')
plt.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 120, 2.0))
plt.yticks(np.arange(0.5, 1, 0.05))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])


# ### 3.5 Predict on test

# In[406]:


X_ALL = X_test.reshape(-1, 6, 1)

Y_predict = clf.predict_classes(X_ALL)


# In[407]:


len(Y_predict)


# #### As Y predict will be class data we will convert to single value to plot. As there are only two classes (0/1) we will simply set the Y as the value of second class (one hot encoded value) 
# 
# So for example Y predicted 
# 
#     Class 0(not attributed)     Class 1(attributed)
# 
#                      0          1
#                      1          0
#                  
# In the above scenarios, Y (is attributed) should be 1, 0 respectively which is also equal to class value of class 1(attributed

# In[408]:


Y_plot =  Y_test[:,1]


# In[409]:


Y_plot.shape


# In[410]:


Y_plot[0:3]


# ### 3.6 Evaluation : Precision / Recall & ROC 

# In[364]:


A_Y_1=tuple([Y_plot == 1])
A_Y_0=tuple([Y_plot == 0])


# In[375]:


print ("True positives are " , len(Y_predict[A_Y_1][Y_predict[A_Y_1] == 1]))
tp_cnn=len(Y_predict[A_Y_1][Y_predict[A_Y_1] == 1])

print ("True negatives are " , len(Y_predict[A_Y_0][Y_predict[A_Y_0] == 0]))
tn_cnn=len(Y_predict[A_Y_0][Y_predict[A_Y_0] == 0])


print ("False positives are " , len(Y_predict[A_Y_0][Y_predict[A_Y_0] == 1]))
fp_cnn=len(Y_predict[A_Y_0][Y_predict[A_Y_0] == 1])
print ("False negatives are " , len(Y_predict[A_Y_1][Y_predict[A_Y_1] == 0]))
fn_cnn=len(Y_predict[A_Y_1][Y_predict[A_Y_1] == 0])


# In[376]:


print ("Precision", tp_cnn/(tp_cnn+fp_cnn)*100)
p_cnn = tp_cnn/(tp_cnn+fp_cnn)


# In[377]:


print ("Recall", tp_cnn/(tp_cnn+fn_cnn)*100)
r_cnn = tp_cnn/(tp_cnn+fn_cnn)


# In[378]:


F1_cnn = 2*p_cnn*r_cnn/(p_cnn+r_cnn)
print ("F1 score CNN", F1_cnn)


# ### In the case of fraud detection Recall of 90% (90% of total fraud clicks were detected) bears a high importance as we wouldnt want to miss any click that is fraudulent. Now having this high Recall results in a lesser precision (66%) but that is something one can live with. It may be ok to call a click that is genuine as fraud, but not vice versa. Meaning its ok not to pay for a genuine click but its not ok to pay for a fraud activity because in terms of volume, fraud activity is more than 99%.
# 
# ### Now lets plot Precision Recall curve

# In[416]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score


# In[412]:


# predict probabilities
probs = clf.predict_proba(X_test.reshape(-1,6,1))
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = clf.predict(X_test.reshape(-1,6,1))
yhat = np.rint(yhat[:,1])
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(Y_plot, probs)


# ### Plot Precision/Recall curve

# In[1]:


# calculate F1 score
f1 = f1_score(Y_plot, yhat)
# calculate precision-recall AUC
auc_cnn = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(Y_plot, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_cnn, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
plt.plot(recall, precision, marker='.')
plt.plot([.91], [.66], marker='X',markersize=12)
plt.xlabel('Recall')
plt.ylabel('Precision')



# show the plot
plt.show()


# ### ROC Curve

# In[384]:


from sklearn.metrics import roc_curve
#y_pred_keras = keras_model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_plot.astype(int), Y_predict)


# In[385]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# ### 3.7 Plot Predicted vs Actual for test data to get a visual representation

# #### We will jitter the data a bit

# In[386]:


for i in range(0,len(Y_plot)):
    #print()
    Y_plot[i] = Y_plot[i] + random.uniform(0, 0.25)
    if i%10000 ==0:
        print (Y_plot[i])


# In[387]:


Y_plot.shape


# In[388]:


Y_predict.shape


# In[389]:


Y_predict = Y_predict.reshape(-1)


# In[390]:


Y_predict = Y_predict.astype(np.float64)


# In[391]:


for i in range(0,len(Y_predict)):
    #print()
    Y_predict[i] = Y_predict[i] + random.uniform(0, 0.25)
    if i %10000 == 0:
        print (Y_predict[i])
        
    


# In[392]:


np.random.seed(19601)

#colors = df2['Wiki']
fig = plt.figure(figsize=(17,17))
#plt.scatter(df2[df2['State']=='Florida']['Votes'], df2[df2['State']=='Florida']['Prediction'], s=200,alpha=0.5, c='#4c2373')
plt.scatter(Y_plot, Y_predict, s=1,alpha=0.5, c='Blue', cmap = 'winter')
fig.suptitle('Scatter Plot - Predicted Vs Actual ',fontsize=30)
plt.ylabel('Predicted',fontsize=30)
plt.xlabel('Actual',fontsize=30)
plt.axis('equal')
#plt.yticks(np.arange(0.0, 1.1, 0.1))
#plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.legend(fontsize=30)
plt.plot( [0,1],[0,1] )
plt.show()

