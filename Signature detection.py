from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import os
from PIL import Image
from sklearn.cross_validation import train_test_split

m,n = 50,50

path1="test";
path2="train";

classes=os.listdir(path2)
x=[]
y=[]
imgfiles=os.listdir(path2);
for img in imgfiles:
    im=Image.open(path2+'\\'+img);
    im=im.convert(mode='RGB')
    imrs=im.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x.append(imrs)
    aa=img[:img.index("_")]
    y.append(aa)
    print(aa)
        
x=np.array(x);
y=np.array(y);

classes=18
filters=32
pool=2
conv=3

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,classes)

model= Sequential()
model.add(Convolution2D(filters,conv,conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(Convolution2D(filters,conv,conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(pool,pool)));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

nb_epoch=5;
batch_size=5;
model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))



x1=[]
x=[]
files=os.listdir(path1);
for img in files:
    x1.append(img[:img.index("_")])
    im = Image.open(path1 + '\\'+img);
    im=im.convert(mode='RGB')
    imrs = im.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x.append(imrs)



xxx=np.array(x);
predictions = model.predict(xxx)

print(predictions)