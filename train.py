import data
import pandas as pd     
import numpy as np
import keras
import os
import os.path
import csv
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,MaxPooling2D,BatchNormalization,Activation
from keras.optimizers import Adam,SGD
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from mylib.models import densesharp, metrics, losses,DenseNet
from mylib.models.DenseNet import createDenseNet
from keras.optimizers import SGD
import pandas as pd
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping,ModelCheckpoint,History
from sklearn.model_selection import train_test_split

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
checkpointer = ModelCheckpoint(filepath='./result/process_h5/%s/weights.{epoch:02d}.h5' % 'fourth26_2', verbose=1,
                               period=1, save_weights_only=False)


densenet_depth = 28
densenet_growth_rate = 12
nb_classes = 2
batch_size = 16

model = createDenseNet(nb_classes=nb_classes,img_dim=[32,32,32,1],depth=densenet_depth,growth_rate = densenet_growth_rate)
model.compile(loss=binary_crossentropy,optimizer=SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True), metrics=['accuracy'])

model.summary()  # print the model

#读入训练集数据和标签
x_train=data.get_dataset()
x_train=np.array(x_train)
x_train_label=data.get_label()
x_train=x_train.reshape(x_train.shape[0],32,32,32,1)
x_train=x_train.astype('float32')/255


#数据增强
#x_train_mixup,x_train_label_mixup = data.mixup_data(x_train,x_train_label,n=50,alpha=0.5)

#x_train=np.r_[x_train,x_train_mixup]
#x_train_label=np.r_[x_train_label,x_train_label_mixup]

#测试集
y_voxel=np.array(data.get_testdataset())
y_voxel=y_voxel.reshape(y_voxel.shape[0],32,32,32,1)
y_voxel=y_voxel.astype('float32')/255

x_train_train, x_train_test, x_label_train, x_label_test = train_test_split(x_train, x_train_label, test_size=0.1, random_state=3)
 
history = model.fit(x_train_train,x_label_train, batch_size=batch_size, epochs=70,validation_data=(x_train_test,x_label_test), verbose=2, shuffle=False)
loss,accuracy = model.evaluate(x_train_train,x_label_train)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,accuracy))
loss,accuracy = model.evaluate(x_train_test,x_label_test)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,accuracy))

####最终估计的地方
#print(model.predict(x_predict))

s = model.predict(y_voxel)
print(s)
#np.savetxt('new.csv', s, delimiter = ',') 

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss']) 
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy&loss')
plt.ylabel('data')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left') 
plt.savefig("picture.png")
plt.show()
plt.close()

'''
# summarize history for loss 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("loss.png")
plt.show()
'''

np.save('./result/npy/result26_2.npy',s[0])
result = np.load('./result/npy/result3.npy')
model.save('./result/final_h5/model_mask26_2.h5')
csv = pd.read_csv("./upload/sample.csv")
csv.iloc[:, 1] = s[:, 1]
csv.columns = ['Id', 'Predicted']
csv.to_csv("./upload/upload26_2.csv", index=None)

