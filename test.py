import numpy as np
import csv
import pandas as pd   
import data
from keras.models import load_model

y_voxel=np.array(data.get_testdataset())
y_voxel=y_voxel.reshape(y_voxel.shape[0],32,32,32,1)
y_voxel=y_voxel.astype('float32')/255

model = load_model('./BestResult/weights.106.h5')
result = model.predict(y_voxel,batch_size = 1)
csv = pd.read_csv("./data/Sample.csv")
csv.iloc[:, 1] = result[:, 1]
csv.columns = ['Id', 'Predicted']
csv.to_csv("./submission.csv", index=None)