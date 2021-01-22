from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import numpy as np
import cv2
from sklearn import preprocessing
from sklearn.externals import joblib



target=cv2.imread('/home/pi/DATN/scikit/target_training_945.png',0)


target=np.float32(target.reshape(945,))

##print np.float32(target)
##target_scaler=preprocessing.MinMaxScaler()
##target=transform(target)
##
data_training=np.float32(cv2.imread('/home/pi/DATN/scikit/csv_training_945.png',0))
data_training = data_training/255.0
print data_training
####data_scaler=preprocessing.MinMaxScaler()
####data_training=transform(data_training)
##print data_training
clf=MLPClassifier(hidden_layer_sizes=(56,),activation='logistic',solver='sgd',learning_rate='adaptive',max_iter=1000)
#,learning_rate='adaptive'
clf.fit(data_training,target)

filename = 'model0807_56.sav'
joblib.dump(clf,filename)

counter =0
for i in range(466,664):
    ob=cv2.imread('/home/pi/Downloads/dataset/dep/dep%s.png'%i,0)/255.0
    result = clf.predict(ob)
    print result
    if(result>0.5):
        counter=counter +1
    

print counter*100/(664-466)


ob=cv2.imread('/home/pi/Downloads/dataset/dep/dep%s.png'%i,0)/255.0

print clf.predict(ob)

ob=cv2.imread('/home/pi/Downloads/dataset/dep/dep555.png',0)/255.0

print clf.predict(ob)
