import cv2
import numpy as np
import csv

mount_of_dep=464
mount_of_xau=479

csv_dep=cv2.imread('D:/do_an_tot_nghiep/python/dataset/dep/dep0.png',0)
target_dep=np.array([[1]])
for i in range(mount_of_dep):
    i=i+1
    image=cv2.imread('D:/do_an_tot_nghiep/python/dataset/dep/dep%s.png'%i,0)
    csv_dep=np.vstack((csv_dep,image))
    target_dep=np.vstack((target_dep,[[1]]))
cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/csv/csv_dep_465.png',csv_dep)
cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/csv/target_dep_465.png',target_dep)



csv_xau=cv2.imread('D:/do_an_tot_nghiep/python/dataset/xau/xau0.png',0)
target_xau=np.array([[0]])
for i in range(mount_of_xau):
    i=i+1
    image_xau=cv2.imread('D:/do_an_tot_nghiep/python/dataset/xau/xau%s.png'%i,0)
    csv_xau=np.vstack((csv_xau,image_xau))
    target_xau=np.vstack((target_xau,[[0]]))
cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/csv/csv_xau_480.png',csv_xau)
cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/csv/target_xau_480.png',target_xau)



######################## training data #######
csv_training=np.vstack((csv_dep,csv_xau))
cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/csv/csv_training_945.png',csv_training)

target_training=np.vstack((target_dep,target_xau))
cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/csv/target_training_945.png',target_training)


