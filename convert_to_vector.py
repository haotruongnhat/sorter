import numpy as np
import cv2

mount_of_dep=665
mount_of_xau=687

for i in range(mount_of_dep):
    image=cv2.imread('D:/do_an_tot_nghiep/python/0207/dep%s.png'%i,0)
    cvt_image=np.concatenate(image).reshape(1,-1)
    cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/dep/dep%s.png'%i,cvt_image)

for i in range(mount_of_xau):
    image=cv2.imread('D:/do_an_tot_nghiep/python/0207/xau%s.png'%i,0)
    cvt_image=np.concatenate(image).reshape(1,-1)
    cv2.imwrite('D:/do_an_tot_nghiep/python/dataset/xau/xau%s.png'%i,cvt_image)
