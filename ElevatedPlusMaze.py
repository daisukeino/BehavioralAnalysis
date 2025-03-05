#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from subprocess import call
from PIL import Image
import sys
import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse

def make_directory(dir_name):
    call("mkdir " + dir_name, shell=True)

def open_multitiff(filename, stack_num):
    
    img_pil = Image.open(filename + ".tif")
    img = []


    count = 0
    while count<stack_num:
        img_pil.seek(count)
        img_tmp = np.asarray(img_pil)
        #img_tmp.flags.writeable = True
        img.append(img_tmp)
        count += 1
        print(count,end=",")
    
    print('\n')
    
    return img


def main():

    #Input filename
    filename = "1817_EP"
    outdir = "out"
    outdir_img = "img"
    outdir_sum = "sum"
    
    #x_pixels = 238
    #y_pixels = 238
    last_frame = 1980
    initial_frame = last_frame-1801
    
    
    stack_num = last_frame
  
    make_directory(outdir)
    make_directory(outdir_img)
    make_directory(outdir_sum)
    
    #Centroid positions
    cx = []
    cy = []
    
    #Open a multitiff
    img = np.array(open_multitiff(filename, stack_num))
    y_pixels, x_pixels= img[0].shape[:3]
    
    background_img = cv2.bitwise_not(img[0])
    
    contour_img = img[0].copy()
    line_img = img[0].copy()*0
    heat_img = img[0].copy()*0
    
    #define the threshold of open- and closed arms
    left_th = int(9*x_pixels/20)
    right_th = int(11*x_pixels/20)
    
    #state 0: closed, 1: open
    state_label = []
    
    #number of entry
    entry_num = 0
      
    for i in range(initial_frame, last_frame, 1):
        print("stack_num: " + str(i))
        
        inv_img = cv2.bitwise_not(img[i-1])
        
        sub_img =  cv2.subtract(inv_img, background_img)
        
        sub_img[0:left_th, 0:left_th] = 0
        sub_img[right_th+1:y_pixels+1, 0:left_th] = 0
        sub_img[0:left_th, right_th+1:x_pixels+1] = 0
        sub_img[right_th+1:y_pixels+1, right_th+1:x_pixels+1] = 0
        

        # Otsu's thresholding
        _, img_th = cv2.threshold(sub_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
      
        img_th_contours, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contour = len(img_th_contours)
        
        #When a single contour was detected
        if (num_contour == 1):

            contour_img[:] = 0
            cv2.drawContours(contour_img, [img_th_contours[0]], -1, 255, -1)
            cv2.imwrite(outdir_img + "/" + str(i)  + ".tif", contour_img)
            
            heat_img = heat_img + contour_img/255
            
            Moment_cnt = cv2.moments(img_th_contours[0])
            
            x_cent = Moment_cnt['m10']/Moment_cnt['m00']
            y_cent = Moment_cnt['m01']/Moment_cnt['m00']
            
            cx.append(x_cent)
            cy.append(y_cent)
            
            #if (x_cent < left_th or x_cent > right_th):
            if (x_cent < int(8*x_pixels/20) or x_cent > int(12*x_pixels/20)):
                state_label.append(1)
            
            else:
                state_label.append(0)

        #When multiple contours were detected
        elif (len(img_th_contours) > 1):
            contour_area = []
            max_j = 0
            
            for j in range (0, num_contour, 1):
                contour_area.append(cv2.contourArea(img_th_contours[j]))
                
                if (j==0):
                    max_area = contour_area[j]
                
                else:
                    if (contour_area[j] > max_area):
                      max_area = contour_area[j]
                      max_j = j
                      
            contour_img[:] = 0
            cv2.drawContours(contour_img, [img_th_contours[max_j]], -1, 255, -1)
            cv2.imwrite(outdir_img + "/" + str(i)  + ".tif", contour_img)
            
            heat_img = heat_img + contour_img/255
            
            Moment_cnt = cv2.moments(img_th_contours[max_j])
            
            x_cent = Moment_cnt['m10']/Moment_cnt['m00']
            y_cent = Moment_cnt['m01']/Moment_cnt['m00']
            
            cx.append(x_cent)
            cy.append(y_cent)
            
            #if (x_cent < left_th or x_cent > right_th):
            if (x_cent < int(8*x_pixels/20) or x_cent > int(12*x_pixels/20)):
                state_label.append(1)
            
            else:
                state_label.append(0)
           
        else:
            print(str(i))
            print("Failure in mice recognition!")
            sys.exit()
    
    for k in range (0, last_frame - initial_frame - 1, 1):
        line_tmp_img = img[0].copy()*0
        cv2.line(line_tmp_img, (round(cx[k]), round(cy[k])), (round(cx[k+1]), round(cy[k+1])), color=(10,0,0), thickness=1)
        line_img = line_img + line_tmp_img
    
   
    
    heat_img = heat_img.astype("uint16")   
    cv2.imwrite(outdir_sum + "/" + filename  + "_line.tif", line_img)
    cv2.imwrite(outdir_sum + "/" + filename  + "_heat.tif", heat_img)
    
    distance_cm = [0]*2
    time_s = [0]*2
    
    for l in range (0, last_frame - initial_frame - 1, 1):
        if(state_label[l+1] == 0):
            distance = np.sqrt((cx[l+1]-cx[l])*(cx[l+1]-cx[l]) + (cy[l+1]-cy[l])*(cy[l+1]-cy[l]))*50/x_pixels
            distance_cm[0] += distance
            time_s[0] += 1/3
        
        else:
            distance = np.sqrt((cx[l+1]-cx[l])*(cx[l+1]-cx[l]) + (cy[l+1]-cy[l])*(cy[l+1]-cy[l]))*50/x_pixels
            distance_cm[1] += distance
            time_s[1] += 1/3
            
            if(state_label[l] == 0):
                entry_num += 1
                
            
            

    wfilename1 = outdir + "/" + filename + "_summary.txt"
    f1 = open(wfilename1, 'w')
    
    f1.write("Total distance (cm)" + '\t' + "Distance in closed arm (cm)" + '\t' + "Distance in open arm (cm)" +  \
             '\t' + "Times in closed arm (s)" + '\t' + "Times in open arm (s)" + "Open arm entry" + '\n' \
             + str(distance_cm[0] + distance_cm[1]) + '\t' + str(distance_cm[0])+ '\t' + str(distance_cm[1])+ \
             '\t' +  str(time_s[0]) +  '\t' +  str(time_s[1]) + '\t' + str(entry_num) + '\n' )
    
        
      
    

    
if __name__ == '__main__':
    main()        
