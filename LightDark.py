#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from subprocess import call
from PIL import Image
import sys
import matplotlib.pyplot as plt
import glob
import os

def make_directory(dir_name):
    call("mkdir " + dir_name, shell=True)

def open_avi(filename1, filename2, stack_num, x1, y1, x2, y2, width):
    
    cap_dark = cv2.VideoCapture(filename1)
    cap_light = cv2.VideoCapture(filename2)
    img = []
    
    count = 0

    while count<stack_num:
        _, frame_dark = cap_dark.read()
        _, frame_light = cap_light.read()
        
        frame_dark = frame_dark[x1:x1+460, y1:y1+460]       
        frame_light = frame_light[x2:x2+460, y2:y2+460]
        
        concat_frame = cv2.hconcat([frame_dark,frame_light])
        
        concat_frame = cv2.cvtColor(concat_frame, cv2.COLOR_BGR2GRAY)
        
        img_tmp = np.asarray(concat_frame)
            
        img.append(img_tmp)
        count += 1
        print(count,end=",")            

    print('\n')
    
    return img


def main():

    #Input filename
    dark_img_path = "1832_LD_dark.avi"
    light_img_path = "1832_LD_light.avi"
    
    #Output filename
    filename = "test"
       
    #Define the starting frame number
    initial_frame  = 82

    outdir = "out"
    outdir_img = "img"
    outdir_sum = "sum"
    outdir_crop = "crop"

    last_frame = initial_frame + 1801
    stack_num = last_frame
    
    make_directory(outdir)
    make_directory(outdir_img)
    make_directory(outdir_sum)
    make_directory(outdir_crop)
    
    #Centroid positions
    cx = []
    cy = []
    
    #Open a multitiff
    img = np.array(open_avi(str(dark_img_path), str(light_img_path), stack_num, 11, 136, 11, 161, 460) )   
    
    background_img = cv2.bitwise_not(img[0])
    
    y_pixels, x_pixels= img[0].shape[:3]

    contour_img = img[0].copy()
    line_img = img[0].copy()*0
    heat_img = img[0].copy()*0
    
        
    for i in range(initial_frame, last_frame, 1):
        print("stack_num: " + str(i))
        
        inv_img = cv2.bitwise_not(img[i-1])        
        sub_img =  cv2.subtract(inv_img, background_img)
        
        #Otsu's thresholding
        _, img_th = cv2.threshold(sub_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
      
        #img_th_contours, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_th_contours, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contour = len(img_th_contours)
        
        
        #When a single contour was detected
        if (num_contour == 1):

            contour_img[:] = 0
            cv2.drawContours(contour_img, [img_th_contours[0]], -1, 255, -1)
            cv2.imwrite(outdir_img + "/th_img_" + str(i)  + ".tif", contour_img)
            cv2.imwrite(outdir_crop + "/raw_img_" + str(i)  + ".tif", img[i-1])
            
            
            heat_img = heat_img + contour_img/255
            
            Moment_cnt = cv2.moments(img_th_contours[0])
            cx.append(Moment_cnt['m10']/Moment_cnt['m00'])
            cy.append(Moment_cnt['m01']/Moment_cnt['m00'])

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
            cv2.imwrite(outdir_img + "/th_img_" + str(i)  + ".tif", contour_img)
            cv2.imwrite(outdir_crop + "/raw_img_" + str(i)  + ".tif", img[i-1])
            
            heat_img = heat_img + contour_img/255
            
            Moment_cnt = cv2.moments(img_th_contours[max_j])
            cx.append(Moment_cnt['m10']/Moment_cnt['m00'])
            cy.append(Moment_cnt['m01']/Moment_cnt['m00'])
           
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
    
    state = 0
    distance_cm = [0]*2
    time_s = [0]*2
    output = []
    num_transition = 0
    #0に変更
    previous_state = 0
    latency_to_light = 0
    
    for l in range (0, last_frame - initial_frame - 1, 1):
        #Dark state
        if(cx[l]  < x_pixels*1/2):
            state = 0
            distance = np.sqrt((cx[l+1]-cx[l])*(cx[l+1]-cx[l]) + (cy[l+1]-cy[l])*(cy[l+1]-cy[l]))*40/x_pixels
            distance_cm[0] += distance
            
            
            if (state != previous_state):
                #if(num_transition == 0):
                    #latency_to_light = time_s[1]
                num_transition += 1
                previous_state = 0
            
            time_s[0] += 1/3
        
        #Light state
        else:
            state = 1
            distance = np.sqrt((cx[l+1]-cx[l])*(cx[l+1]-cx[l]) + (cy[l+1]-cy[l])*(cy[l+1]-cy[l]))*40/x_pixels
            distance_cm[1] += distance
            
            if (state != previous_state):
                if(num_transition == 0):
                    latency_to_light = time_s[0]
                
                num_transition += 1
                previous_state = 1
            
            time_s[1] += 1/3
                
                
            
            
        output.append(str(l/3))
        output.append(str(state))
        output.append(str(distance))
        output.append(str(distance*3))
        
    #write scores
    wfilename1 = outdir + "/" + filename + "_timeseries.txt"
    f1 = open(wfilename1, 'w')
        
    for m in range(0, int(len(output)/4), 1):
            f1.write(output[4*m] + '\t' + output[4*m+1] + '\t'+ output[4*m+2] + '\t' + output[4*m+3] + '\n')
    f1.close

    wfilename2 = outdir + "/" + filename + "_summary.txt"
    f2 = open(wfilename2, 'w')
    
    f2.write("Total distance (cm)" + '\t' + "Distance in dark (cm)" + '\t' + "Distance in light (cm)" +  \
             '\t' + "Times in dark (s)" + '\t' + "Times in light (s)" + '\t' + "# of transition" + '\t' + "Latency to light (s)" +'\n' \
             + str(distance_cm[0] + distance_cm[1]) + '\t' + str(distance_cm[0])+ '\t' + str(distance_cm[1])+ \
             '\t' +  str(time_s[0]) +  '\t' +  str(time_s[1]) +  '\t' +  str(num_transition) +  '\t' +  str(latency_to_light))
    
    f2.close
        
    

    
if __name__ == '__main__':
    main()        
