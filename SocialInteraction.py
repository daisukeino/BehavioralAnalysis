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

def mice_identification (positions1, positions2, pre_pos1, pre_pos2):
    
    dis11 = np.sqrt((positions1[0]-pre_pos1[0])*(positions1[0]-pre_pos1[0]) + (positions1[1]-pre_pos1[1])*(positions1[1]-pre_pos1[1]))
    dis21 = np.sqrt((positions1[0]-pre_pos2[0])*(positions1[0]-pre_pos2[0]) + (positions1[1]-pre_pos2[1])*(positions1[1]-pre_pos2[1]))
                
    dis12 = np.sqrt((positions2[0]-pre_pos1[0])*(positions2[0]-pre_pos1[0]) + (positions2[1]-pre_pos1[1])*(positions2[1]-pre_pos1[1]))
    dis22 = np.sqrt((positions2[0]-pre_pos2[0])*(positions2[0]-pre_pos2[0]) + (positions2[1]-pre_pos2[1])*(positions2[1]-pre_pos2[1]))
    
    min_dist = min([dis11, dis12, dis21, dis22])
                
    if(min_dist == dis11 or min_dist == dis22):
        
        return positions1, positions2
                    
    else:       
        
        return positions2, positions1

def main():

    #Input filename
    filename = "3029_3042_SI"
    outdir = "out"
    outdir_img = "img"
    outdir_sum = "sum"
    
    thresh_area = 100
    
    #x_pixels = 238
    #y_pixels = 238
    last_frame = 1980
    initial_frame = last_frame-1751
    stack_num = last_frame
  
    make_directory(outdir)
    make_directory(outdir_img)
    make_directory(outdir_sum)
    
    #Centroid positions
    cx1 = []
    cy1 = []
    
    cx2 = []
    cy2 = []
    
    #Open a multitiff
    img = np.array(open_multitiff(filename, stack_num))
    y_pixels, x_pixels= img[0].shape[:3]
    
    background_img = cv2.bitwise_not(img[0])
    
    contour_img = img[0].copy()
    line_img1 = img[0].copy()*0
    line_img2 = img[0].copy()*0
    heat_img = img[0].copy()*0
    
    
    #state 0: no contact, 1: contact
    state_label = []
    active_contact_label = []
    
    active_state = 0
    
    #number of contacts
    contact_num = 0
    active_contact_num = 0
      
    for i in range(initial_frame, last_frame, 1):
        print("stack_num: " + str(i))
        
        inv_img = cv2.bitwise_not(img[i-1])
        
        sub_img =  cv2.subtract(inv_img, background_img)
        
        # Otsu's thresholding
        _, img_th = cv2.threshold(sub_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
      
        img_th_contours, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contour = len(img_th_contours)
        
        print(len(img_th_contours))
        
        #Delete small noise signals
        del_list = []
            
        for j in range (0, num_contour, 1):
            if (cv2.contourArea(img_th_contours[j]) < thresh_area):
                
                del_list.append(int(j))
            
        if (len(del_list) > 0):        
            for m in range (len(del_list),  0, -1):
                del img_th_contours[del_list[m-1]]
                
        print(len(img_th_contours))
        
        if (len(img_th_contours) > 2):
            
            contour_area = []
            
            for z in range (0, len(img_th_contours), 1):
                contour_area.append(cv2.contourArea(img_th_contours[z]))
                
            while (len(img_th_contours) > 2):
                del_index = [u for u, v in enumerate(contour_area) if v == min(contour_area)]
                print(del_index[0])
                del img_th_contours[del_index[0]]
        
        print(len(img_th_contours))
        
       
        
        #When two mice detected
        if (len(img_th_contours) == 2):

            contour_img[:] = 0
            cv2.drawContours(contour_img, [img_th_contours[0]], -1, 255, -1)
            cv2.drawContours(contour_img, [img_th_contours[1]], -1, 255, -1)
            cv2.imwrite(outdir_img + "/" + str(i)  + ".tif", contour_img)
            
            heat_img = heat_img + contour_img/255
            
            Moment_cnt1 = cv2.moments(img_th_contours[0])
            Moment_cnt2 = cv2.moments(img_th_contours[1])
            
            x_cent1 = Moment_cnt1['m10']/Moment_cnt1['m00']
            y_cent1 = Moment_cnt1['m01']/Moment_cnt1['m00']
            
            x_cent2 = Moment_cnt2['m10']/Moment_cnt2['m00']
            y_cent2 = Moment_cnt2['m01']/Moment_cnt2['m00']
            

            if (i > initial_frame):
                [x_cent1, y_cent1], [x_cent2, y_cent2] = mice_identification ([x_cent1, y_cent1], [x_cent2, y_cent2], [cx1[i - initial_frame-1], cy1[i - initial_frame-1]], [cx2[i - initial_frame-1], cy2[i - initial_frame-1]])
            
            cx1.append(x_cent1)
            cy1.append(y_cent1)
            
            cx2.append(x_cent2)
            cy2.append(y_cent2)
            
            state_label.append(0)
            active_state = 0
            active_contact_label.append(0)
            

        #When multiple contours were detected
        elif (len(img_th_contours) == 1):
            
            contour_img[:] = 0
            cv2.drawContours(contour_img, [img_th_contours[0]], -1, 255, -1)
            cv2.imwrite(outdir_img + "/" + str(i)  + ".tif", contour_img)
            
            heat_img = heat_img + contour_img/255
                      
            
            heat_img = heat_img + contour_img/255
            
            Moment_cnt = cv2.moments(img_th_contours[0])
            
            x_cent = Moment_cnt['m10']/Moment_cnt['m00']
            y_cent = Moment_cnt['m01']/Moment_cnt['m00']
                        
            
            cx1.append(x_cent)
            cy1.append(y_cent)
            
            cx2.append(x_cent)
            cy2.append(y_cent)
            
            state_label.append(1)
            
            #evaluation of active contacts 
            current_num = i - initial_frame
            
            previous_distance = np.sqrt((cx1[current_num-1]-cx2[current_num-1])*(cx1[current_num-1]-cx2[current_num-1]) + (cy1[current_num-1]-cy2[current_num-1])*(cy1[current_num-1]-cy2[current_num-1]))*50/x_pixels
                     
            if (previous_distance > 10):
                active_state = 1
            
            if (active_state == 1):
                active_contact_label.append(1)
            
            else:
                active_contact_label.append(0)
                
           
        else:
            print(str(i))
            print("Failure in mice recognition!")
            sys.exit()
            
    for k in range (0, last_frame - initial_frame - 1, 1):
        line_tmp_img1 = img[0].copy()*0
        cv2.line(line_tmp_img1, (round(cx1[k]), round(cy1[k])), (round(cx1[k+1]), round(cy1[k+1])), color=(10,0,0), thickness=1)
        line_img1 = line_img1 + line_tmp_img1
        
        line_tmp_img2 = img[0].copy()*0
        cv2.line(line_tmp_img2, (round(cx2[k]), round(cy2[k])), (round(cx2[k+1]), round(cy2[k+1])), color=(10,0,0), thickness=1)
        line_img2 = line_img2 + line_tmp_img2
    
    heat_img = heat_img.astype("uint16") 
    cv2.imwrite(outdir_sum + "/" + filename  + "_line1.tif", line_img1)
    cv2.imwrite(outdir_sum + "/" + filename  + "_line2.tif", line_img2)

    cv2.imwrite(outdir_sum + "/" + filename  + "_heat.tif", heat_img)
    
    distance_cm = [0]*3
    time_s = [0]*3
    
    for l in range (0, last_frame - initial_frame - 1, 1):
        if(state_label[l+1] == 0):
            distance1 = np.sqrt((cx1[l+1]-cx1[l])*(cx1[l+1]-cx1[l]) + (cy1[l+1]-cy1[l])*(cy1[l+1]-cy1[l]))*50/x_pixels
            distance2 = np.sqrt((cx2[l+1]-cx2[l])*(cx2[l+1]-cx2[l]) + (cy2[l+1]-cy2[l])*(cy2[l+1]-cy2[l]))*50/x_pixels
            distance_cm[0] += distance1
            distance_cm[0] += distance2
            time_s[0] += 1/3
        
        else:
            distance1 = np.sqrt((cx1[l+1]-cx1[l])*(cx1[l+1]-cx1[l]) + (cy1[l+1]-cy1[l])*(cy1[l+1]-cy1[l]))*50/x_pixels
            distance2 = np.sqrt((cx2[l+1]-cx2[l])*(cx2[l+1]-cx2[l]) + (cy2[l+1]-cy2[l])*(cy2[l+1]-cy2[l]))*50/x_pixels
            distance_cm[1] += distance1
            distance_cm[1] += distance2
            time_s[1] += 1/3
            
            if(state_label[l] == 0):
                contact_num += 1
                
            if(active_contact_label[l+1] == 1):
                distance_cm[2] += distance1
                distance_cm[2] += distance2
                time_s[2] += 1/3
                
                if(active_contact_label[l] == 0):
                    active_contact_num += 1
                              
                    
    wfilename1 = outdir + "/" + filename + "_summary.txt"
    f1 = open(wfilename1, 'w')
    
    f1.write("Total duration of contacts (s)" + '\t' + "Number of contacts" +  \
             '\t' + "Total duration of active contacts (s)" + '\t' + "Number of active contacts" + '\t' + "Total distance (cm)" +  '\n' \
             + str(time_s[1]) + '\t' + str(contact_num)+ '\t' + str(time_s[2])+ \
             '\t' +  str(active_contact_num) +  '\t' +  str(distance_cm[0]+distance_cm[1]) + '\n' )
    
        
      
    

    
if __name__ == '__main__':
    main()        
