import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import os 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics

lastcapimage = 0

os.chdir('C:\\Users\\xxx\\Desktop\\pplcount\\testimages')                #your folder for storing images

lastcal = 0
x_array=[]
y_array=[]
pic = 0
capture_freq = 5                 ####set time between image captures, this is 1 image per 5 secs

while 1:
    if (time.time()-lastcapimage)>capture_freq:  
        tic = time.time()                              
        cam = cv2.VideoCapture(0)
        grabbed, f1 = cam.read()
        f1 = imutils.resize(f1, width=300)
        time.sleep(0.25)   #### time lag between 2 different photos, adjust according to activity
        grabbed, f2 = cam.read()
        f2 = imutils.resize(f2, width=300)
        cam.release()
        pic+= 1
        lastcapimage = time.time()
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(g1, (11, 11), 0)                  ####gaussian filter to smooth out some random noise
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        g2 = cv2.GaussianBlur(g2, (11, 11), 0)
        frameDelta = cv2.absdiff(g1, g2) 
        thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None,iterations=1)         ### shrink out small noises
        thresh = cv2.dilate(thresh, None, iterations=7)         ### how much to magnify differences between frames
        (_,cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        f3 = f2.copy()
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            if len(approx)==4:
                continue
            if cv2.contourArea(c)<float(150):
                continue
            (x,y),radius = cv2.minEnclosingCircle(c)
            x_array.append(x)
            y_array.append(y)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(f3,center,radius,(0,255,0),2)
        #cv2.putText(f3, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, f3.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        print('time processing pics: ',time.time()-tic)
        cv2.imshow('f1', f1)
        cv2.imshow('f2', f2)
        cv2.imshow('f3',f3)
        cv2.imshow('thresh', thresh)
        cv2.imshow('delta', frameDelta)
#        cv2.imwrite(datetime.datetime.now().strftime("%H%M%S")+"f1.jpg", f1)
#        cv2.imwrite(datetime.datetime.now().strftime("%H%M%S")+"f2.jpg", f2)
#        cv2.imwrite("f3.jpg", f3)
        cv2.waitKey(1000*capture_freq-1000) 
        cv2.destroyAllWindows()
        lastcapimage = time.time()
    
    if pic>=10:                       ###after 10 pictures have been collected
        tic2 = time.time()
        maxppl = min(len(x_array),10)           ###select max ppl possible
        labels = []
        costs = []                              ###dist from datapoints in cluster to centroid
        data = list(zip(x_array,y_array))
        CHindex = []                                    ###CH index to decide number of ppl
        for k in range(1,maxppl+1):                                         ###try differrent clusters
            kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data)
            labels.append(kmeans_model.labels_)
            costs.append(kmeans_model.inertia_)
            if k > 2:
                CHindex.append(metrics.calinski_harabaz_score(data, kmeans_model.labels_))
            print("k:",k, " cost:", kmeans_model.inertia_)
        
        dist = []                                           ###elbow method to decide number of ppl
        
        for i in range(0,maxppl):
            a = np.linalg.norm(np.array([10,costs[9]])-np.array([1,costs[0]]))
            b = np.linalg.norm(np.array([i,costs[i]])-np.array([1,costs[0]]))
            c=  np.linalg.norm(np.array([10,costs[9]])-np.array([i,costs[i]]))      
            s= 0.5*(a+b+c)
            area = (s*(s-a)*(s-b)*(s-c))**0.5
            dist.append(2*area/a)
        
        print('est num of ppl:',dist.index(max(dist))+1,' by elbow ',CHindex.index(max(CHindex))+2, ' by CHindex' )
        print('time processing kmeans: ',time.time()-tic2)
        plt.figure(1)
        plt.clf()
        plt.title('cost vs k')
        plt.plot(range(1,maxppl+1),costs)
        plt.show()
        plt.figure(2)
        plt.clf()
        plt.imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB),zorder=1)
        plt.scatter(x_array,y_array, c=[matplotlib.cm.spectral(float(i) /maxppl) for i in labels[dist.index(max(dist))]],zorder=2);
        x_array = []                ### clear points for next clustering
        y_array = []   
        pic=0
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        