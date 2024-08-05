from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename

from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
from RadarSim import *
from math import sqrt
import webbrowser

main = tkinter.Tk()
main.title("Enhancing Sensor Attack Detection and Isolation for Autonomous Vehicles")
main.geometry("1300x1200")

global gps_filename, lidar_filename
gps_locations = []
gps_xs = []
gps_track = []

lidar_locations = []
lidar_xs = []
lidar_track = []

gps_graph = []
lidar_graph = []
global output

def uploadGPS():
    global gps_filename
    text.delete('1.0', END)
    gps_filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.insert(END,gps_filename+" loaded\n")

def uploadLidar():
    global lidar_filename
    text.delete('1.0', END)
    lidar_filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.insert(END,lidar_filename+" loaded\n")
    
def HJacobian_at(x):
    horiz_dist = x[0]
    altitude   = x[2]
    denom = sqrt(horiz_dist**2 + altitude**2)
    return array ([[horiz_dist/denom, 0., altitude/denom]])

def hx(x):
    return (x[0]**2 + x[2]**2) ** 0.5

def GPSEKF():
    global gps_filename, gps_locations, gps_xs, gps_track
    text.delete('1.0', END)
    gps_locations.clear()
    gps_xs.clear()
    gps_track.clear()
    with open(gps_filename, "r") as file: #reading latitude and longitude values from dataset
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            arr = line.split(",")
            gps_locations.append([float(arr[0]), float(arr[1])])#adding all vehicle locations to array variable
    file.close()
    dt = 0.05
    rk = ExtendedKalmanFilter(dim_x=3, dim_z=1) #Defining extended kalman filter object 
    radar = RadarSim(dt, pos=0., vel=gps_locations[0][0], alt=gps_locations[0][1])#adding inital vehcile location to radar object
    rk.x = array([radar.pos,gps_locations[0][0],gps_locations[0][1]]) #adding initial location to rk object of extended kalman filter
    rk.F = eye(3) + array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]])*dt
    rk.R = radar.alt * 0.05 # 5% of distance
    rk.Q = array([[0, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) * 0.001
    rk.P *= 50
    for i in range(len(gps_locations)): #now loop all GPS locations
        z = radar.get_range() #find the next location range
        gps_track.append((radar.pos, gps_locations[i][0], gps_locations[i][1])) #add current location as tracker
        rk.update(array([z]), HJacobian_at, hx) #update new z location EFK
        gps_xs.append(rk.x)
        rk.predict() #now EKF will predict new location
    xs = asarray(gps_xs)
    track = asarray(gps_track)
    for i in range(len(xs)):
        predict_x = xs[i,1]
        predict_y = xs[i,2]
        track_x = track[i,1]
        track_y = track[i,2]
        text.insert(END,"Original GPS Latitude  : "+str(track_x)+"\n")
        text.insert(END,"Original GPS Longitude : "+str(track_y)+"\n")
        text.insert(END,"EKF Predicted Latitude : "+str(predict_x)+"\n")
        text.insert(END,"EKF Predicted Longitude: "+str(predict_y)+"\n\n")


def LidarEKF():
    global lidar_filename, lidar_locations, lidar_xs, lidar_track
    text.delete('1.0', END)
    lidar_locations.clear()
    lidar_xs.clear()
    lidar_track.clear()
    with open(lidar_filename, "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            arr = line.split(",")
            lidar_locations.append([float(arr[0]), float(arr[1])])
    file.close()
    dt = 0.05
    rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)
    radar = RadarSim(dt, pos=0., vel=lidar_locations[0][0], alt=lidar_locations[0][1])
    rk.x = array([radar.pos,gps_locations[0][0],gps_locations[0][1]])
    rk.F = eye(3) + array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]])*dt
    rk.R = radar.alt * 0.05 # 5% of distance
    rk.Q = array([[0, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) * 0.001
    rk.P *= 50
    for i in range(len(lidar_locations)):
        z = radar.get_range()
        lidar_track.append((radar.pos, lidar_locations[i][0], lidar_locations[i][1]))
        rk.update(array([z]), HJacobian_at, hx)
        lidar_xs.append(rk.x)
        rk.predict()
    xs = asarray(lidar_xs)
    track = asarray(lidar_track)
    for i in range(len(xs)):
        predict_x = xs[i,1]
        predict_y = xs[i,2]
        track_x = track[i,1]
        track_y = track[i,2]
        text.insert(END,"Original Lidar Latitude  : "+str(track_x)+"\n")
        text.insert(END,"Original Lidar Longitude : "+str(track_y)+"\n")
        text.insert(END,"EKF Predicted Latitude : "+str(predict_x)+"\n")
        text.insert(END,"EKF Predicted Longitude: "+str(predict_y)+"\n\n")


def runDetector():
    global output
    global gps_graph, lidar_graph
    global lidar_filename, lidar_locations, lidar_xs, lidar_track
    global gps_filename, gps_locations, gps_xs, gps_track
    gps_graph.clear()
    lidar_graph.clear()
    output = "<html><border><center><table border=1 align=center><tr><th>Detector 1</th><th>Detector 2</th><th>Result</th></tr>"
    text.delete('1.0', END)
    gps_attack = 0
    lidar_attack = 0
    for i in range(len(gps_xs)):
        gps_predict = np.cumsum(gps_xs[i]) #calling cumsum on gps predicted data
        gps_original = np.cumsum(gps_track[i])#calling cumsum on gps original data
        lidar_predict = np.cumsum(lidar_xs[i])#calling cumsum on lidar predicted data
        lidar_original = np.cumsum(lidar_track[i])#calling cumsum on lidaroriginal data
        gps_variation = np.sum(gps_predict) - np.sum(gps_original) #finding cumsum gps variation 
        lidar_variation = np.sum(lidar_predict) - np.sum(lidar_original) #finding cumsum lidar variation
        print(str(gps_variation)+" "+str(lidar_variation))
        if gps_variation < 0 and lidar_variation < 0: #now here run the rules to detect or raise alarm
            output += "<tr><td>No Alarm</td><td>No Alarm</td><td>No Alarm</td></tr>"
            text.insert(END,"No Alarm Detected by GPS & Lidar Detectors\n")
        elif gps_variation > 0 and lidar_variation > 0:
            output += "<tr><td>GPS Alarm</td><td>Lidar Alarm</td><td>GPS & Lidar Attack</td></tr>"
            text.insert(END,"Alarm Detected by GPS & Lidar Detectors\n")
            gps_attack += 1
            lidar_attack += 1
        elif gps_variation > 0 and lidar_variation < 0:
            output += "<tr><td>GPS Alarm</td><td>No Alarm</td><td>GPS Attack</td></tr>"
            text.insert(END,"Alarm Detected by GPS Detectors\n")
            gps_attack += 1
        elif gps_variation < 0 and lidar_variation > 0:
            output += "<tr><td>No Alarm</td><td>Lidar Alarm</td><td>Lidar Attack</td></tr>"
            text.insert(END,"Alarm Detected by Lidar Detectors\n")
            lidar_attack += 1    
        if gps_attack > 0:
            gps_graph.append((i,gps_attack))
        if lidar_attack > 0:
            lidar_graph.append((i,lidar_attack))
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()            
            
            
       
def runGraph():
    global gps_graph, lidar_graph
    gps_X = []
    gps_Y = []
    lidar_X = []
    lidar_Y = []
    for i in range(len(gps_graph)):
        gps_X.append(gps_graph[i][0])
        gps_Y.append(gps_graph[i][1])
    for i in range(len(lidar_graph)):
        lidar_X.append(lidar_graph[i][0])
        lidar_Y.append(lidar_graph[i][1])
    webbrowser.open("output.html",new=1)      
    plt.plot(gps_Y, color = 'red', label = 'GPS Attack')
    plt.plot(lidar_Y, color = 'green', label = 'Lidar Attack')
    plt.title('GPS & Lidar Attack Detection Graph')
    plt.xlabel('Time')
    plt.ylabel('Detection Count')
    plt.legend()
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Enhancing Sensor Attack Detection and Isolation for Autonomous Vehicles')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadgps = Button(main, text="Upload GPS Dataset", command=uploadGPS)
uploadgps.place(x=50,y=100)
uploadgps.config(font=font1)  

uploadlidarButton = Button(main, text="Upload LiDAR Dataset", command=uploadLidar)
uploadlidarButton.place(x=280,y=100)
uploadlidarButton.config(font=font1) 

gpsekfButton = Button(main, text="Run GPS Extended Kalman Filter", command=GPSEKF)
gpsekfButton.place(x=500,y=100)
gpsekfButton.config(font=font1) 

lidarekfButton = Button(main, text="Run LiDAR Extended Kalman Filter", command=LidarEKF)
lidarekfButton.place(x=50,y=150)
lidarekfButton.config(font=font1)

cusumButton = Button(main, text="Run Rule Based CUSUM Detector", command=runDetector)
cusumButton.place(x=380,y=150)
cusumButton.config(font=font1)

graphButton = Button(main, text="Attack Detection Graph", command=runGraph)
graphButton.place(x=720,y=150)
graphButton.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
