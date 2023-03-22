from math import *
from lxml import etree
import numpy as np
import cv2
from matplotlib import pyplot as plt

class ViewParticles():
    def __init__(self, map, grid_directory, particles = None, areas = None, img_scale = 40):
        self.map = map.getMapCropped()

        if areas == None:
            self.areas = map.areas
        else:
            self.areas = areas

        self.particulas = particles

        self.xinit_real, self.yinit_real, self.oriinit_real = self.get_startPos(grid_directory)
       
        self.imscale = img_scale

        self.immax_x = self.imscale * map.mapmax_x
        self.immax_y = self.imscale * map.mapmax_y
        self.img = np.zeros((self.immax_y,self.immax_x,3), np.uint8)        

    def drawParticles(self, newParticles, max_weight):
        self.particulas = newParticles
        # Uncomment  other commented lines to draw all particle line sensors


        for i,particula in enumerate(self.particulas):
            # Calculation of positions in the cv window
            x = int(self.imscale*particula.x)
            y = int(self.imscale*particula.y)

            x_sensor_centro = int(self.imscale*particula.sensor_center_posx)
            y_sensor_centro = int(self.imscale*particula.sensor_center_posy)

            # x_sensor_L1 =  int(self.imscale*particula.sensor_L1_posx)
            # y_sensor_L1 =  int(self.imscale*particula.sensor_L1_posy)

            # x_sensor_L2 =  int(self.imscale*particula.sensor_L2_posx)
            # y_sensor_L2 =  int(self.imscale*particula.sensor_L2_posy)

            # x_sensor_L3 =  int(self.imscale*particula.sensor_L3_posx)
            # y_sensor_L3 =  int(self.imscale*particula.sensor_L3_posy)

            # x_sensor_R1 =  int(self.imscale*particula.sensor_R1_posx)
            # y_sensor_R1 =  int(self.imscale*particula.sensor_R1_posy)

            # x_sensor_R2 =  int(self.imscale*particula.sensor_R2_posx)
            # y_sensor_R2 =  int(self.imscale*particula.sensor_R2_posy)

            # x_sensor_R3 =  int(self.imscale*particula.sensor_R3_posx)
            # y_sensor_R3 =  int(self.imscale*particula.sensor_R3_posy)

            # Color based on weight
            if max_weight-1 == 0:
                color = 255
            else:
                color = ((particula.weight-1)/(max_weight-1))*255

            # Drawings
            cv2.circle(self.img, (x,y), 5, (0,color,255-color), -1)                                       # Draw particle center (Green)
            cv2.line( self.img, (x,y), (x_sensor_centro, y_sensor_centro), (200,150,100),1)     # Draw line from particle center to center line sensor (Grayish)
            
            # cv2.circle(self.img, (x_sensor_centro,y_sensor_centro), 1, (0,0,253), -1)         # Draw sensor Center

            # cv2.circle(self.img, (x_sensor_L1,y_sensor_L1), 1, (0,0,253), -1)                 # Draw sensor Left1
            # cv2.circle(self.img, (x_sensor_L2,y_sensor_L2), 1, (0,0,253), -1)                 # Draw sensor Left2
            # cv2.circle(self.img, (x_sensor_L3,y_sensor_L3), 1, (0,0,253), -1)                 # Draw sensor Left3

            # cv2.circle(self.img, (x_sensor_R1,y_sensor_R1), 1, (0,0,253), -1)                 # Draw sensor Right1
            # cv2.circle(self.img, (x_sensor_R2,y_sensor_R2), 1, (0,0,253), -1)                 # Draw sensor Right2 
            # cv2.circle(self.img, (x_sensor_R3,y_sensor_R3), 1, (0,0,253), -1)                 # Draw sensor Right3

    def drawCentroides(self, centroides, oris, weights):
         # Draws of robot elements
        max_weight = max(weights)
        if len(centroides) < 1: return
        for i,centroide in enumerate(centroides):
            

            x = int(self.imscale*centroide[0])
            y = int(self.imscale*centroide[1])
            if x == None or y == None: return
            radious = 0.25
            if weights[i] == max_weight:
                cv2.circle(self.img,(int(x),int(y)), 10, (0,250,255), -1) # Circulo centrado no centro do robot real
            else:
                cv2.circle(self.img,(int(x),int(y)), 10, (200,0,0), -1) # Circulo centrado no centro do robot real
            cv2.line( self.img, (int(x),int(y)), (int(x+radious*self.imscale*cos(oris[i])), int(y-(radious*self.imscale*sin(oris[i])))), (0,0,255),2) # Linha do centro do robot direcionada segundo orientaçao
        

    def drawFinalPose(self,final_pose):
         # Draws of robot elements
        x,y,ori = final_pose
        x = x*self.imscale
        y = y*self.imscale
        radious = 0.25
        cv2.circle(self.img,(int(x),int(y)), 10, (200,0,0), -1) # Circulo centrado no centro do robot real
        cv2.line( self.img, (int(x),int(y)), (int(x+radious*self.imscale*cos(ori)), int(y-(radious*self.imscale*sin(ori)))), (0,0,255),2) # Linha do centro do robot direcionada segundo orientaçao
       

    def drawReal(self,x,y,ori, diameter, DISTsens, IRangles, num_endpoints):
        # Constants and calculation of positions in the cv window
        y_correction = int(14)*self.imscale     # In OpenCV the (0,0) is top-left but from the simulator is bottom-left
        radious = 0.5*diameter                  # Robot real radious
        
        if x == None or y == None or ori == None : return
        if DISTsens == None : DISTsens = (0,0,0,0)

        sensorDIST_apparture = radians(30)
        centerDIST, leftDIST, rightDIST, backDIST = DISTsens
        sensorDIST_center_ori = IRangles[0]
        sensorDIST_left_ori  = IRangles[1]
        sensorDIST_right_ori = IRangles[2]
        sensorDIST_back_ori = IRangles[3]


        cx = self.imscale*(x+self.xinit_real[0])                     # Center of robot
        cy = abs(y_correction - self.imscale*(y+self.yinit_real[0]))

        # Line Sensors location
        # x_sensor_centro = cx + self.imscale*0.438*cos(radians(ori))
        # y_sensor_centro = abs(cy - self.imscale*0.438*sin(radians(ori)))

        # sensor_L1_posx = x_sensor_centro + self.imscale*3*0.08*cos(radians(ori+90))
        # sensor_L1_posy = y_sensor_centro + self.imscale*3*0.08*sin(radians(ori-90))

        # sensor_L2_posx = x_sensor_centro + self.imscale*2*0.08*cos(radians(ori+90))
        # sensor_L2_posy = y_sensor_centro + self.imscale*2*0.08*sin(radians(ori-90))

        # sensor_L3_posx = x_sensor_centro + self.imscale*1*0.08*cos(radians(ori+90))
        # sensor_L3_posy = y_sensor_centro + self.imscale*1*0.08*sin(radians(ori-90))

        # sensor_R1_posx = x_sensor_centro + self.imscale*3*0.08*cos(radians(ori-90))
        # sensor_R1_posy = y_sensor_centro + self.imscale*3*0.08*sin(radians(ori+90))

        # sensor_R2_posx = x_sensor_centro + self.imscale*2*0.08*cos(radians(ori-90))
        # sensor_R2_posy = y_sensor_centro + self.imscale*2*0.08*sin(radians(ori+90))

        # sensor_R3_posx = x_sensor_centro + self.imscale*1*0.08*cos(radians(ori-90))
        # sensor_R3_posy = y_sensor_centro + self.imscale*1*0.08*sin(radians(ori+90))
        
        # Distance Sensors location
        endpoints_angle = (2*sensorDIST_apparture)/(num_endpoints-1)

        sensorDIST = np.array([[angle, cx + self.imscale*0.5*cos(ori+angle), cy - self.imscale*0.5*sin(ori+angle)] for i, angle in enumerate(IRangles)])
        for j,v in enumerate(IRangles):
            leitura = DISTsens[j]
            for k in range(num_endpoints):
                angle = endpoints_angle*k
                posx = sensorDIST[j][1] + self.imscale*leitura*cos(ori + sensorDIST[j][0] - sensorDIST_apparture + angle)
                posy = sensorDIST[j][2] - self.imscale*leitura*sin(ori +  sensorDIST[j][0] - sensorDIST_apparture + angle)
                
                if j%3==0:
                    cv2.circle(self.img, (int(posx),int(posy)), 2, (0,255,255), -1)
                else:
                    cv2.circle(self.img, (int(posx),int(posy)), 2, (0,0,255), -1)



        sensorDIST_center_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_center_ori)
        sensorDIST_center_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_center_ori)

        sensorDIST_left_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_left_ori)
        sensorDIST_left_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_left_ori)

        sensorDIST_right_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_right_ori)
        sensorDIST_right_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_right_ori)

        sensorDIST_back_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_back_ori)
        sensorDIST_back_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_back_ori)

        
        

        # Draws of robot elements
        cv2.circle(self.img,(int(cx),int(cy)), 20, (250,250,250), 2) # Circulo centrado no centro do robot real
        cv2.line( self.img, (int(cx),int(cy)), (int(cx+radious*self.imscale*cos(ori)), int(cy-(radious*self.imscale*sin(ori)))), (255,255,255),2) # Linha do centro do robot direcionada segundo orientaçao
        
        # cv2.circle(self.img, (int(x_sensor_centro),int(y_sensor_centro)), 2, (0,0,253), -1)
        
        # cv2.circle(self.img, (int(sensor_L1_posx),int(sensor_L1_posy)), 2, (0,0,253), -1)
        # cv2.circle(self.img, (int(sensor_L2_posx),int(sensor_L2_posy)), 2, (0,0,253), -1)
        # cv2.circle(self.img, (int(sensor_L3_posx),int(sensor_L3_posy)), 2, (0,0,253), -1)

        # cv2.circle(self.img, (int(sensor_R1_posx),int(sensor_R1_posy)), 2, (0,0,253), -1)
        # cv2.circle(self.img, (int(sensor_R2_posx),int(sensor_R2_posy)), 2, (0,0,253), -1)
        # cv2.circle(self.img, (int(sensor_R3_posx),int(sensor_R3_posy)), 2, (0,0,253), -1)

        
        cv2.circle(self.img, (int(sensorDIST_left_posx),int(sensorDIST_left_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_center_posx),int(sensorDIST_center_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_right_posx),int(sensorDIST_right_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_back_posx),int(sensorDIST_back_posy)), 2, (0,0,253), -1)



        #print(f'\nGPS: x: {40*x+40*cos(ori)}   y: {40*y+40*sin(ori)}   theta: {ori}')

    def drawMap(self, map):
        for j,area_vertex in enumerate(self.areas):
            cv2.rectangle(self.img,(int(self.imscale*area_vertex[0][0]),int(self.imscale*area_vertex[0][1])),(int(self.imscale*area_vertex[1][0]),int(self.imscale*area_vertex[1][1])),(255,0,0),-1)
        

    def showImg(self):
        cv2.imshow("img",self.img)
        cv2.waitKey(1)

    def clearImg(self):
        self.img = np.zeros((560,1120,3), np.uint8)

    def get_startPos(self,xmlFile):
        """
        Parse the XML
        """
        dic = []
        
        with open(xmlFile) as fobj:
            xml = fobj.read()
        root = etree.fromstring(xml)
        
        for grid in root.getchildren():
            if(grid.tag=="Position"):
                dic.append(grid.attrib)
        
        x = []
        y = []
        ori = []

        for i,v in enumerate(dic):
            x.append(float(v["X"]))
            y.append(float(v["Y"]))
            ori.append(float(v["Dir"]))

        return (x,y,ori) 
