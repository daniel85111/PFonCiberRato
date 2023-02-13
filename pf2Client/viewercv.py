from math import *
from lxml import etree
import numpy as np
import cv2
from matplotlib import pyplot as plt

class ViewParticles():
    def __init__(self, size_x, size_y, particles, areas):

        self.particulas = particles
        self.areas = areas

        self.xinit_real, self.yinit_real, self.oriinit_real = self.get_startPos("../Labs/2223-pf/C2-grid.xml")
        
        if len(self.xinit_real) > 1: # Queremos mais do que 1 robot real? (Acho que nao, mas no grid pode existir mais do que 1)
            self.numberofrobots = len(self.xinit_real)
        # Simulator dimensions = 28 x 14 
        # To create a window with size 1120 x 560
        # 1120/28 = 560/14 = 1/40 
        self.imscale = 40

        self.immax_x = self.imscale * size_x
        self.immax_y = self.imscale * size_y
        self.img = np.zeros((self.immax_y,self.immax_x,3), np.uint8)
        
    def updateParticles(self, newParticles):
        self.particulas = newParticles

    def drawParticles(self):
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

            # Drawings
            cv2.circle(self.img, (x,y), 5, (0,255,0), -1)                                       # Draw particle center (Green)
            cv2.line( self.img, (x,y), (x_sensor_centro, y_sensor_centro), (200,150,100),1)     # Draw line from particle center to center line sensor (Grayish)
            
            # cv2.circle(self.img, (x_sensor_centro,y_sensor_centro), 1, (0,0,253), -1)         # Draw sensor Center

            # cv2.circle(self.img, (x_sensor_L1,y_sensor_L1), 1, (0,0,253), -1)                 # Draw sensor Left1
            # cv2.circle(self.img, (x_sensor_L2,y_sensor_L2), 1, (0,0,253), -1)                 # Draw sensor Left2
            # cv2.circle(self.img, (x_sensor_L3,y_sensor_L3), 1, (0,0,253), -1)                 # Draw sensor Left3

            # cv2.circle(self.img, (x_sensor_R1,y_sensor_R1), 1, (0,0,253), -1)                 # Draw sensor Right1
            # cv2.circle(self.img, (x_sensor_R2,y_sensor_R2), 1, (0,0,253), -1)                 # Draw sensor Right2 
            # cv2.circle(self.img, (x_sensor_R3,y_sensor_R3), 1, (0,0,253), -1)                 # Draw sensor Right3



    def drawReal(self,x,y,ori, diameter, DISTsens):
        # Constants and calculation of positions in the cv window
        y_correction = int(14)*self.imscale     # In OpenCV the (0,0) is top-left but from the simulator is bottom-left
        radious = 0.5*diameter                  # Robot real radious
        
        if x == None or y == None or ori == None : return
        if DISTsens == None : DISTsens = (0,0,0,0)

        sensorDIST_apparture = 30
        centerDIST, leftDIST, rightDIST, backDIST = DISTsens
        sensorDIST_center_ori = 0
        sensorDIST_left_ori  = 60
        sensorDIST_right_ori = -60
        sensorDIST_back_ori = 180


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
        sensorDIST_center_posx = cx + self.imscale*0.5*cos(radians(ori))
        sensorDIST_center_posy = cy - self.imscale*0.5*sin(radians(ori))

        sensorDIST_left_posx = cx + self.imscale*0.5*cos(radians(ori+60))
        sensorDIST_left_posy = cy - self.imscale*0.5*sin(radians(ori+60))

        sensorDIST_right_posx = cx + self.imscale*0.5*cos(radians(ori-60))
        sensorDIST_right_posy = cy - self.imscale*0.5*sin(radians(ori-60))

        sensorDIST_back_posx = cx + self.imscale*0.5*cos(radians(ori+180))
        sensorDIST_back_posy = cy - self.imscale*0.5*sin(radians(ori+180))

        sensorDIST_center_endpointleft_x = sensorDIST_center_posx + self.imscale*centerDIST*cos(radians(ori + sensorDIST_center_ori + sensorDIST_apparture))
        sensorDIST_center_endpointleft_y = sensorDIST_center_posy - self.imscale*centerDIST*sin(radians(ori + sensorDIST_center_ori + sensorDIST_apparture))
        sensorDIST_center_endpointcenter_x = sensorDIST_center_posx + self.imscale*centerDIST*cos(radians(ori + sensorDIST_center_ori))
        sensorDIST_center_endpointcenter_y = sensorDIST_center_posy - self.imscale*centerDIST*sin(radians(ori + sensorDIST_center_ori))
        sensorDIST_center_endpointright_x = sensorDIST_center_posx + self.imscale*centerDIST*cos(radians(ori + sensorDIST_center_ori - sensorDIST_apparture))
        sensorDIST_center_endpointright_y = sensorDIST_center_posy - self.imscale*centerDIST*sin(radians(ori + sensorDIST_center_ori - sensorDIST_apparture))

        sensorDIST_left_endpointleft_x = sensorDIST_left_posx + self.imscale*leftDIST*cos(radians(ori + sensorDIST_left_ori + sensorDIST_apparture))
        sensorDIST_left_endpointleft_y = sensorDIST_left_posy - self.imscale*leftDIST*sin(radians(ori + sensorDIST_left_ori + sensorDIST_apparture))
        sensorDIST_left_endpointcenter_x = sensorDIST_left_posx + self.imscale*leftDIST*cos(radians(ori + sensorDIST_left_ori))
        sensorDIST_left_endpointcenter_y = sensorDIST_left_posy - self.imscale*leftDIST*sin(radians(ori + sensorDIST_left_ori))
        sensorDIST_left_endpointright_x = sensorDIST_left_posx + self.imscale*leftDIST*cos(radians(ori + sensorDIST_left_ori - sensorDIST_apparture))
        sensorDIST_left_endpointright_y = sensorDIST_left_posy - self.imscale*leftDIST*sin(radians(ori + sensorDIST_left_ori - sensorDIST_apparture))

        sensorDIST_right_endpointleft_x = sensorDIST_right_posx + self.imscale*rightDIST*cos(radians(ori + sensorDIST_right_ori + sensorDIST_apparture))
        sensorDIST_right_endpointleft_y = sensorDIST_right_posy - self.imscale*rightDIST*sin(radians(ori + sensorDIST_right_ori + sensorDIST_apparture))
        sensorDIST_right_endpointcenter_x = sensorDIST_right_posx + self.imscale*rightDIST*cos(radians(ori + sensorDIST_right_ori))
        sensorDIST_right_endpointcenter_y = sensorDIST_right_posy - self.imscale*rightDIST*sin(radians(ori + sensorDIST_right_ori))
        sensorDIST_right_endpointright_x = sensorDIST_right_posx + self.imscale*rightDIST*cos(radians(ori + sensorDIST_right_ori - sensorDIST_apparture))
        sensorDIST_right_endpointright_y = sensorDIST_right_posy - self.imscale*rightDIST*sin(radians(ori + sensorDIST_right_ori - sensorDIST_apparture))

        sensorDIST_back_endpointleft_x = sensorDIST_back_posx + self.imscale*backDIST*cos(radians(ori + sensorDIST_back_ori + sensorDIST_apparture))
        sensorDIST_back_endpointleft_y = sensorDIST_back_posy - self.imscale*backDIST*sin(radians(ori + sensorDIST_back_ori + sensorDIST_apparture))
        sensorDIST_back_endpointcenter_x = sensorDIST_back_posx + self.imscale*backDIST*cos(radians(ori + sensorDIST_back_ori))
        sensorDIST_back_endpointcenter_y = sensorDIST_back_posy - self.imscale*backDIST*sin(radians(ori + sensorDIST_back_ori))
        sensorDIST_back_endpointright_x = sensorDIST_back_posx + self.imscale*backDIST*cos(radians(ori + sensorDIST_back_ori - sensorDIST_apparture))
        sensorDIST_back_endpointright_y = sensorDIST_back_posy - self.imscale*backDIST*sin(radians(ori + sensorDIST_back_ori - sensorDIST_apparture))

        
        

        # Draws of robot elements
        cv2.circle(self.img,(int(cx),int(cy)), 20, (250,250,250), 2) # Circulo centrado no centro do robot real
        cv2.line( self.img, (int(cx),int(cy)), (int(cx+radious*self.imscale*cos(radians(ori))), int(cy-(radious*self.imscale*sin(radians(ori))))), (255,255,255),1) # Linha do centro do robot direcionada segundo orientaçao
        
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

        cv2.circle(self.img, (int(sensorDIST_center_endpointleft_x),int(sensorDIST_center_endpointleft_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_center_endpointcenter_x),int(sensorDIST_center_endpointcenter_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_center_endpointright_x),int(sensorDIST_center_endpointright_y)), 2, (0,0,253), -1)

        cv2.circle(self.img, (int(sensorDIST_left_endpointleft_x),int(sensorDIST_left_endpointleft_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_left_endpointcenter_x),int(sensorDIST_left_endpointcenter_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_left_endpointright_x),int(sensorDIST_left_endpointright_y)), 2, (0,0,253), -1)

        cv2.circle(self.img, (int(sensorDIST_right_endpointleft_x),int(sensorDIST_right_endpointleft_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_right_endpointcenter_x),int(sensorDIST_right_endpointcenter_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_right_endpointright_x),int(sensorDIST_right_endpointright_y)), 2, (0,0,253), -1)

        cv2.circle(self.img, (int(sensorDIST_back_endpointleft_x),int(sensorDIST_back_endpointleft_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_back_endpointcenter_x),int(sensorDIST_back_endpointcenter_y)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_back_endpointright_x),int(sensorDIST_back_endpointright_y)), 2, (0,0,253), -1)




        #print(f'\nGPS: x: {40*x+40*cos(ori)}   y: {40*y+40*sin(ori)}   theta: {ori}')

    def drawMap(self, map):
        # See Only map
        # print(len(map))
        # b = map.reshape(10*14, 10*28)
        # plt.imshow(b)
        # plt.show()

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
