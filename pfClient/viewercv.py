from math import *
import numpy as np
import cv2

class ViewParticles():
    def __init__(self, size_x, size_y, particles, areas):

        self.particulas = particles
        self.areas = areas

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



    def drawReal(self,x,y,ori, diameter):
        # Constants and calculation of positions in the cv window
        y_correction = int(14)*self.imscale     # In OpenCV the (0,0) is top-left but from the simulator is bottom-left
        radious = 0.5*diameter                  # Robot real radious
        
        if x == None or y == None or ori == None : return
        
        cx = self.imscale*x                     # Center of robot
        cy = abs(y_correction - self.imscale*y)

        # Line Sensors location
        x_sensor_centro = cx + self.imscale*0.438*cos(radians(ori))
        y_sensor_centro = abs(cy - self.imscale*0.438*sin(radians(ori)))

        sensor_L1_posx = x_sensor_centro + self.imscale*3*0.08*cos(radians(ori+90))
        sensor_L1_posy = y_sensor_centro + self.imscale*3*0.08*sin(radians(ori-90))

        sensor_L2_posx = x_sensor_centro + self.imscale*2*0.08*cos(radians(ori+90))
        sensor_L2_posy = y_sensor_centro + self.imscale*2*0.08*sin(radians(ori-90))
        sensor_L3_posx = x_sensor_centro + self.imscale*1*0.08*cos(radians(ori+90))
        sensor_L3_posy = y_sensor_centro + self.imscale*1*0.08*sin(radians(ori-90))

        sensor_R1_posx = x_sensor_centro + self.imscale*3*0.08*cos(radians(ori-90))
        sensor_R1_posy = y_sensor_centro + self.imscale*3*0.08*sin(radians(ori+90))

        sensor_R2_posx = x_sensor_centro + self.imscale*2*0.08*cos(radians(ori-90))
        sensor_R2_posy = y_sensor_centro + self.imscale*2*0.08*sin(radians(ori+90))

        sensor_R3_posx = x_sensor_centro + self.imscale*1*0.08*cos(radians(ori-90))
        sensor_R3_posy = y_sensor_centro + self.imscale*1*0.08*sin(radians(ori+90))

        # Draws of robot elements
        cv2.circle(self.img,(int(cx),int(cy)), 20, (250,250,250), 2) # Circulo centrado no centro do robot real
        cv2.line( self.img, (int(cx),int(cy)), (int(cx+radious*self.imscale*cos(radians(ori))), int(cy-(radious*self.imscale*sin(radians(ori))))), (255,255,255),1) # Linha do centro do robot direcionada segundo orientaçao
        
        cv2.circle(self.img, (int(x_sensor_centro),int(y_sensor_centro)), 2, (0,0,253), -1)
        
        cv2.circle(self.img, (int(sensor_L1_posx),int(sensor_L1_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensor_L2_posx),int(sensor_L2_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensor_L3_posx),int(sensor_L3_posy)), 2, (0,0,253), -1)

        cv2.circle(self.img, (int(sensor_R1_posx),int(sensor_R1_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensor_R2_posx),int(sensor_R2_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensor_R3_posx),int(sensor_R3_posy)), 2, (0,0,253), -1)


        #print(f'\nGPS: x: {40*x+40*cos(ori)}   y: {40*y+40*sin(ori)}   theta: {ori}')

    def drawMap(self):
        # See Only map
        # b = self.map.reshape(10*self.mapmax_y, 10*self.mapmax_x)
        # plt.imshow(b)
        # plt.show()

        for j,area_vertex in enumerate(self.areas):
            cv2.rectangle(self.img,(int(self.imscale*area_vertex[0][0]),int(self.imscale*area_vertex[0][1])),(int(self.imscale*area_vertex[1][0]),int(self.imscale*area_vertex[1][1])),(255,0,0),-1) 



    def showImg(self):
        cv2.imshow("img",self.img)
        cv2.waitKey(1)

    def clearImg(self):
        self.img = np.zeros((560,1120,3), np.uint8)
