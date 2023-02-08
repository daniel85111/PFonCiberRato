#import particles
import viewercv
import numpy as np
from matplotlib import pyplot as plt
from math import *
from lxml import etree
import random
import cv2

class particula():
    def __init__(self, x, y, ori, w):
        self.x = x
        self.y = y
        self.ori = ori
        self.weight = w
        self.sensor_center_posx = self.x + 0.438*cos(radians(ori))
        self.sensor_center_posy = self.y - 0.438*sin(radians(ori))

        self.sensorDIST_center_posx = self.x + 0.5*cos(radians(ori))
        self.sensorDIST_center_posy = self.y - 0.5*sin(radians(ori))

        self.sensorDIST_left_posx = self.x + 0.5*cos(radians(ori+60))
        self.sensorDIST_left_posy = self.y - 0.5*sin(radians(ori+60))

        self.sensorDIST_right_posx = self.x + 0.5*cos(radians(ori-60))
        self.sensorDIST_right_posy = self.y - 0.5*sin(radians(ori-60))

        self.sensorDIST_back_posx = self.x + 0.5*cos(radians(ori+180))
        self.sensorDIST_back_posy = self.y - 0.5*sin(radians(ori+180))

        self.sensor_L1_posx = self.sensor_center_posx + 3*0.08*cos(radians(ori+90))
        self.sensor_L1_posy = self.sensor_center_posy + 3*0.08*sin(radians(ori-90))

        self.sensor_L2_posx = self.sensor_center_posx + 2*0.08*cos(radians(ori+90))
        self.sensor_L2_posy = self.sensor_center_posy + 2*0.08*sin(radians(ori-90))

        self.sensor_L3_posx = self.sensor_center_posx + 1*0.08*cos(radians(ori+90))
        self.sensor_L3_posy = self.sensor_center_posy + 1*0.08*sin(radians(ori-90))

        self.sensor_R1_posx = self.sensor_center_posx + 3*0.08*cos(radians(ori-90))
        self.sensor_R1_posy = self.sensor_center_posy + 3*0.08*sin(radians(ori+90))

        self.sensor_R2_posx = self.sensor_center_posx + 2*0.08*cos(radians(ori-90))
        self.sensor_R2_posy = self.sensor_center_posy + 2*0.08*sin(radians(ori+90))

        self.sensor_R3_posx = self.sensor_center_posx + 1*0.08*cos(radians(ori-90))
        self.sensor_R3_posy = self.sensor_center_posy + 1*0.08*sin(radians(ori+90))

class filtroParticulas():
    def __init__(self,n_part=4000, mapmax_x=28, mapmax_y=14):
        self.n_part = n_part
        self.sum_weights = n_part
        # self.max_w = 1
        self.last_motors = (0,0)
        self.sum_squarednormalized_weights = 0
        self.particulas = []
        self.mapmax_x = mapmax_x
        self.mapmax_y = mapmax_y
        self.norm_weights = []
        self.areas = self.getAreas()
        
        self.map, self.map_cropped, self.map_scale_factor = self.getMap()

        self.distance_map_cropped, self.distance_map_full = self.getDistanceMap()
        np.savetxt('Dist MAP', self.distance_map_cropped, fmt='%2.1f', delimiter=', ')

        #print(self.map)

        for i in range (self.n_part):
            # Orientacao random
            # self.particulas.append( particula( random.random() * (mapmax_x), random.random() * (mapmax_y), random.random()*360, 1))

            # Orientacao 0
            self.particulas.append( particula( random.random() * (mapmax_x), random.random() * (mapmax_y), 0, 1))


            #self.particulas.append((random.random() * ((mapmax_x-1)+0.5), random.random() * (mapmax_y-1)+0.5, random.random()*360 - 180, 1))
            #self.particulas.append((5, 8, 0))
            
            #self.weights.append(1)
            self.norm_weights.append(1/self.n_part)
            #self.ori.append(self.particulas[i][-1])
            #self.ori.append(0)

        self.biewer = viewercv.ViewParticles(self.mapmax_x, self.mapmax_y, self.particulas, self.areas)

    def parseXML(self,xmlFile):
        """
        Parse the XML
        """
        dic = []
        with open(xmlFile) as fobj:
            xml = fobj.read()
        root = etree.fromstring(xml)
        
        for labs in root.getchildren():
            if(labs.tag=="Wall"):
                for elem in labs.getchildren():
                    #print(elem.tag=="Corner")
                    if(elem.tag=="Corner"):
                        dic.append(elem.attrib)
        #print(dic)
        return dic

    def getAreas(self):
        array = self.parseXML("../Labs/2223-pf/C2-lab.xml")
        areas = []
        for i,v in enumerate(array):
            if i%4 == 0:
                if i != 0:
                    areas.append([[minx,14-maxy],[maxx,14-miny]])
                minx = float(v['X'])
                miny = float(v['Y'])
                maxx = float(v['X'])
                maxy = float(v['Y'])

            if float(v['X']) < minx: minx = float(v['X'])
            if float(v['Y']) < miny: miny = float(v['Y'])
            if float(v['X']) > maxx: maxx = float(v['X'])
            if float(v['Y']) > maxy: maxy = float(v['Y'])
            
            if i == len(array)-1:
                areas.append([[minx,14-maxy],[maxx,14-miny]]) 
                #print(areas)
        #print(areas[1][1])
        return areas    # ([minx,miny], [maxx,maxy])

    def getMap(self):
        scale = 100

        mapstartx = scale*self.mapmax_x
        mapstarty = scale*self.mapmax_y
        mapendx = 2*scale*self.mapmax_x
        mapendy = 2*scale*self.mapmax_y

        topleft = (mapstartx,mapstarty)
        topright = (mapendx,mapstarty)
        bottomright = (mapendx, mapendy)
        bottomleft = (mapstartx, mapendy)

        arr = np.zeros((3*mapstarty,3*mapstartx,1), np.uint8)

        cv2.line(arr, (topleft), (topright),255,1)
        cv2.line(arr, (topright), (bottomright),255,1)
        cv2.line(arr, (bottomright), (bottomleft),255,1)       
        cv2.line(arr, (bottomleft), (topleft),255,1)
        
        for i,v in enumerate(self.areas): # v[0][0] = xmin v[0][1] = ymin
            # print(f'i = {i}\t v= {v}')
            xmin = int(scale*v[0][0])
            ymin = int(scale*v[0][1])
            xmax = int(scale*v[1][0])
            ymax = int(scale*v[1][1])
            # print(f' xmin= {xmin}\t ymin= {ymin}\t xmax= {xmax}\t ymax= {ymax}\t')
            area_topleft     =  (mapstartx+xmin, mapstarty+ymin)
            area_topright    =  (mapstartx+xmax, mapstarty+ymin)
            area_bottomright =  (mapstartx+xmax, mapstarty+ymax)
            area_bottomleft  =  (mapstartx+xmin, mapstarty+ymax)

            cv2.line(arr, (area_topleft), (area_topright), 255,1)
            cv2.line(arr, (area_topright), (area_bottomright), 255,1)
            cv2.line(arr, (area_bottomright), (area_bottomleft), 255,1)
            cv2.line(arr, (area_bottomleft), (area_topleft), 255,1)

        # cv2.imshow("Resized image", arr)
        # cv2.waitKey(0)  
        # plt.imshow(arr)
        # plt.show() 
       
        cropped = arr[mapstarty:mapendy+1, mapstartx:mapendx+1]
        # cv2.imshow("Resized image", cropped)
        # cv2.waitKey(0) ^
        # plt.imshow(cropped)
        # plt.show() 
        return arr,cropped,scale

    def getDistanceMap(self):
        cropped = self.map_cropped
        cropped = cv2.bitwise_not(cropped)
        cropped = cropped.astype(np.uint8)

        full = self.map
        full = cv2.bitwise_not(full)
        full = full.astype(np.uint8)
        

        distmap_cropped = cv2.distanceTransform(cropped, cv2.DIST_L2, 5)
        distmap_full = cv2.distanceTransform(full, cv2.DIST_L2, 5)

        # plt.imshow(distmap_cropped)
        # plt.show()
        # plt.imshow(distmap_full)
        # plt.show() 
        return distmap_cropped, distmap_full

    def odometry_move_particles(self, motors, motors_noise):       
        self.motors = motors
        noite_multiplier = 2
        for i,particula in enumerate (self.particulas):
            # calculate estimated power apply
            out_l = (self.motors[0] + self.last_motors[0]) / 2
            out_r = (self.motors[1] + self.last_motors[1]) / 2
            
            out_l = random.gauss(out_l, noite_multiplier*motors_noise*out_l)   # out_l tem um erro de 1,5%
            out_r = random.gauss(out_r, noite_multiplier*motors_noise*out_r)    # out_r tem um erro de 1,5%

            if out_l > 0.15:
                out_l = 0.15

            if out_r > 0.15:
                out_r = 0.15
            
            # pos
            lin = (out_l + out_r) / 2
            x = particula.x + (lin * cos(radians(particula.ori)))
            y = particula.y - (lin * sin(radians(particula.ori)))

            rot = out_r - out_l # / self.robot_diameter ( = 1 )
            ori = degrees(radians(particula.ori) + rot) % 360


            particula.x = x
            particula.y = y
            particula.ori = ori

            particula.sensor_center_posx = x + 0.438*cos(radians(ori))
            particula.sensor_center_posy = y - 0.438*sin(radians(ori))

            particula.sensorDIST_center_posx = particula.x + 0.5*cos(radians(ori))
            particula.sensorDIST_center_posy = particula.y - 0.5*sin(radians(ori))

            particula.sensorDIST_left_posx = particula.x + 0.5*cos(radians(ori-60))
            particula.sensorDIST_left_posy = particula.y - 0.5*sin(radians(ori-60))

            particula.sensorDIST_right_posx = particula.x + 0.5*cos(radians(ori+60))
            particula.sensorDIST_right_posy = particula.y - 0.5*sin(radians(ori+60))

            particula.sensorDIST_back_posx = particula.x + 0.5*cos(radians(ori+180))
            particula.sensorDIST_back_posy = particula.y - 0.5*sin(radians(ori+180))

            particula.sensor_L1_posx = particula.sensor_center_posx + 3*0.08*cos(radians(ori+90))
            particula.sensor_L1_posy = particula.sensor_center_posy + 3*0.08*sin(radians(ori-90))

            particula.sensor_L2_posx = particula.sensor_center_posx + 2*0.08*cos(radians(ori+90))
            particula.sensor_L2_posy = particula.sensor_center_posy + 2*0.08*sin(radians(ori-90))

            particula.sensor_L3_posx = particula.sensor_center_posx + 1*0.08*cos(radians(ori+90))
            particula.sensor_L3_posy = particula.sensor_center_posy + 1*0.08*sin(radians(ori-90))

            particula.sensor_R1_posx = particula.sensor_center_posx + 3*0.08*cos(radians(ori-90))
            particula.sensor_R1_posy = particula.sensor_center_posy + 3*0.08*sin(radians(ori+90))

            particula.sensor_R2_posx = particula.sensor_center_posx + 2*0.08*cos(radians(ori-90))
            particula.sensor_R2_posy = particula.sensor_center_posy + 2*0.08*sin(radians(ori+90))

            particula.sensor_R3_posx = particula.sensor_center_posx + 1*0.08*cos(radians(ori-90))
            particula.sensor_R3_posy = particula.sensor_center_posy + 1*0.08*sin(radians(ori+90))
            

            self.last_motors = (out_l,out_r)


    def resample(self):
        n = 0.9*self.n_part
        #print(f'Resample condition: {1./self.sum_squarednormalized_weights:.2f} < {n:.2f} -----> {1./self.sum_squarednormalized_weights < n}')
        # if (1./self.sum_squarednormalized_weights < n):
        if True:
            # print("---------- ReSampling!!!!!- -----------")
            indices = []
            C = [0.] +[sum(self.norm_weights[:i+1]) for i in range(self.n_part)]
            u0, j = random.random(), 0

            for u in [(u0+i)/self.n_part for i in range(self.n_part)]:
                while u > C[j]:
                    j+=1

                indices.append(j-1)

            newParticles = []

            for i,v in enumerate(indices):
                newParticles.append(particula(self.particulas[v].x, self.particulas[v].y, self.particulas[v].ori, self.particulas[v].weight))
                newParticles[i].weight = 1
          
            self.particulas = newParticles

    def weights_calculation(self, LINEsens, DISTsens):
        left1,left2,left3,center,right3,right2,right1 = LINEsens
        centerDIST, leftDIST, rightDIST, backDIST = DISTsens

       
        if LINEsens == None : return
        if DISTsens == None : return
        

        
        for i,particula in enumerate(self.particulas):   
            sensorDIST_center_index =( int(self.map_scale_factor*self.mapmax_x+self.map_scale_factor*particula.sensorDIST_center_posx), int(self.map_scale_factor*self.mapmax_y+self.map_scale_factor*particula.sensorDIST_center_posy) )
            sensorDIST_left_index = ( int(self.map_scale_factor*self.mapmax_x+self.map_scale_factor*particula.sensorDIST_left_posx), int(self.map_scale_factor*self.mapmax_y+self.map_scale_factor*particula.sensorDIST_left_posy) )
            sensorDIST_right_index = ( int(self.map_scale_factor*self.mapmax_x+self.map_scale_factor*particula.sensorDIST_right_posx), int(self.map_scale_factor*self.mapmax_y+self.map_scale_factor*particula.sensorDIST_right_posy) )
            sensorDIST_back_index = ( int(self.map_scale_factor*self.mapmax_x+self.map_scale_factor*particula.sensorDIST_back_posx), int(self.map_scale_factor*self.mapmax_y+self.map_scale_factor*particula.sensorDIST_back_posy) )

            # Distance map, each cell (index value) represents 0.1 radius, so (dist value * 10)
            particle_centerDIST = self.distance_map_full[sensorDIST_center_index[1],sensorDIST_center_index[0]]
            particle_leftDIST = self.distance_map_full[sensorDIST_left_index[1],sensorDIST_left_index[0]]
            particle_rightDIST = self.distance_map_full[sensorDIST_right_index[1],sensorDIST_right_index[0]]
            particle_backDIST = self.distance_map_full[sensorDIST_back_index[1],sensorDIST_back_index[0]]

            centerDIFF = (particle_centerDIST/self.map_scale_factor - centerDIST)**2
            leftDIFF = (particle_leftDIST/self.map_scale_factor - leftDIST)**2
            rightDIFF = (particle_rightDIST/self.map_scale_factor - rightDIST)**2
            backDIFF = (particle_backDIST/self.map_scale_factor - backDIST)**2
            
            
            # particula.weight +=  exp(-centerDIFF/sigma)
            particula.weight +=  ( exp(-centerDIFF) + exp(-leftDIFF) + exp(-rightDIFF) + exp(-backDIFF))

            # minDIFF = min(centerDIFF,leftDIFF,rightDIFF,backDIFF)
            # particula.weight += exp(-minDIFF)

            # print(particula.weight)
                

    # Normalize the weights      
    def weights_normalization(self):
        sum_weights = 0
        sum_squarednormalized_weights = 0
        
        # Sum of all weights
        for i,v in enumerate(self.particulas):
            sum_weights += v.weight

        self.sum_weights = sum_weights

        # Normalize all weights
        for i,v in enumerate(self.particulas): 
            normalized_weight = v.weight/self.sum_weights
            self.norm_weights[i] = normalized_weight

            sum_squarednormalized_weights += self.norm_weights[i]**2      # sum(norm_weight[i]^2)

        # Store the sum of all squared normalized weights
        self.sum_squarednormalized_weights = sum_squarednormalized_weights

    # Use the viewer functions of viewercv to show particles in the cv window
    def showParticles(self,real_posx,real_posy,ori, diameter):
        self.biewer.clearImg()
        self.biewer.drawMap(self.map)
        self.biewer.updateParticles(self.particulas)
        self.biewer.drawParticles()
        self.biewer.drawReal(real_posx,real_posy,ori, diameter)
        self.biewer.showImg()