#import particles
import viewercv
import processmap
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
        
        # Sensores de distnacia
        self.sensorDIST_apparture = 30

        self.sensorDIST_center_ori = 0
        self.sensorDIST_center_posx = self.x + 0.5*cos(radians(ori+self.sensorDIST_center_ori))
        self.sensorDIST_center_posy = self.y - 0.5*sin(radians(ori+self.sensorDIST_center_ori))
        self.sensorDIST_center_endpointleft_x = None
        self.sensorDIST_center_endpointleft_y = None
        self.sensorDIST_center_endpointcenter_x = None
        self.sensorDIST_center_endpointcenter_y = None
        self.sensorDIST_center_endpointright_x = None
        self.sensorDIST_center_endpointright_y = None

        self.sensorDIST_left_ori = 60
        self.sensorDIST_left_posx = self.x + 0.5*cos(radians(ori+self.sensorDIST_left_ori))
        self.sensorDIST_left_posy = self.y - 0.5*sin(radians(ori+self.sensorDIST_left_ori))
        self.sensorDIST_left_endpointleft_x = None
        self.sensorDIST_left_endpointleft_y = None
        self.sensorDIST_left_endpointcenter_x = None
        self.sensorDIST_left_endpointcenter_y = None
        self.sensorDIST_left_endpointright_x = None
        self.sensorDIST_left_endpointright_y = None

        self.sensorDIST_right_ori = -60
        self.sensorDIST_right_posx = self.x + 0.5*cos(radians(ori+self.sensorDIST_right_ori))
        self.sensorDIST_right_posy = self.y - 0.5*sin(radians(ori+self.sensorDIST_right_ori))
        self.sensorDIST_right_endpointleft_x = None
        self.sensorDIST_right_endpointleft_y = None
        self.sensorDIST_right_endpointcenter_x = None
        self.sensorDIST_right_endpointcenter_y = None
        self.sensorDIST_right_endpointright_x = None
        self.sensorDIST_right_endpointright_y = None

        self.sensorDIST_back_ori = 180
        self.sensorDIST_back_posx = self.x + 0.5*cos(radians(ori+self.sensorDIST_back_ori))
        self.sensorDIST_back_posy = self.y - 0.5*sin(radians(ori+self.sensorDIST_back_ori))
        self.sensorDIST_back_endpointleft_x = None
        self.sensorDIST_back_endpointleft_y = None
        self.sensorDIST_back_endpointcenter_x = None
        self.sensorDIST_back_endpointcenter_y = None
        self.sensorDIST_back_endpointright_x = None
        self.sensorDIST_back_endpointright_y = None
        
        # Sensores de linha
        self.sensor_center_posx = self.x + 0.438*cos(radians(ori))
        self.sensor_center_posy = self.y - 0.438*sin(radians(ori))

        # self.sensor_L1_posx = self.sensor_center_posx + 3*0.08*cos(radians(ori+90))
        # self.sensor_L1_posy = self.sensor_center_posy + 3*0.08*sin(radians(ori-90))

        # self.sensor_L2_posx = self.sensor_center_posx + 2*0.08*cos(radians(ori+90))
        # self.sensor_L2_posy = self.sensor_center_posy + 2*0.08*sin(radians(ori-90))

        # self.sensor_L3_posx = self.sensor_center_posx + 1*0.08*cos(radians(ori+90))
        # self.sensor_L3_posy = self.sensor_center_posy + 1*0.08*sin(radians(ori-90))

        # self.sensor_R1_posx = self.sensor_center_posx + 3*0.08*cos(radians(ori-90))
        # self.sensor_R1_posy = self.sensor_center_posy + 3*0.08*sin(radians(ori+90))

        # self.sensor_R2_posx = self.sensor_center_posx + 2*0.08*cos(radians(ori-90))
        # self.sensor_R2_posy = self.sensor_center_posy + 2*0.08*sin(radians(ori+90))

        # self.sensor_R3_posx = self.sensor_center_posx + 1*0.08*cos(radians(ori-90))
        # self.sensor_R3_posy = self.sensor_center_posy + 1*0.08*sin(radians(ori+90))

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

        mapa = processmap.map()
        self.areas = mapa.getAreas()
        self.map_scale_factor = mapa.getScale()
        self.map = mapa.getMap()
        self.map_cropped = mapa.getMapCropped()
        self.distance_map_full = mapa.getDistanceMap()
        self.distance_map_cropped = mapa.getDistanceMapCropped()

        for i in range (self.n_part):
            # Orientacao random
            self.particulas.append( particula( random.random() * (mapmax_x), random.random() * (mapmax_y), random.random()*360, 1))

            # Orientacao 0
            # self.particulas.append( particula( random.random() * (mapmax_x), random.random() * (mapmax_y), 0, 1))


            #self.particulas.append((random.random() * ((mapmax_x-1)+0.5), random.random() * (mapmax_y-1)+0.5, random.random()*360 - 180, 1))
            #self.particulas.append((5, 8, 0))
            
            #self.weights.append(1)
            self.norm_weights.append(1/self.n_part)
            #self.ori.append(self.particulas[i][-1])
            #self.ori.append(0)

        self.visualizer = viewercv.ViewParticles(self.mapmax_x, self.mapmax_y, self.particulas, self.areas)

    def odometry_move_particles(self, motors, motors_noise):       
        self.motors = motors
        noite_multiplier = 5
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
            particula.sensorDIST_center_posy = particula.y + 0.5*sin(radians(ori))

            particula.sensorDIST_left_posx = particula.x + 0.5*cos(radians(ori-60))
            particula.sensorDIST_left_posy = particula.y + 0.5*sin(radians(ori-60))

            particula.sensorDIST_right_posx = particula.x + 0.5*cos(radians(ori+60))
            particula.sensorDIST_right_posy = particula.y + 0.5*sin(radians(ori+60))

            particula.sensorDIST_back_posx = particula.x + 0.5*cos(radians(ori+180))
            particula.sensorDIST_back_posy = particula.y + 0.5*sin(radians(ori+180))

            # particula.sensor_L1_posx = particula.sensor_center_posx + 3*0.08*cos(radians(ori+90))
            # particula.sensor_L1_posy = particula.sensor_center_posy + 3*0.08*sin(radians(ori-90))

            # particula.sensor_L2_posx = particula.sensor_center_posx + 2*0.08*cos(radians(ori+90))
            # particula.sensor_L2_posy = particula.sensor_center_posy + 2*0.08*sin(radians(ori-90))

            # particula.sensor_L3_posx = particula.sensor_center_posx + 1*0.08*cos(radians(ori+90))
            # particula.sensor_L3_posy = particula.sensor_center_posy + 1*0.08*sin(radians(ori-90))

            # particula.sensor_R1_posx = particula.sensor_center_posx + 3*0.08*cos(radians(ori-90))
            # particula.sensor_R1_posy = particula.sensor_center_posy + 3*0.08*sin(radians(ori+90))

            # particula.sensor_R2_posx = particula.sensor_center_posx + 2*0.08*cos(radians(ori-90))
            # particula.sensor_R2_posy = particula.sensor_center_posy + 2*0.08*sin(radians(ori+90))

            # particula.sensor_R3_posx = particula.sensor_center_posx + 1*0.08*cos(radians(ori-90))
            # particula.sensor_R3_posy = particula.sensor_center_posy + 1*0.08*sin(radians(ori+90))
            

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
    
    # Para o segundo metodo de calculo dos pesos
    def calculateDistanceEndpoints(self,centerDIST, leftDIST, rightDIST, backDIST):
        for i,particula in enumerate(self.particulas):  
            particula.sensorDIST_center_endpointleft_x = particula.sensorDIST_center_posx + centerDIST*cos(radians(particula.ori + particula.sensorDIST_center_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_center_endpointleft_y = particula.sensorDIST_center_posy - centerDIST*sin(radians(particula.ori + particula.sensorDIST_center_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_center_endpointcenter_x = particula.sensorDIST_center_posx + centerDIST*cos(radians(particula.ori + particula.sensorDIST_center_ori))
            particula.sensorDIST_center_endpointcenter_y = particula.sensorDIST_center_posy - centerDIST*sin(radians(particula.ori + particula.sensorDIST_center_ori))
            particula.sensorDIST_center_endpointright_x = particula.sensorDIST_center_posx + centerDIST*cos(radians(particula.ori + particula.sensorDIST_center_ori - particula.sensorDIST_apparture))
            particula.sensorDIST_center_endpointright_y = particula.sensorDIST_center_posy - centerDIST*sin(radians(particula.ori + particula.sensorDIST_center_ori - particula.sensorDIST_apparture))

            particula.sensorDIST_left_endpointleft_x = particula.sensorDIST_left_posx + leftDIST*cos(radians(particula.ori + particula.sensorDIST_left_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_left_endpointleft_y = particula.sensorDIST_left_posy - leftDIST*sin(radians(particula.ori + particula.sensorDIST_left_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_left_endpointcenter_x = particula.sensorDIST_left_posx + leftDIST*cos(radians(particula.ori + particula.sensorDIST_left_ori))
            particula.sensorDIST_left_endpointcenter_y = particula.sensorDIST_left_posy - leftDIST*sin(radians(particula.ori + particula.sensorDIST_left_ori))
            particula.sensorDIST_left_endpointright_x = particula.sensorDIST_left_posx + leftDIST*cos(radians(particula.ori + particula.sensorDIST_left_ori - particula.sensorDIST_apparture))
            particula.sensorDIST_left_endpointright_y = particula.sensorDIST_left_posy - leftDIST*sin(radians(particula.ori + particula.sensorDIST_left_ori - particula.sensorDIST_apparture))

            particula.sensorDIST_right_endpointleft_x = particula.sensorDIST_right_posx + rightDIST*cos(radians(particula.ori + particula.sensorDIST_right_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_right_endpointleft_y = particula.sensorDIST_right_posy - rightDIST*sin(radians(particula.ori + particula.sensorDIST_right_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_right_endpointcenter_x = particula.sensorDIST_right_posx + rightDIST*cos(radians(particula.ori + particula.sensorDIST_right_ori))
            particula.sensorDIST_right_endpointcenter_y = particula.sensorDIST_right_posy - rightDIST*sin(radians(particula.ori + particula.sensorDIST_right_ori))
            particula.sensorDIST_right_endpointright_x = particula.sensorDIST_right_posx + rightDIST*cos(radians(particula.ori + particula.sensorDIST_right_ori - particula.sensorDIST_apparture))
            particula.sensorDIST_right_endpointright_y = particula.sensorDIST_right_posy - rightDIST*sin(radians(particula.ori + particula.sensorDIST_right_ori - particula.sensorDIST_apparture))

            particula.sensorDIST_back_endpointleft_x = particula.sensorDIST_back_posx + backDIST*cos(radians(particula.ori + particula.sensorDIST_back_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_back_endpointleft_y = particula.sensorDIST_back_posy - backDIST*sin(radians(particula.ori + particula.sensorDIST_back_ori + particula.sensorDIST_apparture))
            particula.sensorDIST_back_endpointcenter_x = particula.sensorDIST_back_posx + backDIST*cos(radians(particula.ori + particula.sensorDIST_back_ori))
            particula.sensorDIST_back_endpointcenter_y = particula.sensorDIST_back_posy - backDIST*sin(radians(particula.ori + particula.sensorDIST_back_ori))
            particula.sensorDIST_back_endpointright_x = particula.sensorDIST_back_posx + backDIST*cos(radians(particula.ori + particula.sensorDIST_back_ori - particula.sensorDIST_apparture))
            particula.sensorDIST_back_endpointright_y = particula.sensorDIST_back_posy - backDIST*sin(radians(particula.ori + particula.sensorDIST_back_ori - particula.sensorDIST_apparture))



    def weights_calculation(self, LINEsens, DISTsens):
        left1,left2,left3,center,right3,right2,right1 = LINEsens
        centerDIST, leftDIST, rightDIST, backDIST = DISTsens

       
        if LINEsens == None : return
        if DISTsens == None : return
        self.calculateDistanceEndpoints(centerDIST, leftDIST, rightDIST, backDIST)
        # Ainda nao fiz nada aqui para o segundo metodo do calculo dos pesos    
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


            centerDIFF = (particle_centerDIST - centerDIST)**2
            leftDIFF = (particle_leftDIST - leftDIST)**2
            rightDIFF = (particle_rightDIST - rightDIST)**2
            backDIFF = (particle_backDIST - backDIST)**2
            
            
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
    def showParticles(self,real_posx,real_posy,ori, diameter, DISTsens):
        self.visualizer.clearImg()
        self.visualizer.drawMap(self.map)
        self.visualizer.updateParticles(self.particulas)
        self.visualizer.drawParticles()
        self.visualizer.drawReal(real_posx,real_posy,ori, diameter, DISTsens)
        self.visualizer.showImg()