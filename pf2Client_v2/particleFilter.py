#import particles
import numpy as np
from matplotlib import pyplot as plt
from math import *
from lxml import etree
import random
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift

class particula():
    def __init__(self, x, y, ori, w, IRangles, num_endpoints):
        self.x = x
        self.y = y
        self.ori = ori
        self.weight = w
        self.num_endpoints = num_endpoints
        
        # Sensores de distnacia

        self.sensorDIST_apparture = radians(30)
        self.endpoints_angle = (2*self.sensorDIST_apparture)/(self.num_endpoints-1)
        self.endpoints = np.empty([len(IRangles),self.num_endpoints,2])
        self.sensorDIST = np.array([[angle, self.x + 0.5*cos(ori+angle), self.y - 0.5*sin(ori+angle)] for i, angle in enumerate(IRangles)])
        # self.sensorDIST = [[angle, (self.x + 0.5*cos(ori+angle), self.y - 0.5*sin(ori+angle)), []] for i, angle in enumerate(IRangles)]
        # self.sensorDIST = []
        # for i in range(len(IRangles)):
        #     info = []
        #     info.append(IRangles[i])
        #     info.append((self.x + 0.5*cos(ori+IRangles[i]), self.y - 0.5*sin(ori+IRangles[i])))
        #     info.append([])
        #     self.sensorDIST.append(info)
        # print(self.sensorDIST[0][1][0])
        self.sensorDIST_center_ori = IRangles[0]
        self.sensorDIST_center_posx = self.x + 0.5*cos(ori+self.sensorDIST_center_ori)
        # print(self.sensorDIST_center_posx)
        self.sensorDIST_center_posy = self.y - 0.5*sin(ori+self.sensorDIST_center_ori)
        self.sensorDIST_center_endpointleft_x = None
        self.sensorDIST_center_endpointleft_y = None
        self.sensorDIST_center_endpointcenter_x = None
        self.sensorDIST_center_endpointcenter_y = None
        self.sensorDIST_center_endpointright_x = None
        self.sensorDIST_center_endpointright_y = None

        self.sensorDIST_left_ori = IRangles[1]
        self.sensorDIST_left_posx = self.x + 0.5*cos(ori+self.sensorDIST_left_ori)
        self.sensorDIST_left_posy = self.y - 0.5*sin(ori+self.sensorDIST_left_ori)
        self.sensorDIST_left_endpointleft_x = None
        self.sensorDIST_left_endpointleft_y = None
        self.sensorDIST_left_endpointcenter_x = None
        self.sensorDIST_left_endpointcenter_y = None
        self.sensorDIST_left_endpointright_x = None
        self.sensorDIST_left_endpointright_y = None

        self.sensorDIST_right_ori = IRangles[2]
        self.sensorDIST_right_posx = self.x + 0.5*cos(ori+self.sensorDIST_right_ori)
        self.sensorDIST_right_posy = self.y - 0.5*sin(ori+self.sensorDIST_right_ori)
        self.sensorDIST_right_endpointleft_x = None
        self.sensorDIST_right_endpointleft_y = None
        self.sensorDIST_right_endpointcenter_x = None
        self.sensorDIST_right_endpointcenter_y = None
        self.sensorDIST_right_endpointright_x = None
        self.sensorDIST_right_endpointright_y = None

        self.sensorDIST_back_ori = IRangles[3]
        self.sensorDIST_back_posx = self.x + 0.5*cos(ori+self.sensorDIST_back_ori)
        self.sensorDIST_back_posy = self.y - 0.5*sin(ori+self.sensorDIST_back_ori)
        self.sensorDIST_back_endpointleft_x = None
        self.sensorDIST_back_endpointleft_y = None
        self.sensorDIST_back_endpointcenter_x = None
        self.sensorDIST_back_endpointcenter_y = None
        self.sensorDIST_back_endpointright_x = None
        self.sensorDIST_back_endpointright_y = None
        
        # Sensores de linha
        self.sensor_center_posx = self.x + 0.438*cos(ori)
        self.sensor_center_posy = self.y - 0.438*sin(ori)

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
    def __init__(self, map, IRangles, num_endpoints, n_part=4000):
        self.bandwidth = None
        self.centroides = None
        self.IRangles = IRangles
        self.num_endpoints = num_endpoints
        self.n_part = n_part
        self.sum_weights = n_part
        self.max_w = 1

        self.last_motors = (0,0)
        self.sum_squarednormalized_weights = 0
        # self.particulas = []
        self.particulas = np.empty(self.n_part, dtype=particula)
        self.norm_weights = []

        self.map_scale_factor = map.scale
        self.distance_map_full = map.getDistanceMap()
        self.mapmax_x = map.mapmax_x
        self.mapmax_y = map.mapmax_y
        
        self.x_offset = self.map_scale_factor*self.mapmax_x
        self.y_offset = self.map_scale_factor*self.mapmax_y


        for i in range (self.n_part):
            # Orientacao random
            # self.particulas[i] = particula(np.random.random() * self.mapmax_x, np.random.random() * self.mapmax_y, random.random()*360, 1, self.IRangles,self.num_endpoints)
            

            # Orientacao 0
            self.particulas[i] = particula(np.random.random() * self.mapmax_x, np.random.random() * self.mapmax_y, 0, 1, self.IRangles, self.num_endpoints)


            #self.particulas.append((random.random() * ((mapmax_x-1)+0.5), random.random() * (mapmax_y-1)+0.5, random.random()*360 - 180, 1))
            #self.particulas.append((5, 8, 0))
            
            #self.weights.append(1)
            self.norm_weights.append(1/self.n_part)
            #self.ori.append(self.particulas[i][-1])
            #self.ori.append(0)

    def odometry_move_particles(self, motors, motors_noise, collision):
        if collision:
            self.motors = (-self.last_motors[0],-self.last_motors[1])
        else:
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
            x = particula.x + (lin * cos(particula.ori))
            y = particula.y - (lin * sin(particula.ori))

            rot = out_r - out_l # / self.robot_diameter ( = 1 )
            ori = (particula.ori + rot) % 6.28318530


            particula.x = x
            particula.y = y
            particula.ori = ori

            particula.sensor_center_posx = x + 0.438*cos(ori)
            particula.sensor_center_posy = y - 0.438*sin(ori)

            particula.sensorDIST_center_posx = particula.x + 0.5*cos(ori+self.IRangles[0])
            particula.sensorDIST_center_posy = particula.y - 0.5*sin(ori+self.IRangles[0])

            particula.sensorDIST_left_posx = particula.x + 0.5*cos(ori+self.IRangles[1])
            particula.sensorDIST_left_posy = particula.y - 0.5*sin(ori+self.IRangles[1])

            particula.sensorDIST_right_posx = particula.x + 0.5*cos(ori+self.IRangles[2])
            particula.sensorDIST_right_posy = particula.y - 0.5*sin(ori+self.IRangles[2])

            particula.sensorDIST_back_posx = particula.x + 0.5*cos(ori+self.IRangles[3])
            particula.sensorDIST_back_posy = particula.y - 0.5*sin(ori+self.IRangles[3])

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


    # def resample(self):
    #     n = 0.999*self.n_part
    #     #print(f'Resample condition: {1./self.sum_squarednormalized_weights:.2f} < {n:.2f} -----> {1./self.sum_squarednormalized_weights < n}')
    #     # if (1./self.sum_squarednormalized_weights < n):
    #     if True:
    #         # print("---------- ReSampling!!!!!- -----------")
    #         indices = []
    #         C = [0.] +[sum(self.norm_weights[:i+1]) for i in range(self.n_part)]
    #         u0, j = random.random(), 0

    #         for u in [(u0+i)/self.n_part for i in range(self.n_part)]:
    #             while u > C[j]:
    #                 j+=1

    #             indices.append(j-1)

    #         newParticles = []

    #         for i,v in enumerate(indices):
    #             newParticles.append(particula(self.particulas[v].x, self.particulas[v].y, self.particulas[v].ori, self.particulas[v].weight, self.IRangles))
    #             newParticles[i].weight = 1
          
    #         self.particulas = newParticles

    # Here, np.cumsum() is used to calculate the cumulative sum of the normalized weights.
    # np.insert() is used to insert a zero at the beginning of the array to make sure the first particle is selected with non-zero probability.
    # np.random.rand() is used to generate the uniform random numbers.
    # Finally, np.zeros() is used to initialize the indices array with zeros, and dtype=np.int64 is specified to make sure it has the same data type as the original indices list.

    def resample(self):
        n = 0.999 * self.n_part
        # if (1./self.sum_squarednormalized_weights < n):
        if True:
            # Calculate cumulative sum of normalized weights
            cum_weights = np.cumsum(self.norm_weights)
            cum_weights = np.insert(cum_weights, 0, 0)  # Insert 0 at the beginning

            # Generate uniform random numbers and find corresponding indices
            u0, j = np.random.rand(), 0
            u_values = [(u0 + i) / self.n_part for i in range(self.n_part)]
            indices = np.zeros(self.n_part, dtype=np.int64)
            for i in range(self.n_part):
                while u_values[i] > cum_weights[j]:
                    j += 1
                indices[i] = j - 1

            # Create new particles with equal weight
            # newParticles = []
            newParticles = np.empty(self.n_part, dtype=particula)
            for i, v in enumerate(indices):
                # newParticles.append(particula(self.particulas[v].x, self.particulas[v].y, self.particulas[v].ori, self.particulas[v].weight, self.IRangles))
                newParticles[i] = particula(self.particulas[v].x, self.particulas[v].y, self.particulas[v].ori, self.particulas[v].weight, self.IRangles, self.num_endpoints)
                newParticles[i].weight = 1

            self.particulas = newParticles
    
    # Para o segundo metodo de calculo dos pesos
    def calculateDistanceEndpoints(self, DISTsens):
        centerDIST, leftDIST, rightDIST, backDIST = DISTsens
        acceptable_limit = 3            #IMPORTANTE
        for i,particula in enumerate(self.particulas): 
            if centerDIST <= acceptable_limit:
                particula.sensorDIST_center_endpointleft_x = particula.sensorDIST_center_posx + centerDIST*cos(particula.ori + particula.sensorDIST_center_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_center_endpointleft_y = particula.sensorDIST_center_posy - centerDIST*sin(particula.ori + particula.sensorDIST_center_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_center_endpointcenter_x = particula.sensorDIST_center_posx + centerDIST*cos(particula.ori + particula.sensorDIST_center_ori)
                particula.sensorDIST_center_endpointcenter_y = particula.sensorDIST_center_posy - centerDIST*sin(particula.ori + particula.sensorDIST_center_ori)
                particula.sensorDIST_center_endpointright_x = particula.sensorDIST_center_posx + centerDIST*cos(particula.ori + particula.sensorDIST_center_ori - particula.sensorDIST_apparture)
                particula.sensorDIST_center_endpointright_y = particula.sensorDIST_center_posy - centerDIST*sin(particula.ori + particula.sensorDIST_center_ori - particula.sensorDIST_apparture)
            else:
                particula.sensorDIST_center_endpointleft_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_center_endpointleft_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_center_endpointcenter_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_center_endpointcenter_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_center_endpointright_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_center_endpointright_y = 2*self.y_offset/self.map_scale_factor - 1

            if leftDIST <= acceptable_limit:
                particula.sensorDIST_left_endpointleft_x = particula.sensorDIST_left_posx + leftDIST*cos(particula.ori + particula.sensorDIST_left_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_left_endpointleft_y = particula.sensorDIST_left_posy - leftDIST*sin(particula.ori + particula.sensorDIST_left_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_left_endpointcenter_x = particula.sensorDIST_left_posx + leftDIST*cos(particula.ori + particula.sensorDIST_left_ori)
                particula.sensorDIST_left_endpointcenter_y = particula.sensorDIST_left_posy - leftDIST*sin(particula.ori + particula.sensorDIST_left_ori)
                particula.sensorDIST_left_endpointright_x = particula.sensorDIST_left_posx + leftDIST*cos(particula.ori + particula.sensorDIST_left_ori - particula.sensorDIST_apparture)
                particula.sensorDIST_left_endpointright_y = particula.sensorDIST_left_posy - leftDIST*sin(particula.ori + particula.sensorDIST_left_ori - particula.sensorDIST_apparture)
            else:
                particula.sensorDIST_left_endpointleft_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_left_endpointleft_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_left_endpointcenter_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_left_endpointcenter_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_left_endpointright_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_left_endpointright_y = 2*self.y_offset/self.map_scale_factor - 1

            if rightDIST <= acceptable_limit:
                particula.sensorDIST_right_endpointleft_x = particula.sensorDIST_right_posx + rightDIST*cos(particula.ori + particula.sensorDIST_right_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_right_endpointleft_y = particula.sensorDIST_right_posy - rightDIST*sin(particula.ori + particula.sensorDIST_right_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_right_endpointcenter_x = particula.sensorDIST_right_posx + rightDIST*cos(particula.ori + particula.sensorDIST_right_ori)
                particula.sensorDIST_right_endpointcenter_y = particula.sensorDIST_right_posy - rightDIST*sin(particula.ori + particula.sensorDIST_right_ori)
                particula.sensorDIST_right_endpointright_x = particula.sensorDIST_right_posx + rightDIST*cos(particula.ori + particula.sensorDIST_right_ori - particula.sensorDIST_apparture)
                particula.sensorDIST_right_endpointright_y = particula.sensorDIST_right_posy - rightDIST*sin(particula.ori + particula.sensorDIST_right_ori - particula.sensorDIST_apparture)
            else:
                particula.sensorDIST_right_endpointleft_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_right_endpointleft_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_right_endpointcenter_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_right_endpointcenter_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_right_endpointright_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_right_endpointright_y = 2*self.y_offset/self.map_scale_factor - 1
            
            if backDIST <= acceptable_limit:
                particula.sensorDIST_back_endpointleft_x = particula.sensorDIST_back_posx + backDIST*cos(particula.ori + particula.sensorDIST_back_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_back_endpointleft_y = particula.sensorDIST_back_posy - backDIST*sin(particula.ori + particula.sensorDIST_back_ori + particula.sensorDIST_apparture)
                particula.sensorDIST_back_endpointcenter_x = particula.sensorDIST_back_posx + backDIST*cos(particula.ori + particula.sensorDIST_back_ori)
                particula.sensorDIST_back_endpointcenter_y = particula.sensorDIST_back_posy - backDIST*sin(particula.ori + particula.sensorDIST_back_ori)
                particula.sensorDIST_back_endpointright_x = particula.sensorDIST_back_posx + backDIST*cos(particula.ori + particula.sensorDIST_back_ori - particula.sensorDIST_apparture)
                particula.sensorDIST_back_endpointright_y = particula.sensorDIST_back_posy - backDIST*sin(particula.ori + particula.sensorDIST_back_ori - particula.sensorDIST_apparture)
            else:
                particula.sensorDIST_back_endpointleft_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_back_endpointleft_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_back_endpointcenter_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_back_endpointcenter_y = 2*self.y_offset/self.map_scale_factor - 1
                particula.sensorDIST_back_endpointright_x = 2*self.x_offset/self.map_scale_factor - 1
                particula.sensorDIST_back_endpointright_y = 2*self.y_offset/self.map_scale_factor - 1
        

    def weights_calculation(self, LINEsens, DISTsens, metodo):
        # left1,left2,left3,center,right3,right2,right1 = LINEsens
        centerDIST, leftDIST, rightDIST, backDIST = DISTsens
       
        if LINEsens == None : return
        if DISTsens == None : return
        self.max_w = 1
        # Metodo 1 
        if metodo == 1:
            for i,particula in enumerate(self.particulas):  
                sensorDIST_center_index =( 
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_center_posx), 
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_center_posy) 
                )
                sensorDIST_left_index = ( 
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_left_posx), 
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_left_posy) 
                )
                sensorDIST_right_index = ( 
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_right_posx), 
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_right_posy) 
                )
                sensorDIST_back_index = ( 
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_back_posx), 
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_back_posy) 
                )

                particle_centerDIST =  self.distance_map_full[sensorDIST_center_index[1],sensorDIST_center_index[0]]
                particle_leftDIST = self.distance_map_full[sensorDIST_left_index[1],sensorDIST_left_index[0]]
                particle_rightDIST = self.distance_map_full[sensorDIST_right_index[1],sensorDIST_right_index[0]]
                particle_backDIST = self.distance_map_full[sensorDIST_back_index[1],sensorDIST_back_index[0]]

                centerDIFF = (particle_centerDIST - centerDIST)**2
                leftDIFF = (particle_leftDIST - leftDIST)**2
                rightDIFF = (particle_rightDIST - rightDIST)**2
                backDIFF = (particle_backDIST - backDIST)**2
                
                dummy = min(centerDIFF, leftDIFF, rightDIFF, backDIFF)
                particula.weight +=  ( exp(-dummy))
                
                # particula.weight +=  exp(-centerDIFF/sigma)
                # particula.weight +=  ( exp(-centerDIFF) + exp(-leftDIFF) + exp(-rightDIFF) + exp(-backDIFF))
                if particula.weight > self.max_w: self.max_w = particula.weight

        # Metodo 2v2
        elif metodo == 4:
            acceptable_limit = 3            #IMPORTANTE
            for i,particula in enumerate(self.particulas): 
                pesosSENS = []
                for j,v in enumerate(particula.sensorDIST):
                    leitura = DISTsens[j]
                    value = 1000
                    pesosDIST = []
                    if leitura <= acceptable_limit:
                        for k in range(particula.num_endpoints):
                            angle = particula.endpoints_angle*k
                            posx = particula.sensorDIST[j][1] + leitura*cos(particula.ori + particula.sensorDIST[j][0] - particula.sensorDIST_apparture + angle)
                            posy = particula.sensorDIST[j][2] - leitura*sin(particula.ori +  particula.sensorDIST[j][0] - particula.sensorDIST_apparture + angle)

                            idx = int(self.x_offset+self.map_scale_factor*posx)
                            idy = int(self.y_offset+self.map_scale_factor*posy)
                            distancemap_value = self.distance_map_full[idy,idx]
                            pesosDIST.append(distancemap_value)
                        # print(pesosDIST)
                        value = min(pesosDIST)**2

                    pesosSENS.append(value)
                # if i == 3: 
                #     print(pesosSENS)
                particula.weight +=  exp((-pesosSENS[0]) + exp(-pesosSENS[1]) + exp(-pesosSENS[2]) + exp(-pesosSENS[3]))
        
        # Metodo 2
        elif metodo == 2:
            self.calculateDistanceEndpoints(DISTsens) # Ativar para metodo 2 (Apenas esta função gasta +20 ms)

            for i,particula in enumerate(self.particulas): 
                sensorDIST_center_endpointleft_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_center_endpointleft_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_center_endpointleft_y)
                )
                sensorDIST_center_endpointcenter_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_center_endpointcenter_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_center_endpointcenter_y)
                )
                sensorDIST_center_endpointright_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_center_endpointright_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_center_endpointright_y)
                )
                
                sensorDIST_left_endpointleft_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_left_endpointleft_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_left_endpointleft_y)
                )
                sensorDIST_left_endpointcenter_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_left_endpointcenter_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_left_endpointcenter_y)
                )
                sensorDIST_left_endpointright_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_left_endpointright_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_left_endpointright_y)
                )

                sensorDIST_right_endpointleft_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_right_endpointleft_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_right_endpointleft_y)
                )
                sensorDIST_right_endpointcenter_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_right_endpointcenter_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_right_endpointcenter_y)
                )
                sensorDIST_right_endpointright_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_right_endpointright_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_right_endpointright_y)
                )

                sensorDIST_back_endpointleft_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_back_endpointleft_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_back_endpointleft_y)
                )
                sensorDIST_back_endpointcenter_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_back_endpointcenter_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_back_endpointcenter_y)
                )
                sensorDIST_back_endpointright_index = (
                    int(self.x_offset+self.map_scale_factor*particula.sensorDIST_back_endpointright_x),
                    int(self.y_offset+self.map_scale_factor*particula.sensorDIST_back_endpointright_y)
                )

                particle_sensorDIST_center_endpointleft_DIST = self.distance_map_full[sensorDIST_center_endpointleft_index[1],sensorDIST_center_endpointleft_index[0]]
                particle_sensorDIST_center_endpointcenter_DIST = self.distance_map_full[sensorDIST_center_endpointcenter_index[1],sensorDIST_center_endpointcenter_index[0]]
                particle_sensorDIST_center_endpointright_DIST = self.distance_map_full[sensorDIST_center_endpointright_index[1],sensorDIST_center_endpointright_index[0]]
                centerDIFF_2 = min(particle_sensorDIST_center_endpointleft_DIST, particle_sensorDIST_center_endpointcenter_DIST, particle_sensorDIST_center_endpointright_DIST)**2

                particle_sensorDIST_left_endpointleft_DIST = self.distance_map_full[sensorDIST_left_endpointleft_index[1],sensorDIST_left_endpointleft_index[0]]
                particle_sensorDIST_left_endpointcenter_DIST = self.distance_map_full[sensorDIST_left_endpointcenter_index[1],sensorDIST_left_endpointcenter_index[0]]
                particle_sensorDIST_left_endpointright_DIST = self.distance_map_full[sensorDIST_left_endpointright_index[1],sensorDIST_left_endpointright_index[0]]
                leftDIFF_2 = min(particle_sensorDIST_left_endpointleft_DIST, particle_sensorDIST_left_endpointcenter_DIST, particle_sensorDIST_left_endpointright_DIST)**2


                particle_sensorDIST_right_endpointleft_DIST = self.distance_map_full[sensorDIST_right_endpointleft_index[1],sensorDIST_right_endpointleft_index[0]]
                particle_sensorDIST_right_endpointcenter_DIST = self.distance_map_full[sensorDIST_right_endpointcenter_index[1],sensorDIST_right_endpointcenter_index[0]]
                particle_sensorDIST_right_endpointright_DIST = self.distance_map_full[sensorDIST_right_endpointright_index[1],sensorDIST_right_endpointright_index[0]]
                rightDIFF_2 = min(particle_sensorDIST_right_endpointleft_DIST, particle_sensorDIST_right_endpointcenter_DIST, particle_sensorDIST_right_endpointright_DIST)**2
                

                particle_sensorDIST_back_endpointleft_DIST = self.distance_map_full[sensorDIST_back_endpointleft_index[1],sensorDIST_back_endpointleft_index[0]]
                particle_sensorDIST_back_endpointcenter_DIST = self.distance_map_full[sensorDIST_back_endpointcenter_index[1],sensorDIST_back_endpointcenter_index[0]]
                particle_sensorDIST_back_endpointright_DIST = self.distance_map_full[sensorDIST_back_endpointright_index[1],sensorDIST_back_endpointright_index[0]]
                backDIFF_2 = min(particle_sensorDIST_back_endpointleft_DIST, particle_sensorDIST_back_endpointcenter_DIST, particle_sensorDIST_back_endpointright_DIST)**2


                particula.weight +=  exp((-centerDIFF_2) + exp(-leftDIFF_2) + exp(-rightDIFF_2) + exp(-backDIFF_2))
                
                # minimumDIFF = min(centerDIFF_2,leftDIFF_2,rightDIFF_2,backDIFF_2)
                # particula.weight += ( exp(-minimumDIFF))

                # array = np.array(DISTsens)
                # minimum_index = array.argmin()
                # DIFFs_2 = (centerDIFF_2,leftDIFF_2,rightDIFF_2,backDIFF_2)
                # particula.weight += exp(-DIFFs_2[minimum_index])

                if particula.weight > self.max_w: self.max_w = particula.weight

        # Metodo 3
        elif metodo == 3:
            raio_robot = 0.5
            for i,particula in enumerate(self.particulas): 
                DISTsens_min = min(DISTsens) + raio_robot
                # print(DISTsens_min)
                distmappartvalue = self.distance_map_full[
                    self.y_offset + int(self.map_scale_factor*particula.y),
                    self.x_offset + int(self.map_scale_factor*particula.x)
                ]
                # print(distmappartvalue)
                dummy2 = (DISTsens_min - distmappartvalue)**2
                particula.weight +=  (exp(-dummy2))
                if particula.weight > self.max_w: self.max_w = particula.weight
        
                

    # Normalize the weights      
    def weights_normalization(self):
        sum_weights = 0
        sum_squarednormalized_weights = 0
        
        # Sum of all weights
        for i,particula in enumerate(self.particulas):
            sum_weights += particula.weight

        self.sum_weights = sum_weights

        # Normalize all weights
        for i,particula in enumerate(self.particulas): 
            normalized_weight = particula.weight/self.sum_weights
            self.norm_weights[i] = normalized_weight

            sum_squarednormalized_weights += self.norm_weights[i]**2      # sum(norm_weight[i]^2)

        # Store the sum of all squared normalized weights
        self.sum_squarednormalized_weights = sum_squarednormalized_weights

    def getFinalPose(self):
        x = 0
        y = 0
        ori = 0
        orix = 0
        oriy = 0
        peso_temp = 0
        for i,particula in enumerate(self.particulas):
            peso_temp = self.norm_weights[i]
            x += particula.x * peso_temp
            y += particula.y * peso_temp
            orix += cos(particula.ori)
            oriy += sin(particula.ori)
            ori += atan2(oriy,orix) * peso_temp
            # ori += particula.ori * peso_temp
        
        return (x,y,ori)
    
    # def cluster(self):
    #     # Armazenar as posições X e Y de todas as partículas
    #     X = np.zeros((self.n_part, 2))
    #     for i in range(self.n_part):
    #         X[i, 0] = self.particulas[i].x
    #         X[i, 1] = self.particulas[i].y

    #     # Executar o clustering por meanshift
    #     ms = MeanShift(bandwidth=self.bandwidth)
    #     ms.fit(X)
        
    #     # Obter os rótulos das clusters e seus centróides
    #     labels = ms.labels_
    #     centroids = ms.cluster_centers_
    #     self.centroides = centroids
    #     # Atualizar as posições X e Y das partículas para os centróides das clusters correspondentes
    #     # for i in range(self.n_part):
    #         # self.particulas[i].x = centroids[labels[i], 0]
    #         # self.particulas[i].y = centroids[labels[i], 1]
    
    def cluster(self):
        print(f'{self.particulas[2].endpoints[0][1]}')
        # Armazenar as posições X e Y de todas as partículas
        X = np.array([[p.x, p.y] for p in self.particulas])

        # Executar o clustering por meanshift
        # ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True, n_jobs=-1)
        ms = DBSCAN(eps=1, metric='l1', algorithm='auto', leaf_size=30)
        ms.fit(X)

        # Obter os rótulos das clusters e seus centróides
        labels = ms.labels_
        print(np.max(labels))
        centroids = None
        self.centroides = centroids
