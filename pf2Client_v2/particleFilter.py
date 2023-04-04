#import particles
import numpy as np
from matplotlib import pyplot as plt
from math import *
from lxml import etree
import random
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from scipy.spatial.distance import pdist, squareform

# -------------------------------------------- Elementos em cada particula, classe particula ------------------------------------------------------
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

# ------------------------------------------------------- Filtro de particulas ----------------------------------------------------------------
class filtroParticulas():
    def __init__(self, map, IRangles, num_endpoints, n_part=4000):
        self.bandwidth = None
        self.centroides = None
        self.centroides_oris = None
        self.centroides_weights = None
        self.centroides_cov = None

        self.IRangles = IRangles
        self.num_endpoints = num_endpoints
        self.num_particles = n_part
        self.effective_num_particles = 0
        self.sum_weights = n_part
        self.max_w = 1

        self.last_motors = (0,0)
        # self.movement_counter = 0
        # self.movement_counter_trigger = 0.25
        self.sum_squarednormalized_weights = 0
        # self.particulas = []
        self.noise_multiplier = 10
        self.particulas = np.empty(self.num_particles, dtype=particula)
        self.norm_weights = np.ones(self.num_particles)
        self.pesonorm = 1/self.num_particles

        self.map = map
        self.map_scale_factor = map.scale
        self.distance_map_full = map.getDistanceMap()
        self.mapmax_x = map.mapmax_x
        self.mapmax_y = map.mapmax_y
        
        self.x_offset = self.map_scale_factor*self.mapmax_x
        self.y_offset = self.map_scale_factor*self.mapmax_y

        self.createNewParticleSet()
    
    # ----------------------------------------- Criação de novo conjunto de particulas  
    def createNewParticleSet(self):
        for i in range (self.num_particles):
            #  Distribuicao e Orientacao aleatoria 
            self.particulas[i] = particula(np.random.random() * self.mapmax_x, np.random.random() * self.mapmax_y, random.random()*360, 1, self.IRangles,self.num_endpoints)
            

            # Distribuicao aleatoria e Orientacao 0
            # self.particulas[i] = particula(np.random.random() * self.mapmax_x, np.random.random() * self.mapmax_y, 0, 1, self.IRangles, self.num_endpoints)

    # ----------------------------------------- Movimentacao das particulas com base na odometria do robo
    def odometry_move_particles(self, motors, motors_noise, collision):
        if collision:
            self.motors = (-self.last_motors[0],-self.last_motors[1])
        else:
            self.motors = motors

        noise_multiplier = 5
        # self.movement_counter += abs(self.motors[0])+abs(self.motors[1])

        for i,particula in enumerate (self.particulas):
            # calculate estimated power apply
            out_l = (self.motors[0] + self.last_motors[0]) / 2
            out_r = (self.motors[1] + self.last_motors[1]) / 2
            
            out_l = random.gauss(out_l, self.noise_multiplier*motors_noise*out_l)   # out_l tem um erro de 1,5%
            out_r = random.gauss(out_r, self.noise_multiplier*motors_noise*out_r)    # out_r tem um erro de 1,5%

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

    # ---------------------------------------------- RESAMPLE ---------------------------------------------------------
    # def resample(self):
    #     n = 0.999*self.num_particles
    #     #print(f'Resample condition: {1./self.sum_squarednormalized_weights:.2f} < {n:.2f} -----> {1./self.sum_squarednormalized_weights < n}')
    #     # if (1./self.sum_squarednormalized_weights < n):
    #     if True:
    #         # print("---------- ReSampling!!!!!- -----------")
    #         indices = []
    #         C = [0.] +[sum(self.norm_weights[:i+1]) for i in range(self.num_particles)]
    #         u0, j = random.random(), 0

    #         for u in [(u0+i)/self.num_particles for i in range(self.num_particles)]:
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
        n = 0.999 * self.num_particles
        # if (1./self.sum_squarednormalized_weights < n):
        # if (self.movement_counter > self.movement_counter_trigger):
        #     self.movement_counter = 0
        if True:
            # Calculate cumulative sum of normalized weights
            cum_weights = np.cumsum(self.norm_weights)
            cum_weights = np.insert(cum_weights, 0, 0)  # Insert 0 at the beginning

            # Generate uniform random numbers and find corresponding indices
            u0, j = np.random.rand(), 0
            u_values = [(u0 + i) / self.num_particles for i in range(self.num_particles)]
            indices = np.zeros(self.num_particles, dtype=np.int64)
            for i in range(self.num_particles): # Mudar para nao aceitar particulas em posicao invalida (dentro de parede)
                while u_values[i] > cum_weights[j]:
                    j += 1
                indices[i] = j - 1

            # Create new particles with equal weight
            # newParticles = []
            newParticles = np.empty(self.num_particles, dtype=particula)
            for i, v in enumerate(indices):
                # newParticles.append(particula(self.particulas[v].x, self.particulas[v].y, self.particulas[v].ori, self.particulas[v].weight, self.IRangles))
                newParticles[i] = particula(self.particulas[v].x, self.particulas[v].y, self.particulas[v].ori, 1, self.IRangles, self.num_endpoints)

            self.particulas = newParticles
        
    def sis_resample(self):
        n = 0.999 * self.num_particles
        # if (1./self.sum_squarednormalized_weights < n):
        # if (self.movement_counter > self.movement_counter_trigger):
        #     self.movement_counter = 0

        if True:
            # Calculate cumulative sum of normalized weights
            cum_weights = np.cumsum(self.norm_weights)

            # Determine the spacing between particles
            spacing = 1 / self.num_particles

            # Generate a random offset in the range [0, spacing)
            u0 = np.random.rand() * spacing

            # Find corresponding indices using systematic resampling
            u_values = np.arange(u0, spacing*self.num_particles, spacing)
            j = 0
            indices = np.zeros(self.num_particles, dtype=np.int64)
            for i in range(self.num_particles):
                while u_values[i] > cum_weights[j]:
                    j += 1
                indices[i] = j - 1

            # Create new particles with equal weight
            newParticles = np.empty(self.num_particles, dtype=particula)
            for i, v in enumerate(indices):
                xpos = self.x_offset + self.map_scale_factor*self.particulas[v].x
                ypos = self.y_offset + self.map_scale_factor*self.particulas[v].y
                
                #self.norm_weights[i] = self.pesonorm
                if self.map.isValidLocation(xpos,ypos):
                    newParticles[i] = particula(self.particulas[v].x, self.particulas[v].y, self.particulas[v].ori, 1, self.IRangles, self.num_endpoints)
                else:
                    newParticles[i] = particula(np.random.random() * self.mapmax_x, np.random.random() * self.mapmax_y, random.random()*360, 1, self.IRangles,self.num_endpoints)
            self.particulas = newParticles


    # https://github.com/iris-ua/iris_lama/blob/master/src/pf_slam2d.cpp
    # def sistematic_resample(self):
    def sistematic_resample(self):
        self.num_particles
        sample_idx = 0

        interval = 1.0 / float(self.num_particles)

        target = interval * random.uniform(0, 1)
        cw = 0.0
        n = 0
        for i in range(self.num_particles):
            cw += self.norm_weights[i]

            while cw > target:
                sample_idx[n] = i
                n += 1
                target += interval

        # generate a new set of particles
        ps = 1 - self.current_particle_set_
        self.particles_[ps] = [None] * self.num_particles

        for i in range(self.num_particles):
            idx = sample_idx[i]

            self.particles_[ps][i] = self.particles_[self.current_particle_set_][idx]
            self.particles_[ps][i].weight = 0.0
            self.particles_[ps][i].weight_sum = self.particles_[self.current_particle_set_][idx].weight_sum

            self.particles_[ps][i].dm = DynamicDistanceMapPtr(DynamicDistanceMap(self.particles_[self.current_particle_set_][idx].dm))
            self.particles_[ps][i].occ = FrequencyOccupancyMapPtr(FrequencyOccupancyMap(self.particles_[self.current_particle_set_][idx].occ))

        self.particles_[self.current_particle_set_].clear()
        self.current_particle_set_ = ps

    # ---------------------------------------------- EXTRA (Para o segundo metodo de calculo dos pesos (APENAS 3 endpoints por sensor))----------
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
    # ------------------------------------------------------ Fim da funcao usada apenas no metodo 2 --------------------------------   
     
    # ----------------------------------------------------- Calculo do peso atribuido a cada particula --------------------------------
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
        
        # Metodo 4 (= Metodo 2, versão 2)
        elif metodo == 4:
            acceptable_limit = 3            #IMPORTANTE
            for i,particula in enumerate(self.particulas): 
                pesosSENS = []
                for j,v in enumerate(particula.sensorDIST):
                    leitura = DISTsens[j]
                    value = particula.weight
                    pesosDIST = []
                    if leitura <= acceptable_limit:
                        for k in range(particula.num_endpoints):
                            angle = particula.endpoints_angle*k
                            posx = particula.sensorDIST[j][1] + leitura*cos(particula.ori + particula.sensorDIST[j][0] - particula.sensorDIST_apparture + angle)
                            posy = particula.sensorDIST[j][2] - leitura*sin(particula.ori +  particula.sensorDIST[j][0] - particula.sensorDIST_apparture + angle)
                
                            idx = int(self.x_offset+self.map_scale_factor*posx)
                            idy = int(self.y_offset+self.map_scale_factor*posy)
                            distancemap_value = self.distance_map_full[idy,idx]
                            pesosDIST.append(distancemap_value**2)
                        # print(pesosDIST)
                        # Utilizar desvio padrão para o calculo dos pesos
                        value = 0.5*min(pesosDIST)

                    pesosSENS.append(value)
                # if i == 3: 
                #     print(pesosSENS)
                particula.weight +=  exp((-pesosSENS[0]) + exp(-pesosSENS[1]) + exp(-pesosSENS[2]) + exp(-pesosSENS[3]))
                if particula.weight > self.max_w: self.max_w = particula.weight

        # -> Esta versao do metodo 4 apesar de usar numpy é bastante mais lenta
        # if metodo == 4: 
        #     acceptable_limit = 3  # IMPORTANTE
        #     for i, particula in enumerate(self.particulas):
        #         pesosSENS = np.zeros(len(particula.sensorDIST))
        #         for j, v in enumerate(particula.sensorDIST):
        #             leitura = DISTsens[j]
        #             value = particula.weight
        #             pesosDIST = np.zeros(particula.num_endpoints)
        #             if leitura <= acceptable_limit:
        #                 for k in range(particula.num_endpoints):
        #                     angle = particula.endpoints_angle * k
        #                     posx = particula.sensorDIST[j][1] + leitura * np.cos(
        #                         particula.ori + particula.sensorDIST[j][0] - particula.sensorDIST_apparture + angle)
        #                     posy = particula.sensorDIST[j][2] - leitura * np.sin(
        #                         particula.ori + particula.sensorDIST[j][0] - particula.sensorDIST_apparture + angle)

        #                     idx = int(self.x_offset + self.map_scale_factor * posx)
        #                     idy = int(self.y_offset + self.map_scale_factor * posy)
        #                     distancemap_value = self.distance_map_full[idy, idx]
        #                     pesosDIST[k] = distancemap_value
        #                 value = np.min(pesosDIST) ** 2

        #             pesosSENS[j] = value

        #         particula.weight += np.exp((-pesosSENS[0]) + np.exp(-pesosSENS[1]) + np.exp(-pesosSENS[2]) + np.exp(-pesosSENS[3]))
        #         if particula.weight > self.max_w:
        #             self.max_w = particula.weight

        # Metodo 5 (TENHO DE PASSAR PARA NUMPY)
        elif metodo == 5: # (Metodo 5 = Metodo 4 + Metodo 3)
            raio_robot = 0.5                # Metodo 2: info
            acceptable_limit = 2            # Metodo 4: IMPORTANTE
            for i,particula in enumerate(self.particulas): 
                # Metodo 3
                DISTsens_min = min(DISTsens) + raio_robot
                # print(DISTsens_min)
                distmappartvalue = self.distance_map_full[
                    self.y_offset + int(self.map_scale_factor*particula.y),
                    self.x_offset + int(self.map_scale_factor*particula.x)
                ]
                # print(distmappartvalue)
                dummy2 = 0.5*(DISTsens_min - distmappartvalue)**2
                aumento_1 =  (exp(-dummy2))
                # Metodo 4
                pesosSENS = []
                value = particula.weight
                for j,v in enumerate(particula.sensorDIST):
                    leitura = DISTsens[j]
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
                        value = 0.5*min(pesosDIST)**2

                    pesosSENS.append(value)
                # if i == 3: 
                #     print(pesosSENS)
                particula.weight +=  exp((-pesosSENS[0]) + exp(-pesosSENS[1]) + exp(-pesosSENS[2]) + exp(-pesosSENS[3])) + aumento_1

                # if value != particula.weight:
                #     particula.weight +=  exp((-pesosSENS[0]) + exp(-pesosSENS[1]) + exp(-pesosSENS[2]) + exp(-pesosSENS[3])) + aumento_1
                if particula.weight > self.max_w: self.max_w = particula.weight
    # ------------------------------------ Fim dos métodos para calculo de pesos ------------------------------------------
                

    # ------------------------------------------- Normalizar os pesos -----------------------------------------------------   
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

            sum_squarednormalized_weights += normalized_weight**2      # sum(norm_weight[i]^2)

        # Store the sum of all squared normalized weights
        self.sum_squarednormalized_weights = sum_squarednormalized_weights
        self.effective_num_particles = 1 / sum_squarednormalized_weights # By "Probabilistic Robotics" de Sebastian Thrun, Wolfram Burgard e Dieter Fox.
    # ------------------------------------------- Fim da normalizacao de pesos -----------------------------------------------
   
    
   
    # ------------------------------------------ CLUSTER DBSCAN ----------------------------------
    def cluster(self):
        # Armazenar as posições X e Y de todas as partículas
        # X = []
        # for p in self.particulas:
        #     X.append([p.x, p.y])
        # X = np.vstack(X)
        X = np.array([[p.x, p.y] for p in self.particulas])
        ori = np.array([p.ori for p in self.particulas])
        weight = np.array([p.weight for p in self.particulas])


        # Selecionar apenas os pontos dentro de um raio
        # D = squareform(pdist(X))
        # indices = np.where(np.any(D <= 0.9, axis=1))[0]
        # X = X[indices]
        # ori = ori[indices]

        # Executar o clustering por DBSCAN
        dbs = DBSCAN(eps=1, metric='l1', algorithm='auto') # default values: leaf_size=30, min_samples = 5
        dbs.fit(X)

        # Obter os rótulos das clusters e seus centróides
        labels = dbs.labels_

        # Encontrar os centroides e as médias de orientação
        centroids = []
        orientations = []
        weights = []
        covariances = []

        for label in np.unique(labels):
            if label == -1:  # descartar pontos que não foram classificados em uma cluster
                continue
            indices = np.where(labels == label)[0]
            
            points = X[indices]
            orientations_in_cluster = ori[indices]
            weights_in_cluster = weight[indices]

            centroid = points.mean(axis=0)
            orientation = np.arctan2(np.sin(orientations_in_cluster).mean(), np.cos(orientations_in_cluster).mean())
            weight_centroid = weights_in_cluster.sum()
            covariance = np.cov(points, rowvar=False, bias = True)
            
            centroids.append(centroid)
            orientations.append(orientation)
            weights.append(weight_centroid)
            covariances.append(covariance)
            
        # print(covariances)
        # print()
        
        self.centroides = centroids
        self.centroides_oris = orientations
        self.centroides_weights = weights
        self.centroides_cov = covariances
    
    # ---- Versoes antigas do cluster
    # def cluster(self):
    #     # Armazenar as posições X e Y de todas as partículas
    #     X = np.zeros((self.num_particles, 2))
    #     for i in range(self.num_particles):
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
    #     # for i in range(self.num_particles):
    #         # self.particulas[i].x = centroids[labels[i], 0]
    #         # self.particulas[i].y = centroids[labels[i], 1]


    
    # def cluster(self):
    #     # Armazenar as posições X e Y de todas as partículas
    #     X = np.array([[p.x, p.y] for p in self.particulas])
    #     ori = np.array([p.ori for p in self.particulas])
    #     weight = np.array([p.weight for p in self.particulas])

    #     # Executar o clustering por meanshift
    #     # ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True, n_jobs=-1)
    #     ms = DBSCAN(eps=1, metric='l1', algorithm='auto', leaf_size=30)
    #     ms.fit(X)

    #     # Obter os rótulos das clusters e seus centróides
    #     labels = ms.labels_

    #     # Encontrar os centroides, médias de orientação e médias de peso
    #     centroids = []
    #     orientations = []
    #     weights = []
    #     for label in np.unique(labels):
    #         if label == -1:  # descartar pontos que não foram classificados em uma cluster
    #             continue
    #         mask = (labels == label)
    #         points = X[mask]
    #         orientations_in_cluster = ori[mask]
    #         # weights_in_cluster = weight[mask]
    #         centroid = points.mean(axis=0)

    #         orientation = np.arctan2(np.sin(orientations_in_cluster).mean(), np.cos(orientations_in_cluster).mean())
    #         # weight = weights_in_cluster.mean()
    #         centroids.append(centroid)
    #         orientations.append(orientation)
    #         # weights.append(weight)

    #     # centroids é uma lista de tuplas (x, y) com as coordenadas dos centroides
    #     # orientations é uma lista com as médias de orientação de cada cluster
    #     # weights é uma lista com as médias de peso de cada cluster

    #     # print(np.max(labels))
    #     self.centroides = centroids
    #     self.centroides_oris = orientations
    # ------------------------------------------ Fim da funcao de CLUSTER ----------------------------------

    # ------------------------- Obter posiçao final com todas as particulas considerando so pesos (Nao utilizada, apenas para primeiro teste)
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

