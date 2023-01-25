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
        self.max_normalized_weight = 1/self.n_part # Para já não é utilizado 
        self.areas = self.getAreas()
        
        self.map, self.map_scale_factor = self.getMap()

        self.distance_map = self.getDistanceMap()
        np.savetxt('Dist MAP', self.distance_map, fmt='%2.1f', delimiter=', ')

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

        # Tamanho imagem = 28 x 14 -> 1120 x 560 = simulação*40         x40 
        self.imscale = 4*self.map_scale_factor
        self.immax_x = self.imscale * self.mapmax_x
        self.immax_y = self.imscale * self.mapmax_y
        self.img = np.zeros((self.immax_y,self.immax_x,3), np.uint8)

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
        return areas

    def getMap(self):
        arr = np.array([],dtype=np.uint8)
        scale = 10

        for l in range(scale*self.mapmax_y):
            # collum = []
            for c in range(scale*self.mapmax_x):
                sum = 0
                for j,k in enumerate(self.areas):

                    if c >= scale*float(k[0][0]) and c+1 <= scale*float(k[1][0]) and l >= scale*float(k[0][1]) and l+1 <= scale*float(k[1][1]):
                        sum += 1
                        break

                if sum != 0:
                    arr = np.concatenate((arr,[255]))
                else:
                    arr = np.concatenate((arr,[0]))

            # line.append(collum)

        return arr,scale

    def getDistanceMap(self):
        b = self.map.reshape(self.map_scale_factor*self.mapmax_y, self.map_scale_factor*self.mapmax_x)
        b = cv2.bitwise_not(b)
        b = b.astype(np.uint8)
        

        dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
        # print(dist)
        # plt.imshow(dist)
        # plt.show()
        return dist

    def odometry_move_particles(self, motors, motors_noise):       
        self.motors = motors
        noite_multiplier = 3
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
        n = 0.99*self.n_part
        #print(f'Resample condition: {1./self.sum_squarednormalized_weights:.2f} < {n:.2f} -----> {1./self.sum_squarednormalized_weights < n}')
        if (1./self.sum_squarednormalized_weights < n):
        # if True:
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
            # Out of bounds test
            if (particula.x > self.mapmax_x-0.5 or particula.x < 0.5 or particula.y > self.mapmax_y-0.5 or particula.y < 0.5):   # Está fora do mapa/bate na parede
               
                out_o_b = True
                # particula.weight = smaller_weightvalue

            else: # Está dentro do mapa
                out_o_b = False
           
            # Se nao estiver out of bounds    
            if not out_o_b:        # Estando dentro do mapa 
                # Indice da posicao do sensor do centro (X,Y) no mapa com fator de escala (arredondamento para inteiro, coordenadas)
                
                sensorDIST_center_index =( int(self.map_scale_factor*particula.sensorDIST_center_posx), int(self.map_scale_factor*particula.sensorDIST_center_posy) )
                sensorDIST_left_index = ( int(self.map_scale_factor*particula.sensorDIST_left_posx), int(self.map_scale_factor*particula.sensorDIST_left_posy) )
                sensorDIST_right_index = ( int(self.map_scale_factor*particula.sensorDIST_right_posx), int(self.map_scale_factor*particula.sensorDIST_right_posy) )
                sensorDIST_back_index = ( int(self.map_scale_factor*particula.sensorDIST_back_posx), int(self.map_scale_factor*particula.sensorDIST_back_posy) )

                # Distance map, each cell (index value) represents 0.1 radius, so (dist value * 10)
                particle_centerDIST = self.distance_map[sensorDIST_center_index[1],sensorDIST_center_index[0]]
                particle_leftDIST = self.distance_map[sensorDIST_left_index[1],sensorDIST_left_index[0]]
                particle_rightDIST = self.distance_map[sensorDIST_right_index[1],sensorDIST_right_index[0]]
                particle_backDIST = self.distance_map[sensorDIST_back_index[1],sensorDIST_back_index[0]]

                centerDIFF = abs(particle_centerDIST/10 - centerDIST)
                leftDIFF = abs(particle_leftDIST/10 - leftDIST)
                rightDIFF = abs(particle_rightDIST/10 - rightDIST)
                backDIFF = abs(particle_backDIST/10 - backDIST)

                particula.weight +=  exp(-centerDIFF)
                # particula.weight +=  ( 0.25 * exp(-centerDIFF) + 0.25 * exp(-leftDIFF) + 0.25 * exp(-rightDIFF) + 0.25 * exp(-backDIFF))

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

            if normalized_weight > self.max_normalized_weight:
                self.max_normalized_weight = normalized_weight # Para já não é utilizado 

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