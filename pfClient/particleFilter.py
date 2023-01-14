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

        print(self.map)

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
        array = self.parseXML("../Labs/2223-pf/C1-lab.xml")
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
        arr = np.array([],dtype=bool)
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
                    arr = np.concatenate((arr,[True]))
                else:
                    arr = np.concatenate((arr,[False]))

            # line.append(collum)

        return arr,scale


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
        n = 0.999*self.n_part
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

    def weights_calculation(self, sens):
        left1,left2,left3,center,right3,right2,right1 = sens
        
        # Mudar apenas este valor
        bigger_weightvalue = 0.75
        smaller_weightvalue = 1-bigger_weightvalue
       
        if sens == None : return
        
        for i,particula in enumerate(self.particulas):
            # Out of bounds test
            if (particula.x > self.mapmax_x-0.5 or particula.x < 0.5 or particula.y > self.mapmax_y-0.5 or particula.y < 0.5):   # Está fora do mapa/bate na parede
               
                out_o_b = True
                # particula.weight = smaller_weightvalue

            else: # Está dentro do mapa
                out_o_b = False
           
            # Se nao estiver out of bounds    
            if not out_o_b:        # Estando dentro do mapa 
                # Indice da posicao do sensor do centro (X,Y) no mapa com fator de escala
                center_sensor_index = ( int(self.map_scale_factor*particula.sensor_center_posx), int(self.map_scale_factor*particula.sensor_center_posy) )

                left1_sensor_index = ( int(self.map_scale_factor*particula.sensor_L1_posx), int(self.map_scale_factor*particula.sensor_L1_posy) )
                left2_sensor_index = ( int(self.map_scale_factor*particula.sensor_L2_posx), int(self.map_scale_factor*particula.sensor_L2_posy) )
                left3_sensor_index = ( int(self.map_scale_factor*particula.sensor_L3_posx), int(self.map_scale_factor*particula.sensor_L3_posy) )
 
                right1_sensor_index = ( int(self.map_scale_factor*particula.sensor_R1_posx), int(self.map_scale_factor*particula.sensor_R1_posy) )
                right2_sensor_index = ( int(self.map_scale_factor*particula.sensor_R2_posx), int(self.map_scale_factor*particula.sensor_R2_posy) )
                right3_sensor_index = ( int(self.map_scale_factor*particula.sensor_R3_posx), int(self.map_scale_factor*particula.sensor_R3_posy) )
                
                
                # Indice no mapa = (Y-1)*mapLineSIZE + X-1
                center_sensor_mapindex = self.map_scale_factor*self.mapmax_x*(center_sensor_index[1]) + center_sensor_index[0]
                
                left1_sensor_mapindex = self.map_scale_factor*self.mapmax_x*(left1_sensor_index[1]) + left1_sensor_index[0]
                left2_sensor_mapindex = self.map_scale_factor*self.mapmax_x*(left2_sensor_index[1]) + left2_sensor_index[0]
                left3_sensor_mapindex = self.map_scale_factor*self.mapmax_x*(left3_sensor_index[1]) + left3_sensor_index[0]
               
                right1_sensor_mapindex = self.map_scale_factor*self.mapmax_x*(right1_sensor_index[1]) + right1_sensor_index[0]
                right2_sensor_mapindex = self.map_scale_factor*self.mapmax_x*(right2_sensor_index[1]) + right2_sensor_index[0]
                right3_sensor_mapindex = self.map_scale_factor*self.mapmax_x*(right3_sensor_index[1]) + right3_sensor_index[0]

                # print(f'center_sensor_index: {center_sensor_index} \n MapIndex: {center_sensor_mapindex}')

                # CENTER ------------------------------------------------------------
                # Sensor center real Dentro da area AZUL
                if center == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[center_sensor_mapindex]:
                        particula.weight += bigger_weightvalue

                    else:                   
                        particula.weight += smaller_weightvalue

                # Sensor center real Fora da area AZUL
                else:               
                    if self.map[center_sensor_mapindex]:
                        particula.weight += smaller_weightvalue 

                    else:
                        particula.weight += bigger_weightvalue
                # END CENTER ------------------------------------------------------------

                # RIGHT 1 --------------------------------------------------------------
                # Sensor right1 real Dentro da area AZUL
                if right1 == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[right1_sensor_mapindex]:
                        particula.weight += bigger_weightvalue

                    else:                   
                        particula.weight += smaller_weightvalue

                # Sensor right1 real Fora da area AZUL
                else:               
                    if self.map[right1_sensor_mapindex]:
                        particula.weight += smaller_weightvalue 

                    else:
                        particula.weight += bigger_weightvalue
                # END RIGHT 1 ------------------------------------------------------------

                # RIGHT 2 ------------------------------------------------------------
                # Sensor right2 real Dentro da area AZUL
                if right2 == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[right2_sensor_mapindex]:
                        particula.weight += bigger_weightvalue

                    else:                   
                        particula.weight += smaller_weightvalue

                # Sensor right2 real Fora da area AZUL
                else:               
                    if self.map[right2_sensor_mapindex]:
                        particula.weight += smaller_weightvalue 

                    else:
                        particula.weight += bigger_weightvalue
                # END RIGHT 2 ------------------------------------------------------------

                # RIGHT 3 ------------------------------------------------------------
                # Sensor right3 real Dentro da area AZUL
                if right3 == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[right3_sensor_mapindex]:
                        particula.weight += bigger_weightvalue

                    else:                   
                        particula.weight += smaller_weightvalue

                # Sensor right2 real Fora da area AZUL
                else:               
                    if self.map[right3_sensor_mapindex]:
                        particula.weight += smaller_weightvalue 

                    else:
                        particula.weight += bigger_weightvalue  
                # END RIGHT 3 ------------------------------------------------------------     

                # LEFT 1 ------------------------------------------------------------
                 # Sensor left1 real Dentro da area AZUL
                if left1 == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[left1_sensor_mapindex]:
                        particula.weight += bigger_weightvalue

                    else:                   
                        particula.weight += smaller_weightvalue

                # Sensor left1 real Fora da area AZUL
                else:               
                    if self.map[left1_sensor_mapindex]:
                        particula.weight += smaller_weightvalue 

                    else:
                        particula.weight += bigger_weightvalue
                # END LEFT 1 ------------------------------------------------------------

                # LEFT 2 ------------------------------------------------------------
                # Sensor left2 real Dentro da area AZUL
                if left2 == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[left2_sensor_mapindex]:
                        particula.weight += bigger_weightvalue

                    else:                   
                        particula.weight += smaller_weightvalue

                # Sensor left2 real Fora da area AZUL
                else:               
                    if self.map[left2_sensor_mapindex]:
                        particula.weight += smaller_weightvalue 

                    else:
                        particula.weight += bigger_weightvalue
                # END LEFT 2 ------------------------------------------------------------

                # LEFT 3 ------------------------------------------------------------
                 # Sensor left3 real Dentro da area AZUL
                if left3 == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[left3_sensor_mapindex]:
                        particula.weight += bigger_weightvalue

                    else:                   
                        particula.weight += smaller_weightvalue

                # Sensor left3 real Fora da area AZUL
                else:               
                    if self.map[left3_sensor_mapindex]:
                        particula.weight += smaller_weightvalue 

                    else:
                        particula.weight += bigger_weightvalue
                # END LEFT 3 ------------------------------------------------------------

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
        self.biewer.drawMap()
        self.biewer.updateParticles(self.particulas)
        self.biewer.drawParticles()
        self.biewer.drawReal(real_posx,real_posy,ori, diameter)
        self.biewer.showImg()