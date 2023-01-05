#import particles
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

class filtroParticulas():
    def __init__(self,n_part=4000, mapmax_x=28, mapmax_y=14):
        self.n_part = n_part
        self.s_w = n_part
        self.max_w = 1
        self.last_motors = (0,0)
        self.sum_square_w = 0
        self.particulas = []
        self.mapmax_x = mapmax_x
        self.mapmax_y = mapmax_y
        self.norm_weights = []
        self.max_normalized_weight = 1/self.n_part
        self.areas = self.getAreas()
        self.map = self.getMap()
        print(self.map)

        for i in range (self.n_part):
            self.particulas.append( particula( random.random() * (mapmax_x), random.random() * (mapmax_y), random.random()*360, 1))

            #self.particulas.append((random.random() * ((mapmax_x-1)+0.5), random.random() * (mapmax_y-1)+0.5, random.random()*360 - 180, 1))
            #self.particulas.append((5, 8, 0))
            
            #self.weights.append(1)
            self.norm_weights.append(1/self.n_part)
            #self.ori.append(self.particulas[i][-1])
            #self.ori.append(0)

        # Tamanho imagem = 28 x 14 -> 1120 x 560 = simulação*40         x40  
        self.immax_x = 40 * self.mapmax_x
        self.immax_y = 40 * self.mapmax_y
        self.img = np.zeros((self.immax_y,self.immax_x,3), np.uint8)

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
        for l in range(10*self.mapmax_y):
            collum = []
            for c in range(10*self.mapmax_x):
                sum = 0
                for j,k in enumerate(self.areas):

                    if c >= 10*float(k[0][0]) and c+1 <= 10*float(k[1][0]) and l >= 10*float(k[0][1]) and l+1 <= 10*float(k[1][1]):
                        sum += 1
                        break

                if sum != 0:
                    arr = np.concatenate((arr,[True]))
                else:
                    arr = np.concatenate((arr,[False]))

            # line.append(collum)

        return arr


    def odometry_move(self, motors, motors_noise):       
        self.motors = motors
        updatedmotors = False
        for i,particula in enumerate (self.particulas):
            # calculate estimated power apply
            out_l = (self.motors[0] + self.last_motors[0]) / 2
            out_r = (self.motors[1] + self.last_motors[1]) / 2
            
            out_l = random.gauss(out_l, 3*motors_noise*out_l)   # out_l tem um erro de 1,5%
            out_r = random.gauss(out_r, 3*motors_noise*out_r)    # out_r tem um erro de 1,5%

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
            
            if not updatedmotors:
                self.last_motors = (out_l,out_r)
                updatedmotors = True

    def resample(self):
        #print(f'sum(S_W**2) -> {self.sum_square_w}')
        #print(1./self.sum_square_w)
        n = self.n_part
        if (1./self.sum_square_w < n):
        #if False:
        #if True:
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

    def w_calc(self, flag_loc,robot_compass):
        self.max_w = 1
        if flag_loc == None : return
        
        for i,particula in enumerate(self.particulas):
            # Out of bounds test
            if (particula.x > self.mapmax_x-0.5 or particula.x < 0.5 or particula.y > self.mapmax_y-0.5 or particula.y < 0.5):   # Está fora do mapa/bate na parede
               
                out_o_b = True
                particula.weight = 0.02

            else: # Está dentro do mapa
                out_o_b = False
           
            # Se nao estiver out of bounds    
            if not out_o_b:        # Estando dentro do mapa 
                # Indice da posicao do sensor do centro (X,Y)
                center_sensor_index = ( int(10*particula.sensor_center_posx), int(10*particula.sensor_center_posy) )
                
                # Indice no mapa = (Y-1)*mapLineSIZE + X-1
                center_sensor_mapindex = (center_sensor_index[1]-1)*10*self.mapmax_x + center_sensor_index[0]
                # print(f'center_sensor_index: {center_sensor_index} \n MapIndex: {center_sensor_mapindex}')
                
                # Robot real Dentro da area AZUL
                if flag_loc == 1:
                    # Comparar com o mapa e atribuir peso
                    if self.map[center_sensor_mapindex]:
                        particula.weight += 0.98

                    else:                   
                        particula.weight += 0.02

                # Robot real Fora da area AZUL
                else:               
                    if self.map[center_sensor_mapindex]:
                        particula.weight += 0.02 

                    else:
                        particula.weight += 0.98


            if particula.weight > self.max_w:
                self.max_w = particula.weight
            
    def w_norm(self):
        s_w = 0
        ssw = 0
        
        for i,v in enumerate(self.particulas):
            s_w += v.weight

        self.s_w = s_w
        
        
        for i,v in enumerate(self.particulas): 
            nw = v.weight/self.s_w
            self.norm_weights[i] = nw
            if nw > self.max_normalized_weight:
                self.max_normalized_weight = nw
            ssw += self.norm_weights[i]**2 

        self.sum_square_w = ssw
 
    def drawParticles(self):
        #print(self.max_w)

        for i,particula in enumerate(self.particulas):
            x = int(40*particula.x)
            y = int(40*particula.y)
            ori = particula.ori
            x_sensor_centro = int(40*particula.sensor_center_posx)
            y_sensor_centro = int(40*particula.sensor_center_posy)


            # cv2.circle(self.img, (x,y), 5, (0,0.95*254+0.05*254*(self.norm_weights[i]/self.max_normalized_weight),0.05*254+0.95*254*(self.norm_weights[i]/self.max_normalized_weight)), -1)
            if particula.weight > 0.5:
                cv2.circle(self.img, (x,y), 5, (0,200,0), -1)

            else:
                cv2.circle(self.img, (x,y), 2, (0,0,200), -1)

            cv2.line( self.img, (x,y), (x_sensor_centro, y_sensor_centro), (200,150,100),2)
            cv2.circle(self.img, (x_sensor_centro,y_sensor_centro), 1, (0,0,253), -1)


    def drawReal(self,x,y,ori):
        if x == None or y == None or ori == None : return
        cx = int(40*x)
        cy = int(abs(40*(14-y)))
        x_sensor_centro = int(40*(x + 0.438*cos(radians(ori))))
        y_sensor_centro = int(abs(40*(14 - y - 0.438*sin(radians(ori)))))

        
        cv2.circle(self.img,(cx,cy), 20, (150,150,0), -1) # Circulo centrado no centro do robot real
        cv2.line( self.img, (cx,cy), (cx+int(20*cos(radians(ori))), cy-int(20*sin(radians(ori)))), (255,0,0),2) # Linha do centro do robot direcionada segundo orientaçao
        cv2.circle(self.img, (x_sensor_centro,y_sensor_centro), 1, (0,0,253), -1)

        #print(f'\nGPS: x: {40*x+40*cos(ori)}   y: {40*y+40*sin(ori)}   theta: {ori}')

    def drawMap(self):
        # See Only map
        # b = self.map.reshape(10*self.mapmax_y, 10*self.mapmax_x)
        # plt.imshow(b)
        # plt.show()

        for j,k in enumerate(self.areas):
            cv2.rectangle(self.img,(int(40*k[0][0]),int(40*k[0][1])),(int(40*k[1][0]),int(40*k[1][1])),(255,0,0),-1) 



    def showImg(self):
        cv2.imshow("img",self.img)
        cv2.waitKey(25)

    def clearImg(self):
        self.img = np.zeros((560,1120,3), np.uint8)

