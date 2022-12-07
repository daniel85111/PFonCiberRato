#import particles
import numpy as np
from math import *
import random
import cv2

class particula():
    def __init__(self, x, y, ori, w):
        self.x = x
        self.y = y
        self.ori = ori
        self.weight = w

class filtroParticulas():
    def __init__(self,n_part=2000, mapmax_x=28, mapmax_y=14):
        self.n_part = n_part
        self.s_w = n_part
        self.max_w = 1
        self.last_motors = (0,0)
        self.sum_square_w = 0
        self.particulas = []
        self.mapmax_x = mapmax_x
        self.mapmax_y = mapmax_y
        self.norm_weights = []
        self.areas = [

            # (13.5,4) a (20,9.5)
            [[13.5, 14-9.5], [20, 14-4]],

            # (4.5,4) a (10,9.5)
            [[4.5, 14-9.5], [10, 14-4]],
            
            # (0,2) a (28,2.5)
            [[0, 14-2.5], [28, 14-2]],
            
            # (0,10) a (28,11)
            [[0, 14-11], [28, 14-10]],
            
            # (25.5,0) a (28,14)
            [[25.5, 14-14], [28, 14-0]],
            
            # (2,0) a (4,14)
            [[2, 14-14], [4, 14-0]]          
            ]
        for i in range (self.n_part):
            self.particulas.append(particula(random.random() * ((mapmax_x-1)+0.5), random.random() * (mapmax_y-1)+0.5, random.random()*360 - 180, 1))

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

    def odometry_move(self, motors):       
        self.motors = motors

        for i,v in enumerate (self.particulas):
            # calculate estimated power apply
            out_l = (self.motors[0] + self.last_motors[0]) / 2
            out_r = (self.motors[1] + self.last_motors[1]) / 2
            
            out_l = random.gauss(out_l, 0.075*out_l)   # out_l tem um erro de 1,5%
            out_r = random.gauss(out_r, 0.075*out_r)    # out_r tem um erro de 1,5%

            if out_l > 0.15:
                out_l = 0.15

            if out_r > 0.15:
                out_r = 0.15
            
            # pos
            lin = (out_l + out_r) / 2
            x = v.x + (lin * cos(radians(v.ori)))
            y = v.y - (lin * sin(radians(v.ori)))

            rot = out_r - out_l # / self.robot_diameter ( = 1 )
            ori = degrees(radians(v.ori) + rot) % 360


            v.x = x
            v.y = y
            v.ori = ori
            
            
            self.last_motors = (out_l,out_r)
        

    def resample(self):
        #print(f'sum(S_W**2) -> {self.sum_square_w}')
        print(1./self.sum_square_w)
        n = 0.95 * self.n_part
        if (1./self.sum_square_w < n):
        #if False:
        #if True:
            print("---------- ReSampling!!!!!- -----------")
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

    def w_calc(self, flag_loc):
        self.max_w = 1
           
        for i,v in enumerate(self.particulas):
            if (v.x > self.mapmax_x-0.4 or v.x < 0+0.4 or v.y > self.mapmax_y-0.4 or v.y < 0+0.4):   # Está fora do mapa
                out_o_b = True
                v.weight = 0.5

            else: # Está dentro do mapa
                out_o_b = False
                
            if not out_o_b:        # Estando dentro do mapa
                sum = 0  
                if flag_loc == 1:       # Robot real Dentro da area
                    #if ( (v.x < 20 and v.x>13.5 and v.y < 9.5 and v.y > 4) or (v.x > 4.5 and v.x < 10 and v.y < 9.5 and v.y > 4) ):   # Particula dentro da area
                    for j,k in enumerate(self.areas):
                        if ( (v.x > k[0][0] and v.x < k[1][0] and v.y > k[0][1] and v.y < k[1][1]) ):
                            sum += 1

                    if sum > 0:
                        v.weight += 1
                    else:                   # Particula fora da area
                        v.weight -= 5

                else:               # Robot real Fora da area
                    #if ( (v.x < 20 and v.x>13.5 and v.y < 9.5 and v.y > 4) or (v.x > 4.5 and v.x < 10 and v.y < 9.5 and v.y > 4) ):     # Particula fora da area
                    for j,k in enumerate(self.areas):
                        if ( (v.x > k[0][0] and v.x < k[1][0] and v.y > k[0][1] and v.y < k[1][1]) ):
                            v.weight -= 5
                    if sum > 0:
                        v.weight -= 5 
                    else:
                        v.weight += 1
            if v.weight < 1:
                v.weight = 0.5

            if v.weight > self.max_w:
                self.max_w = v.weight
            
    def w_norm(self):
        s_w = 0
        ssw = 0
        
        for i,v in enumerate(self.particulas):
            s_w += v.weight

        self.s_w = s_w
        
        
        for i,v in enumerate(self.particulas): 
            self.norm_weights[i] = v.weight/self.s_w
            ssw += self.norm_weights[i]**2 

        self.sum_square_w = ssw
 
    def drawParticles(self):
        print(self.max_w)

        for i,v in enumerate(self.particulas):
            x = int(40*v.x)
            y = int(40*v.y)
            ori = v.ori

            if v.weight < 1:
                cv2.circle(self.img, (x,y), 5, (0,0,255), -1)
                cv2.line( self.img, (x,y), (x+int(10*cos(radians(ori))), y-int(10*sin(radians(ori)))), (255,0,0),2)

            else:
                cv2.circle(self.img,(x,y), 5, (0,255,0), -1)
                cv2.line( self.img, (x,y), (x+int(10*cos(radians(ori))), y-int(10*sin(radians(ori)))), (255,0,0),2)

    def drawReal(self,x,y,ori):
        cx = int(40*x)
        cy = int(40*y)
        
        cv2.circle(self.img,(cx,cy), 20, (150,150,0), -1) # Circulo centrado no centro do robot real
        cv2.line( self.img, (cx,cy), (cx+int(20*cos(radians(ori))), cy-int(20*sin(radians(ori)))), (255,0,0),2) # Linha do centro do robot direcionada segundo orientaçao
        #print(f'\nGPS: x: {40*x+40*cos(ori)}   y: {40*y+40*sin(ori)}   theta: {ori}')

    def drawMap(self):
        

        for j,k in enumerate(self.areas):
            cv2.rectangle(self.img,(int(40*k[0][0]),int(40*k[0][1])),(int(40*k[1][0]),int(40*k[1][1])),(255,0,0),-1) 



    def showImg(self):
        cv2.imshow("img",self.img)
        cv2.waitKey(25)

    def clearImg(self):
        self.img = np.zeros((560,1120,3), np.uint8)

