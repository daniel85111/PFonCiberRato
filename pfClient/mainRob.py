
import sys
from croblink import *
from math import *
import xml.etree.ElementTree as ET
#import particles
import particleFilter
import numpy as np
import random
import cv2
import timeit

CELLROWS=7
CELLCOLS=14

class MyRob(CRobLinkAngs):
    def __init__(self, rob_name, rob_id, angles, host):
        CRobLinkAngs.__init__(self, rob_name, rob_id, angles, host)
        self.rob_name = rob_name
        
        self.motors = (0.0,0.0)
        self.last_motors = (0.0,0.0)

        self.flag_loc = None

        # Position
        self.firstRun = 1
        self.initialx = None
        self.initialy = None
        self.x = None
        self.y = None
        self.posx = None
        self.posy = None
        self.x_od_pos = 0
        self.y_od_pos = 0
        self.ori= 0
        self.robot_diameter = 1
        self.border = None

        self.x_posmin = 0
        self.y_posmin = 0
        self.x_posmax = 0
        self.y_posmax = 0


        # Lista de particulas
        self.mapmax_x = 28
        self.mapmax_y = 14
        self.particulas = particleFilter.filtroParticulas(n_part=5000)
        #self.i = 0
        #self.inputfilter = []
        
        
        
        # Tamanho imagem = 28 x 14 -> 1120 x 560 = simulação*40         x40  
        self.immax_x = 1120
        self.immax_y = 560
        self.img = np.zeros((self.immax_y,self.immax_x,3), np.uint8)

    def setMap(self, labMap):
        self.labMap = labMap

    def printMap(self):
        for l in reversed(self.labMap):
            print(''.join([str(l) for l in l]))

    def run(self):
        if self.status != 0:
            print("Connection refused or error")
            quit()

        state = 'stop'
        stopped_state = 'run'

        while True:
            start = timeit.default_timer()
            self.readSensors()
            if self.measures.gpsReady:
                self.x = self.measures.x
                self.y = self.measures.y
            
            if self.measures.compassReady:
                self.ori= self.measures.compass
            
            if self.measures.endLed:
                print(self.rob_name + " exiting")
                self.motors = (0.0,0.0)
                quit()
                

            if state == 'stop' and self.measures.start:
                state = stopped_state
                self.motors = (0.0,0.0)
                

            if state != 'stop' and self.measures.stop:
                stopped_state = state
                state = 'stop'
                self.motors = (0.0,0.0)

            if state == 'run':
                if self.measures.visitingLed==True:
                    state='wait'
                if self.measures.ground==0:
                    self.setVisitingLed(True);
                self.wander()
            elif state=='wait':
                self.setReturningLed(True)
                if self.measures.visitingLed==True:
                    self.setVisitingLed(False)
                if self.measures.returningLed==True:
                    state='return'
                self.driveMotors(0.0,0.0)
                self.motors = (0.0,0.0)
            elif state=='return':
                if self.measures.visitingLed==True:
                    self.setVisitingLed(False)
                if self.measures.returningLed==True:
                    self.setReturningLed(False)
                self.wander()

            if self.firstRun:                               # Primeiro movimento
                if self.x is not None:
                    self.initialx = self.x                              # Receber coordenada X do GPS
                    #print(self.initialx )
                if self.y is not None:
                    self.initialy = self.y                              # Receber coordenada Y do GPS
                    #print(self.initialy )

                if self.y is not None and self.x is not None:
                    self.firstRun = 0                                   # Já deu o arranque

            # real pos
            if self.firstRun == 0:                          # Se já arrancou
                self.posx = self.measures.x - self.initialx + 5                  # Coordenada X em relação á posicao inicial
                self.posy = self.measures.y - self.initialy + 6          # Coordenada Y em relação à posição inicial 
            
            self.odometry_move(self.motors)                         # Movimento calculado por odometria
            self.particulas.odometry_move(self.motors, self.motorsNoise)              # Mover particulas 
            self.particulas.w_calc(self.flag_loc,self.measures.compass)                        # Calcular pesos de cada particula
            self.particulas.w_norm()                                # Normalizar peso de cada particula
            stop3 = timeit.default_timer()              
            self.particulas.resample()                              # Resample de particulas
            stop = timeit.default_timer()
            
            # Imagem Open CV
            self.particulas.clearImg()                              # Limpar imagem visualizada
            self.particulas.drawMap()                               # Desenhar mapa 
            self.particulas.drawReal(self.posx,self.posy,self.ori)  # Desenhar posição real do robot
            self.particulas.drawParticles()                         # Desenhar todas as particulas
            self.particulas.showImg()                               # Mostrar imagem
            stop2 = timeit.default_timer()
            
            print(f'Elapsed time: {1000*(stop-start):.0f}ms\t Time for Image Show: {1000*(stop2-stop):.0f}\t Total: {1000*(stop2-start):.0f}\t Time resample(): {1000*(stop-stop3):.0f}')

                

    def wander(self):
        

        center_id = 0
        left_id = 1
        right_id = 2
        back_id = 3

        if    self.measures.irSensor[center_id] > 5.0\
           or self.measures.irSensor[left_id]   > 5.0\
           or self.measures.irSensor[right_id]  > 5.0\
           or self.measures.irSensor[back_id]   > 5.0:
            #print('Rotate left')
            #self.driveMotors(-0.1,+0.1)
            lpow = -0.1
            rpow = +0.1
            self.motors = (lpow, rpow) 
            self.driveMotors(lpow,rpow)
            # if self.firstRun == 0:                          # Se já arrancou
            #     self.posx = self.measures.x - self.initialx + 5                  # Coordenada X em relação á posicao inicial
            #     self.posy = self.measures.y - self.initialy + 6          # Coordenada Y em relação à posição inicial 


        elif self.measures.irSensor[left_id]> 2.7:
            #print('Rotate slowly right')
            #self.driveMotors(0.1,0.0)
            lpow = 0.1
            rpow = 0.0
            self.motors = (lpow, rpow)
            self.driveMotors(lpow,rpow)
            # if self.firstRun == 0:                          # Se já arrancou
            #     self.posx = self.measures.x - self.initialx + 5                  # Coordenada X em relação á posicao inicial
            #     self.posy = self.measures.y - self.initialy + 6          # Coordenada Y em relação à posição inicial 


        elif self.measures.irSensor[right_id]> 2.7:
            #print('Rotate slowly left')
            #self.driveMotors(0.0,0.1)
            lpow = 0.0
            rpow = 0.1
            self.motors = (lpow, rpow)
            self.driveMotors(lpow,rpow)
            # if self.firstRun == 0:                          # Se já arrancou
            #     self.posx = self.measures.x - self.initialx + 5                  # Coordenada X em relação á posicao inicial
            #     self.posy = self.measures.y - self.initialy + 6          # Coordenada Y em relação à posição inicial 

            

        else:
            #$print('Go')
            if (self.measures.stop) :
                lpow = 0.0
                rpow = 0.0
                self.motors = (lpow, rpow)
                self.driveMotors(self.motors)


            else:
                

                #print(f'Xg: {self.posx:.2f}    Yg: {self.posy:.2f}    thetag: {self.ori}')

                lpow = 0.1
                rpow = 0.1
                self.motors = (lpow, rpow) 
                self.driveMotors(lpow,rpow)                     # Andar com velocidade constante (L = 0.1, R = 0.1)

        
        sens = list(map(int, self.measures.lineSensor))         # Linha de Sensores
        #print(f'\n{sens}\n')
        left_1,left_2,left_,center,right_3,right2,right_1 = sens


        if (center == 1):
            self.flag_loc = 1
        else:
            self.flag_loc = 0

        
        #print(self.motorsNoise)

                

    def odometry_move(self, motors):
        
        self.motors = motors        

        # calculate estimated power apply
        out_l = (self.motors[0] + self.last_motors[0]) / 2
        out_r = (self.motors[1] + self.last_motors[1]) / 2
        
        #out_l = random.gauss(out_l, 0.0015*out_l)   # out_l tem um erro de 1,5%
        #out_r = random.gauss(out_r, 0.0015*out_r)    # out_r tem um erro de 1,5%
        if out_l > 0.15:
            out_l = 0.15
        if out_r > 0.15:
            out_r = 0.15
            
        # pos
        lin = (out_l + out_r) / 2
        x = self.x_od_pos + 2*(lin * cos(radians(self.ori)))
        y = self.y_od_pos - 2*(lin * sin(radians(self.ori)))
        
        rot = (out_r - out_l) / self.robot_diameter         # self.robot_diameter = 1 
        self.ori = degrees(radians(self.ori) + rot) % 360
        ori = self.ori
        #print(f'x0: {x:.2f}    y0: {y:.2f}    theta0: {ori}' )

        self.x_od_pos = x # Posicao X por odometria
        self.y_od_pos = y # Posicao Y por odometria


            

class Map():
    def __init__(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        
        self.labMap = [[' '] * (CELLCOLS*2-1) for i in range(CELLROWS*2-1) ]
        i=1
        for child in root.iter('Row'):
           line=child.attrib['Pattern']
           row =int(child.attrib['Pos'])
           if row % 2 == 0:  # this line defines vertical lines
               for c in range(len(line)):
                   if (c+1) % 3 == 0:
                       if line[c] == '|':
                           self.labMap[row][(c+1)//3*2-1]='|'
                       else:
                           None
           else:  # this line defines horizontal lines
               for c in range(len(line)):
                   if c % 3 == 0:
                       if line[c] == '-':
                           self.labMap[row][c//3*2]='-'
                       else:
                           None
               
           i=i+1


rob_name = "pClient1"
host = "localhost"
pos = 1
mapc = None

for i in range(1, len(sys.argv),2):
    if (sys.argv[i] == "--host" or sys.argv[i] == "-h") and i != len(sys.argv) - 1:
        host = sys.argv[i + 1]
    elif (sys.argv[i] == "--pos" or sys.argv[i] == "-p") and i != len(sys.argv) - 1:
        pos = int(sys.argv[i + 1])
    elif (sys.argv[i] == "--robname" or sys.argv[i] == "-r") and i != len(sys.argv) - 1:
        rob_name = sys.argv[i + 1]
    elif (sys.argv[i] == "--map" or sys.argv[i] == "-m") and i != len(sys.argv) - 1:
        mapc = Map(sys.argv[i + 1])
    else:
        print("Unkown argument", sys.argv[i])
        quit()

if __name__ == '__main__':
    rob=MyRob(rob_name,pos,[0.0,60.0,-60.0,180.0],host)
    if mapc != None:
        rob.setMap(mapc.labMap)
        rob.printMap()
    
    rob.run()
