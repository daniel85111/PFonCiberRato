
import sys
from croblink import *
from math import *
import xml.etree.ElementTree as ET

import particleFilter
import processmap
import viewercv
import numpy as np
import random
import timeit
import time

import threading
import queue
from pynput import keyboard

CELLROWS=7
CELLCOLS=14

class MyRob(CRobLinkAngs):
    def __init__(self, rob_name, rob_id, IRangles, host):
        CRobLinkAngs.__init__(self, rob_name, rob_id, IRangles, host)
        # Constants
        self.rob_name = rob_name
        self.robot_diameter = 1
        self.mapmax_x = 28
        self.mapmax_y = 14

        # Movement(sensors, actuators)
        self.backwards_counter = -1
        self.motors = (0.0,0.0)
        self.last_motors = (0.0,0.0)
        self.LINEsens = None
        self.DISTsens = None

        # Pos
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

        self.rotation = 0

        # define a variable to be updated by the thread
        self.keyboard_variable = ""

        # create a queue to hold input from the keyboard
        self.input_queue = queue.Queue()

        # create a thread to update the queue
        self.input_thread = threading.Thread(target=self.get_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        self.autorun = True
        self.last_runmode = 0
        self.filer_runmode = 1
        self.cluster = 0

        # Mapa
        self.mapa = processmap.map(lab_directory="../Labs/2223-pf/C2-lab.xml")

        # Viwer
        self.visualizer = viewercv.ViewParticles(self.mapa, grid_directory = "../Labs/2223-pf/C2-grid.xml")
        self.visualizer.drawMap(self.mapa)

        # Particle filter
        self.filtro_particulas = particleFilter.filtroParticulas(self.mapa, IRangles, n_part=3000)
        self.visualizer.drawParticles(self.filtro_particulas.particulas, self.filtro_particulas.max_w)
        self.visualizer.drawReal(self.x_od_pos, self.y_od_pos, self.ori, self.robot_diameter,  self.DISTsens, IRangles)
        self.visualizer.showImg()

    # define a function to get input from the keyboard and put it into a queue
    def get_input(self):
        def on_press(key):
            try:
                value = key.char
            except AttributeError:
                value = str(key)
            if value == "Key.esc":  # exit if the user presses the Esc key
                return False
            self.input_queue.put(value)

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def setMap(self, labMap):
        self.labMap = labMap

    def printMap(self):
        for l in reversed(self.labMap):
            print(''.join([str(l) for l in l]))

    def run(self, IRangles):
        if self.status != 0:
            print("Connection refused or error")
            quit()

        state = 'stop'
        stopped_state = 'run'

        while True:
            # start = timeit.default_timer()
            self.readSensors()
            try:
                # get input from the queue without blocking
                self.keyboard_variable = self.input_queue.get_nowait()
            except queue.Empty:
                pass
            # use the updated value of my_variable
            # print("my_variable:", self.keyboard_variable)
            if self.measures.gpsReady:
                self.x = self.measures.x
                self.y = self.measures.y
            
            if self.measures.compassReady:
                self.ori= radians(self.measures.compass)
            
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
                    self.setVisitingLed(True)
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
                if self.y is not None:
                    self.initialy = self.y                              # Receber coordenada Y do GPS

                if self.y is not None and self.x is not None:
                    self.firstRun = 0                                   # Já tem as coordenadas iniciais

            
            # Se já tem as coordenadas iniciais
            if self.firstRun == 0:                         
                self.posx = self.measures.x - self.initialx           # Coordenada X em relação á posicao inicial + Offset
                self.posy = self.measures.y - self.initialy           # Coordenada Y em relação à posição inicial + Offset
            
            self.odometry_move_robot(self.motors)                                 # Movimento do robot calculado por odometria
            self.update_particle_filter(IRangles, state,  weight_calculation_method = 2)    # Atualizar filtro de particulas
            



    def update_particle_filter(self, IRangles, state, weight_calculation_method = 3):
        time_move_particles = 0
        time_weight_calculation = 0
        time_weight_normalization = 0
        time_resample = 0
        time_drawing = 0
        if self.keyboard_variable == "f":
            self.keyboard_variable = ""
            if self.filer_runmode == 0:
                self.filer_runmode = 1
            else:
                self.filer_runmode = 0


        
        if(state != "stop"):
            self.visualizer.clearImg()
            start = time.perf_counter()
            self.filtro_particulas.odometry_move_particles(self.motors, self.motorsNoise, self.measures.collision)    # Mover particulas 
            end = time.perf_counter()
            time_move_particles = 1000*(end-start)
            
            start = time.perf_counter()
            if self.filer_runmode == 1:
                self.filtro_particulas.weights_calculation(self.LINEsens, self.DISTsens, weight_calculation_method)  # Calcular pesos de cada particula
            end = time.perf_counter()
            time_weight_calculation = 1000*(end-start)
            start = time.perf_counter()
            self.visualizer.drawMap(self.mapa)
            self.visualizer.drawParticles(self.filtro_particulas.particulas, self.filtro_particulas.max_w)
            

            self.visualizer.drawReal(self.posx, self.posy, self.ori, self.robot_diameter, self.DISTsens, IRangles)
            end = time.perf_counter()
            time_drawing = 1000*(end-start)

            start = time.perf_counter()
            if self.filer_runmode == 1:
                self.filtro_particulas.weights_normalization()            # Normalizar peso de cada particula            
            end = time.perf_counter()
            time_weight_normalization = 1000*(end-start)

            final_pose = self.filtro_particulas.getFinalPose()
            self.visualizer.drawFinalPose(final_pose)
            if self.filtro_particulas.centroides is not None:
                self.visualizer.drawCentroides(self.filtro_particulas.centroides)
            self.visualizer.showImg()


            start = time.perf_counter()
            if self.filer_runmode == 1:
                self.filtro_particulas.resample()      # Resample de particulas
            end = time.perf_counter()
            time_resample = 1000*(end-start)

            if self.keyboard_variable == "c":
                self.keyboard_variable = ""
                start = time.perf_counter()
                self.filtro_particulas.cluster()
                end = time.perf_counter()
                print(end-start)



        
        total = time_move_particles + time_weight_calculation + time_weight_normalization + time_resample + time_drawing


        # print(f'tempo total = {total:.0f} ms\n\t\t\t(Resample: {(time_resample):.1f} | W_update: {(time_weight_calculation):.1f} | W_norm: {(time_weight_normalization):.1f} | Od_Move: {(time_move_particles):.1f} | Image: {(time_drawing):.1f})')
        

    def wander(self):
        center_id = 0
        left_id = 1
        right_id = 2
        back_id = 3
        
        if self.keyboard_variable == "Key.space":
            if self.autorun:
                self.autorun = False
            else:
                self.autorun = True
            self.keyboard_variable = ""

        if self.autorun:
            if self.last_runmode == 0:
                self.last_runmode = 1
            if not self.measures.collision and self.backwards_counter < 0:
                if    self.measures.irSensor[center_id] > 4.0:
                    #print('Rotate left')
                    #self.driveMotors(-0.1,+0.1)
                    if self.rotation == 0:
                        lpow = -0.15
                        
                    else:
                        lpow = +0.15
                    rpow = -lpow
                    self.motors = (lpow, rpow) 
                    self.driveMotors(lpow,rpow)

                elif self.measures.irSensor[left_id]> 1.2:
                    #print('Rotate slowly right')
                    #self.driveMotors(0.1,0.0)
                    lpow = 0.1
                    rpow = 0.0
                    self.motors = (lpow, rpow)
                    self.driveMotors(lpow,rpow)

                elif self.measures.irSensor[right_id]> 1.2:
                    #print('Rotate slowly left')
                    #self.driveMotors(0.0,0.1)
                    lpow = 0.0
                    rpow = 0.1
                    self.motors = (lpow, rpow)
                    self.driveMotors(lpow,rpow)

                else:
                    self.rotation = random.randint(0,1)
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
            else:
                if self.backwards_counter < 0:
                    self.backwards_counter = 3
                lpow = -0.1
                rpow = -0.1
                self.motors = (lpow, rpow) 
                self.driveMotors(lpow,rpow)                     # Andar com velocidade constante (L = 0.1, R = 0.1)
                self.backwards_counter -= 1
        else:
            lpow, rpow = self.motors
            if self.last_runmode == 1:
                self.last_runmode = 0
                lpow = 0.0
                rpow = 0.0
            
            if self.keyboard_variable == "Key.up":
                self.keyboard_variable = ""
                if self.motors[0]<0.15:
                    lpow += 0.01
                if self.motors[1]<0.15:
                    rpow += 0.01
            
            if self.keyboard_variable == "Key.down":
                self.keyboard_variable = ""
                if self.motors[0]>-0.15:
                    lpow -= 0.01
                if self.motors[1]>-0.15:
                    rpow -= 0.01
            
            if self.keyboard_variable == "Key.left":
                self.keyboard_variable = ""
                if self.motors[0]>-0.15:
                    lpow -= 0.01
                if self.motors[1]<0.15:
                    rpow += 0.01

            if self.keyboard_variable == "Key.right":
                self.keyboard_variable = ""
                if self.motors[0]<0.15:
                    lpow += 0.01
                if self.motors[1]>-0.15:
                    rpow -= 0.01

            self.motors = (lpow, rpow)
            self.driveMotors(lpow,rpow) #Parar
        
        self.LINEsens = list(map(int, self.measures.lineSensor))         # Linha de Sensores

        # Sensor de distacia
        IRsens = self.measures.irSensor
        
        distancias = []
        for i,v in enumerate(IRsens):
            distancias.append(1/v)
            # if v != 0.0:
            #     distancia = 1/v
            #     if distancia < 6:
            #         distancias.append(distancia)
            #     else:
            #         distancias.append(10)
            # else:
            #     distancias.append(10)
        self.DISTsens = distancias
        # print(f'Center: {distancias[center_id]:.2f}\tLeft: {distancias[left_id]:.2f}\tRight: {distancias[right_id]:.2f}\tBack: {distancias[back_id]:.2f}')


    def odometry_move_robot(self, motors):
        
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
        x = self.x_od_pos + 2*(lin * cos(self.ori))
        y = self.y_od_pos + 2*(lin * sin(self.ori))
        
        rot = (out_r - out_l) / self.robot_diameter         # self.robot_diameter = 1 
        self.ori = (self.ori + rot)  % 6.28318530

        #print(f'x0: {x:.2f}    y0: {y:.2f}    theta0: {self.ori}' )

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
    IRangles = [0.0,75.0,-75.0,180.0]
    rob=MyRob(rob_name,pos,IRangles,host)
    if mapc != None:
        rob.setMap(mapc.labMap)
        rob.printMap()

    for i,v in enumerate(IRangles):
        IRangles[i] = radians(v)
    rob.run(IRangles)
