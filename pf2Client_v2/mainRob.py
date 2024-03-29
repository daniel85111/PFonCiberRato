
import sys
import logging
import datetime
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

# Simulador info
CELLROWS=7
CELLCOLS=14

# 1 = Ativo; Outro = Inativo
log = 1
viewer = 0


# Logs
if log == 1:
    data_hora_atual = datetime.datetime.now()
    data_hora_formatada = data_hora_atual.strftime('%Y-%m-%d_%H-%M-%S')
    nome_arquivo = f"./pflogs/log_{data_hora_formatada}.txt"
    logging.basicConfig(filename=nome_arquivo, level=logging.DEBUG,format='%(message)s')




class MyRob(CRobLinkAngs):
    def __init__(self, rob_name, rob_id, IRangles, host):
        CRobLinkAngs.__init__(self, rob_name, rob_id, IRangles, host)
        # Constants
        self.rob_name = rob_name
        self.robot_diameter = 1
        self.mapmax_x = 28
        self.mapmax_y = 14

        # Movement(sensors, actuators)
        self.movement_counter = 0
        self.movement_counter_trigger = 0.35
        self.resample_flag = 0
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

        # FILTRO de PARTICULAS
        self.n_part = 50  # Numero de particulas desejado no filtro 
        self.weight_calculation_method = 4        # Weight calculation method (3 = metodo1; 4 = metodo2)

        # ENDPOINTS number
        self.num_endpoints = 5

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
        self.filter_runmode = 1
        self.key_up = False
        self.key_down = False
        self.key_left = False
        self.key_right = False
        self.filter_startover = False
        self.cluster = 0



        # Mapa
        self.mapa = processmap.map(lab_directory="../Labs/2223-pf/C2-lab.xml")

        # Viwer
        self.visualizer = viewercv.ViewParticles(self.mapa, grid_directory = "../Labs/2223-pf/C4-grid.xml")
        self.visualizer.drawMap(self.mapa)

        # Particle filter
        self.filtro_particulas = particleFilter.filtroParticulas(self.mapa, IRangles, self.num_endpoints, self.n_part)


        if viewer == 1:
            self.visualizer.drawParticles(self.filtro_particulas.particulas, self.filtro_particulas.max_w)
            self.visualizer.drawReal(self.x_od_pos, self.y_od_pos, self.ori, self.robot_diameter,  self.DISTsens, IRangles, self.num_endpoints)
            self.visualizer.showImg()
            self.visualizer.saveImg()


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

    # -------------------------------------- Run Cycle -------------------------------------
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

            # Primeiro movimento, offset de gps
            if self.firstRun:                               
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
            self.check_keyboard()
            self.update_particle_filter(IRangles, state,  weight_calculation_method = self.weight_calculation_method)    # Atualizar filtro de particulas
    # ---------------------------------------------- End of Run Cycle ----------------------------------------------------------
    
    # ---------------------------------------------- Keyboard ------------------------------------------------------------------        
    # Verificar todos os ciclos se a ultima tecla é responsavel por alterar variaveis de controlo
    # Espaço - Condução (Automática/manual)
    def check_keyboard(self):

        if self.keyboard_variable == "Key.space":
            # logging.info(f'Key.space pressed: self.autorun changed\n')

            if self.autorun:
                self.autorun = False
            else:
                self.autorun = True
            self.keyboard_variable = ""
        
        if self.keyboard_variable == "r":
            # logging.info(f'r pressed: filter_startover\n')
            self.keyboard_variable = ""  
            self.filter_startover = True

        if self.keyboard_variable == "f":
            # logging.info(f'f pressed: filter_runmode changed\n')
            self.keyboard_variable = ""     
            if self.filter_runmode == 0:
                self.filter_runmode = 1
            else:
                self.filter_runmode = 0
        
        if not self.autorun:
            if self.keyboard_variable == "Key.up":
                self.key_up = True
                self.keyboard_variable = ""

            if self.keyboard_variable == "Key.down":
                self.keyboard_variable = ""
                self.key_down = True
            
            if self.keyboard_variable == "Key.left":
                self.keyboard_variable = ""
                self.key_left = True
                
            if self.keyboard_variable == "Key.right":
                self.keyboard_variable = ""
                self.key_right = True
    # --------------------------------------------------End of Keyboard ---------------------------------------

    # ------------------------------------------------- Particle Filter ---------------------------------------
    # Função responsável por atualizar ao filtro de particulas todos os ciclos (Restart, Move, Weight_calc, Cluster, Draw, Resample, LOG)
    def update_particle_filter(self, IRangles, state, weight_calculation_method = 2):
        time_move_particles = 0
        time_weight_calculation = 0
        time_weight_normalization = 0
        time_resample = 0
        time_drawing = 0
        time_clustering = 0

        # Restart
        if self.filter_startover:
            self.filtro_particulas.createNewParticleSet()
            if viewer == 1:
                self.visualizer.clearImg()
                self.visualizer.drawMap(self.mapa)
                self.visualizer.drawParticles(self.filtro_particulas.particulas, self.filtro_particulas.max_w)
                self.visualizer.drawReal(self.x_od_pos, self.y_od_pos, self.ori, self.robot_diameter,  self.DISTsens, IRangles, self.num_endpoints)
                self.visualizer.showImg()
            self.filter_startover = False

        if(state != "stop"):
            start_overall = time.perf_counter()
            if viewer == 1:
                self.visualizer.clearImg()
            # Mover particulas
            start = time.perf_counter()
            self.filtro_particulas.odometry_move_particles(self.motors, self.motorsNoise, self.measures.collision)    # Mover particulas 
            end = time.perf_counter()
            time_move_particles = 1000*(end-start)
            
            # Efetuamos o resample e atualizamos o peso das particulas apenas caso runmode = 1
            start = time.perf_counter()
            if self.filter_runmode == 1:
                self.filtro_particulas.weights_calculation(self.LINEsens, self.DISTsens, weight_calculation_method)  # Calcular pesos de cada particula
            end = time.perf_counter()
            time_weight_calculation = 1000*(end-start)


            start = time.perf_counter()
            if self.filter_runmode == 1:
                self.filtro_particulas.weights_normalization()            # Normalizar peso de cada particula            
            end = time.perf_counter()
            time_weight_normalization = 1000*(end-start)

            # Cluster
            start = time.perf_counter()
            self.filtro_particulas.cluster()
            end = time.perf_counter()
            time_clustering = 1000*(end-start)

            # Draw
            start = time.perf_counter()
            if viewer == 1:
                self.visualizer.drawMap(self.mapa)
                self.visualizer.drawParticles(self.filtro_particulas.particulas, self.filtro_particulas.max_w)
                self.visualizer.drawReal(self.posx, self.posy, self.ori, self.robot_diameter, self.DISTsens, IRangles, self.num_endpoints)
            end = time.perf_counter()
            time_drawing = 1000*(end-start)

            # final_pose = self.filtro_particulas.getFinalPose()
            # self.visualizer.drawFinalPose(final_pose)
            if viewer == 1:
                if self.filtro_particulas.centroides is not None:
                    self.visualizer.drawCentroides(self.filtro_particulas.centroides, self.filtro_particulas.centroides_oris, self.filtro_particulas.centroides_weights, self.filtro_particulas.centroides_cov)
                self.visualizer.showImg()

            # Resample
            if (self.movement_counter > self.movement_counter_trigger):
                self.movement_counter = 0
                start = time.perf_counter()
                if self.filter_runmode == 1:
                    self.filtro_particulas.sistematic_resample()      # Resample sistematico de particulas
                    # self.filtro_particulas.resample()      # Resample aleatorio de particulas

                end = time.perf_counter()
                time_resample = 1000*(end-start)
            

            # Fazer isto dentro do particleFilter ??? (antes de resample?? -> Indiferente!)
            erro = 'Nao calculado'
            x_mp = 'Nao calculado'
            y_mp = 'Nao calculado'
            ori_mp = 'Nao calculado'
            if(self.filtro_particulas.centroides_weights) is not None:
                centroides_count = len(self.filtro_particulas.centroides_weights)
                weight_centroide_mais_provavel = max(self.filtro_particulas.centroides_weights)
                idx_centroide_mais_provavel = self.filtro_particulas.centroides_weights.index(weight_centroide_mais_provavel)

                x_mp = self.filtro_particulas.centroides[idx_centroide_mais_provavel][0]
                y_mp = self.filtro_particulas.centroides[idx_centroide_mais_provavel][1]
                ori_mp = self.filtro_particulas.centroides_oris[idx_centroide_mais_provavel]
                
                if centroides_count > 1:
                    lista_sem_max = []
                    for num in self.filtro_particulas.centroides_weights:
                        if num != weight_centroide_mais_provavel:
                            lista_sem_max.append(num)
                        else:
                            lista_sem_max.append(0)
                    # lista_sem_max = [num for num in self.filtro_particulas.centroides_weights if num != weight_centroide_mais_provavel]
                    weight_centroide_mais_provavel2 = max(lista_sem_max)
                    idx_centroide_mais_provavel2 = lista_sem_max.index(weight_centroide_mais_provavel2)

                    x_mp2 = self.filtro_particulas.centroides[idx_centroide_mais_provavel2][0]
                    y_mp2 = self.filtro_particulas.centroides[idx_centroide_mais_provavel2][1]
                    ori_mp2 = self.filtro_particulas.centroides_oris[idx_centroide_mais_provavel2]
                else:
                    x_mp2 = self.filtro_particulas.centroides[idx_centroide_mais_provavel][0]
                    y_mp2 = self.filtro_particulas.centroides[idx_centroide_mais_provavel][1]
                    ori_mp2 = self.filtro_particulas.centroides_oris[idx_centroide_mais_provavel]

                # erro = (x_mp-self.posx-4.5, 14-y_mp-self.posy-11.5, ori_mp-self.ori)

            # ---------> LOG 
            total = time_move_particles + time_weight_calculation + time_weight_normalization + time_resample + time_clustering + time_drawing 

            # print(f'tempo total = {total:.0f}ms\n\t\t\t(Resample: {(time_resample):.1f} | W_update: {(time_weight_calculation):.1f} | W_norm: {(time_weight_normalization):.1f} | Od_Move: {(time_move_particles):.1f} | CL: {(time_clustering):.1f})')
            # print(f'Erro = x:{erro[0]:.2}, y:{erro[1]:.2}, ori:{erro[2]:.2}\n')
            if viewer == 1:
                self.visualizer.saveImg()
            # logging.info(f'tempo de calculos = {total:.0f}ms (Resample: {(time_resample):.1f} | W_update: {(time_weight_calculation):.1f} | W_norm: {(time_weight_normalization):.1f} | Od_Move: {(time_move_particles):.1f} | CL: {(time_clustering):.1f})')
            # logging.info(f'{total:.0f}\t')
            
            # logging.info(f'Centroide mais provavel (x,y,ori)->{x_mp:.2f},{y_mp:.2f},{ori_mp:.0f}')
            # logging.info(f'{x_mp:.2f} {y_mp:.2f} {ori_mp:.0f}\t')

            # logging.info(f'Pose real (x,y,ori)->{self.posx-4.5:.2f},{self.posy-11.5:.2f},{self.ori:.0f}')
            # logging.info(f'{self.posx-4.5:.2f} {self.posy-11.5:.2f} {self.ori:.0f}\t')


            # logging.info(f'Error from real->{erro[0]},{erro[1]},{erro[2]}')
            # logging.info(f'Centroids weights->{self.filtro_particulas.centroides_weights}')
            # logging.info(f'Centroids Covariances->{self.filtro_particulas.centroides_cov}')
            # logging.info(f'Effective number of particles->{self.filtro_particulas.effective_num_particles:.4f}')
            # logging.info(f'{self.filtro_particulas.effective_num_particles:.4f}\t')

            if log == 1:
                logging.info(f'{self.measures.time} \t{total:.4f}\t{centroides_count}\t{x_mp:.4f} {y_mp:.4f} {ori_mp:.4f}\t{x_mp2:.4f} {y_mp2:.4f} {ori_mp2:.4f}\t{self.posx+4.5:.4f} {14-self.posy-11.5:.4f} {self.ori:.4f}\t{self.filtro_particulas.effective_num_particles:.4f}\n')
            
            end_overall = time.perf_counter()
            time_overall = 1000*(end_overall-start_overall)
            # print(f'tempo total = {time_overall:.1f}')
            # ---------> LOG END
    # -------------------------------------------- End of Particle Filter ---------------------------------------

    # -------------------------------------------- Movement and Sensor Data -------------------------------------
    def wander(self):
        center_id = 0
        left_id = 1
        right_id = 2
        back_id = 3
        temporrestante = int(self.simTime)-self.measures.time
        
        if temporrestante <= 0:
            exit()
        #Se estiver em modo automatico
        if self.autorun:
            if self.last_runmode == 0:
                self.last_runmode = 1

            # Se nao estiver numa colisão nem num movimento de recuperação de colisão:    
            #   Executa codigo de controlo automatico do robot
            if not self.measures.collision and self.backwards_counter < 0:

                # Se estiver na proximidade de uma parede
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

                # Se não estiver na proximidade de uma parede
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
                        self.driveMotors(lpow,rpow)             # Andar com velocidade constante (L = 0.1, R = 0.1)
            
            # Movimento de recuperação de colisão
            else:

                if self.backwards_counter < 0:
                # Utiliza 3 ciclos de simução na recuperação de colisão
                    self.backwards_counter = 5
                lpow = -0.1
                rpow = -0.15
                self.motors = (lpow, rpow) 
                self.driveMotors(lpow,rpow)                     # Andar com velocidade constante (L = 0.1, R = 0.1)
                self.backwards_counter -= 1
    
        #Se estiver em modo manual
        else:
            lpow, rpow = self.motors
            manual_speed_per_click = 0.01
            manual_speed_treshold = 0.13

            if self.last_runmode == 1:
                self.last_runmode = 0
                #Parar
                lpow = 0.0
                rpow = 0.0

            # Controlar manualmente o robot com as setas do teclado
            if self.key_up:
                if self.motors[0] < manual_speed_treshold:
                    lpow += manual_speed_per_click
                if self.motors[1] < manual_speed_treshold:
                    rpow += manual_speed_per_click
                self.key_up = not self.key_up
            
            if self.key_down:
                if self.motors[0] > -manual_speed_treshold:
                    lpow -= manual_speed_per_click
                if self.motors[1] > -manual_speed_treshold:
                    rpow -= manual_speed_per_click
                self.key_down = not self.key_down
            
            if self.key_left:
                if self.motors[0] > -manual_speed_treshold:
                    lpow -= manual_speed_per_click
                if self.motors[1] < manual_speed_treshold:
                    rpow += manual_speed_per_click
                self.key_left = not self.key_left

            if self.key_right:
                if self.motors[0] < manual_speed_treshold:
                    lpow += manual_speed_per_click
                if self.motors[1] > -manual_speed_treshold:
                    rpow -= manual_speed_per_click
                self.key_right = not self.key_right

            self.motors = (lpow, rpow)
            self.driveMotors(lpow,rpow) 
        
        self.LINEsens = list(map(int, self.measures.lineSensor))         # Linha de Sensores

        # Sensor de distacia
        IRsens = self.measures.irSensor
        
        distancias = []
        for i,v in enumerate(IRsens):
            if v == 0: 
                distancias.append(None)
                return
            
            distancias.append(1/v)
            # distancias.append(10)

        self.DISTsens = distancias
        # print(f'Center: {distancias[center_id]:.2f}\tLeft: {distancias[left_id]:.2f}\tRight: {distancias[right_id]:.2f}\tBack: {distancias[back_id]:.2f}')
    # -------------------------------------------- End of Movement and Sensor Data -------------------------------------

    # -------------------------------------------- Odometria do Robo ---------------------------------------------------
    def odometry_move_robot(self, motors):
        
        self.motors = motors

        self.movement_counter += abs(self.motors[0])+abs(self.motors[1])       
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
    # -------------------------------------------- End of Odometria do Robo ---------------------------------------------------
    

            

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

# ----------------------------- Main -------------------------
if __name__ == '__main__':
    IRangles = [0.0,75.0,-75.0,180.0] # Ajust IR angles here!!! Order:(Center,Left,Right,Back)
    rob=MyRob(rob_name,pos,IRangles,host)
    if mapc != None:
        rob.setMap(mapc.labMap)
        rob.printMap()

    for i,v in enumerate(IRangles):
        IRangles[i] = radians(v)
    rob.run(IRangles)
