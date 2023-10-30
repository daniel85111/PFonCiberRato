from math import *
from lxml import etree
import numpy as np
import cv2
from matplotlib import pyplot as plt
import datetime
import os

# Obtenha a data atual
data_atual = datetime.date.today()

# Formate a data como YYYY-MM-DD
data_formatada = data_atual.strftime('%Y-%m-%d')

# Cria um diretório para as imagens
directory = f'./pflogs/{data_formatada}'
if not os.path.exists(directory):
    os.makedirs(directory)

class ViewParticles():
    def __init__(self, map, grid_directory, particles = None, areas = None, img_scale = 40):
        self.map = map.getMapCropped()

        if areas == None:
            self.areas = map.areas
        else:
            self.areas = areas

        self.particulas = particles

        self.xinit_real, self.yinit_real, self.oriinit_real = self.get_startPos(grid_directory)
       
        self.imscale = img_scale

        self.immax_x = self.imscale * map.mapmax_x
        self.immax_y = self.imscale * map.mapmax_y
        self.img = np.zeros((self.immax_y,self.immax_x,3), np.uint8) 
        self.img = cv2.bitwise_not(self.img)  

    # ---------------------------- Draw each Particle ------------------------
    def drawParticles(self, newParticles, max_weight):
        self.particulas = newParticles
        # Uncomment  other commented lines to draw all particle line sensors


        for i,particula in enumerate(self.particulas):
            # Calculation of positions in the cv window
            x = int(self.imscale*particula.x)
            y = int(self.imscale*particula.y)

            x_sensor_centro = int(self.imscale*particula.sensor_center_posx)
            y_sensor_centro = int(self.imscale*particula.sensor_center_posy)

            # Color based on weight
            if max_weight-1 == 0:
                color = 255
            else:
                color = ((particula.weight-1)/(max_weight-1))*255

            # Drawings
            cv2.circle(self.img, (x,y), 5, (0,color,255-color), -1)                             # Draw particle center (Green)
            cv2.line( self.img, (x,y), (x_sensor_centro, y_sensor_centro), (200,150,100),1)     # Draw line from particle center to center line sensor (Grayish)
    # ---------------------------- End of Draw each Particle -----------------------

    # ---------------------------- Desenhar Centroides ----------------------
    def drawCentroides(self, centroides, oris, weights, covariances):
        if len(centroides) < 1: return
        if covariances is None: return
        if len(covariances) > 15: return
        max_weight = max(weights)
        for i,centroide in enumerate(centroides):
            

            x = int(self.imscale*centroide[0])
            y = int(self.imscale*centroide[1])
            if x == None or y == None: return

            radious = int(sqrt(2000*abs(covariances[i][0,1])))

            if weights[i] == max_weight:
                cv2.circle(self.img,(int(x),int(y)), radious, (0,250,255), -1) # Circulo centrado no centro do robot real
            else:
                cv2.circle(self.img,(int(x),int(y)), radious, (200,0,0), -1) # Circulo centrado no centro do robot real
            cv2.line( self.img, (int(x),int(y)), (int(x+radious*cos(oris[i])), int(y-(radious*sin(oris[i])))), (0,0,255),2) # Linha do centro do robot direcionada segundo orientaçao
    # ----------------------------- End of Desenhar Centroides ------------------------------------

    # ------------------------------- Desenhar o robo real ----------------------------------------
    def drawReal(self,x,y,ori, diameter, DISTsens, IRangles, num_endpoints):
        # Constants and calculation of positions in the cv window
        y_correction = int(14)*self.imscale     # In OpenCV the (0,0) is top-left but from the simulator is bottom-left
        radious = 0.5*diameter                  # Robot real radious
        
        if x == None or y == None or ori == None : return
        if DISTsens == None : DISTsens = (0,0,0,0)

        sensorDIST_apparture = radians(30)
        sensorDIST_center_ori = IRangles[0]
        sensorDIST_left_ori  = IRangles[1]
        sensorDIST_right_ori = IRangles[2]
        sensorDIST_back_ori = IRangles[3]

        # Center of robot
        cx = self.imscale*(x+self.xinit_real[0])                     
        cy = abs(y_correction - self.imscale*(y+self.yinit_real[0]))
        
        # Distance Sensors location
        endpoints_angle = (2*sensorDIST_apparture)/(num_endpoints-1)

        sensorDIST = np.array([[angle, cx + self.imscale*0.5*cos(ori+angle), cy - self.imscale*0.5*sin(ori+angle)] for i, angle in enumerate(IRangles)])
        for j,v in enumerate(IRangles):
            leitura = DISTsens[j]
            for k in range(num_endpoints):
                angle = endpoints_angle*k
                posx = sensorDIST[j][1] + self.imscale*leitura*cos(ori + sensorDIST[j][0] - sensorDIST_apparture + angle)
                posy = sensorDIST[j][2] - self.imscale*leitura*sin(ori +  sensorDIST[j][0] - sensorDIST_apparture + angle)
                
                if j%3==0:
                    cv2.circle(self.img, (int(posx),int(posy)), 2, (0,255,255), -1)
                else:
                    cv2.circle(self.img, (int(posx),int(posy)), 2, (0,0,255), -1)



        sensorDIST_center_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_center_ori)
        sensorDIST_center_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_center_ori)

        sensorDIST_left_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_left_ori)
        sensorDIST_left_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_left_ori)

        sensorDIST_right_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_right_ori)
        sensorDIST_right_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_right_ori)

        sensorDIST_back_posx = cx + self.imscale*0.5*cos(ori+sensorDIST_back_ori)
        sensorDIST_back_posy = cy - self.imscale*0.5*sin(ori+sensorDIST_back_ori)

        
        

        # Draws of robot elements
        cv2.circle(self.img,(int(cx),int(cy)), 20, (0,0,0), 2) # Circulo centrado no centro do robot real
        cv2.line( self.img, (int(cx),int(cy)), (int(cx+radious*self.imscale*cos(ori)), int(cy-(radious*self.imscale*sin(ori)))), (0,0,0),2) # Linha do centro do robot direcionada segundo orientaçao

        
        cv2.circle(self.img, (int(sensorDIST_left_posx),int(sensorDIST_left_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_center_posx),int(sensorDIST_center_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_right_posx),int(sensorDIST_right_posy)), 2, (0,0,253), -1)
        cv2.circle(self.img, (int(sensorDIST_back_posx),int(sensorDIST_back_posy)), 2, (0,0,253), -1)
        
        #print(f'\nGPS: x: {40*x+40*cos(ori)}   y: {40*y+40*sin(ori)}   theta: {ori}')
    # --------------------- End of Desenhar o robo real ----------------------------------
    
    
    # ----------------- Desenhar mapa -------------------
    def drawMap(self, map):
        for j,area_vertex in enumerate(self.areas):
            cv2.rectangle(self.img,(int(self.imscale*area_vertex[0][0]),int(self.imscale*area_vertex[0][1])),(int(self.imscale*area_vertex[1][0]),int(self.imscale*area_vertex[1][1])),(255,0,0),-1)
        

    def showImg(self):
        # cv2.imshow("img",self.img)
        img_not = cv2.bitwise_not(self.img)
        cv2.imshow("img",self.img)
        # cv2.imshow("img",img_not)


        cv2.waitKey(1)

    def clearImg(self):
        self.img = np.zeros((560,1120,3), np.uint8)
        self.img = cv2.bitwise_not(self.img)
    # ------------- End of Desenhar mapa ----------------

    # ------------- Image Log ---------------------
    def saveImg(self):
        # Obtenha a data e hora atual
        data_hora_atual = datetime.datetime.now()
        # Formate a data e hora como YYYY-MM-DD_HH-MM-SS
        data_hora_formatada = data_hora_atual.strftime('%Y-%m-%d_%H-%M-%S')
        # Salva a imagem em um arquivo
        filename = os.path.join(directory, 'imagem_{}.jpg'.format(data_hora_formatada))
        # img_not = cv2.bitwise_not(self.img)
        cv2.imwrite(filename, self.img)

    # ------------ End of Image Log ---------------

    # ----------------- Posicao real original do robo no mapa -------------------
    def get_startPos(self,xmlFile):
        """
        Parse the XML
        """
        dic = []
        
        with open(xmlFile) as fobj:
            xml = fobj.read()
        root = etree.fromstring(xml)
        
        for grid in root.getchildren():
            if(grid.tag=="Position"):
                dic.append(grid.attrib)
        
        x = []
        y = []
        ori = []

        for i,v in enumerate(dic):
            x.append(float(v["X"]))
            y.append(float(v["Y"]))
            ori.append(float(v["Dir"]))

        return (x,y,ori)
    # ----------------- End of Posicao real original do robo no mapa ---------------
    

    # -------- Final pose is not used (early test) -------------
    def drawFinalPose(self,final_pose):
        # Draws of robot elements
        x,y,ori = final_pose
        x = x*self.imscale
        y = y*self.imscale
        radious = 0.25
        cv2.circle(self.img,(int(x),int(y)), 10, (200,0,0), -1) # Circulo centrado no centro do robot real
        cv2.line( self.img, (int(x),int(y)), (int(x+radious*self.imscale*cos(ori)), int(y-(radious*self.imscale*sin(ori)))), (0,0,255),2) # Linha do centro do robot direcionada segundo orientaçao
    # ----- End of Final pose is not used (early test) -----------
    

