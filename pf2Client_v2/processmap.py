import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import *
from lxml import etree
class map():
    def __init__(self, lab_directory = None, scale = 100, mapmax_x = 28, mapmax_y = 14):
        self.directory = lab_directory
        self.scale = scale
        self.mapmax_x = mapmax_x
        self.mapmax_y = mapmax_y

        self.areas = self.calculateAreas()
        self.map_full, self.map_cropped, self.map_validation, self.map_validation_cropped = self.calculateMap() #Validation is solid. Used in resample
        self.distance_map_cropped, self.distance_map_full = self.calculateDistanceMap()

        np.savetxt('Dist MAP', self.distance_map_cropped, fmt='%2.2f', delimiter=', ')
    
    #----------------------------------------------------------------    
    def getAreas(self):
        return self.areas

    def getScale(self):
        return self.scale

    def getDirectory(self):
        return self.directory
    
    def getMap(self):
        return self.map_full
    
    def getMapCropped(self):
        return self.map_cropped
    
    def getDistanceMap(self):
        return self.distance_map_full
    
    def getDistanceMapCropped(self):
        return self.distance_map_cropped
    
    # ---------------------------------------------------------------
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
        #print(dic)
        return dic

    def calculateAreas(self, lab_directory=None):
        if self.directory != None:
            array = self.parseXML(self.directory)
        elif lab_directory != None:
            array = self.parseXML(lab_directory)
        else:
            return

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
        #print(areas[1][1])
        return areas    # ([minx,miny], [maxx,maxy])

    def calculateMap(self):
        mapstartx = self.scale*self.mapmax_x
        mapstarty = self.scale*self.mapmax_y
        mapendx = 2*self.scale*self.mapmax_x
        mapendy = 2*self.scale*self.mapmax_y

        topleft = (mapstartx,mapstarty)
        topright = (mapendx,mapstarty)
        bottomright = (mapendx, mapendy)
        bottomleft = (mapstartx, mapendy)

        arr = np.zeros((3*mapstarty,3*mapstartx,1), np.uint8)
        arr_solid = np.zeros((3*mapstarty,3*mapstartx,1), np.uint8)


        cv2.line(arr, (topleft), (topright),255,1)
        cv2.line(arr, (topright), (bottomright),255,1)
        cv2.line(arr, (bottomright), (bottomleft),255,1)       
        cv2.line(arr, (bottomleft), (topleft),255,1)

        cv2.rectangle(arr_solid,(0,0),(topleft),255,-1)
        cv2.rectangle(arr_solid,(mapstartx,0),(topright),255,-1)
        cv2.rectangle(arr_solid,(mapendx,0),(3*mapstartx,mapstarty),255,-1)
        cv2.rectangle(arr_solid,(0,mapstarty),(bottomleft),255,-1)
        cv2.rectangle(arr_solid,(topright),(3*mapstartx,mapendy),255,-1)
        cv2.rectangle(arr_solid,(0,mapendy),(mapstartx,3*mapstarty),255,-1)
        cv2.rectangle(arr_solid,(bottomleft),(mapendx,3*mapstarty),255,-1)
        cv2.rectangle(arr_solid,(bottomright),(3*mapstartx,3*mapstarty),255,-1)

        
        for i,v in enumerate(self.areas): # v[0][0] = xmin v[0][1] = ymin
            # print(f'i = {i}\t v= {v}')
            xmin = int(self.scale*v[0][0])
            ymin = int(self.scale*v[0][1])
            xmax = int(self.scale*v[1][0])
            ymax = int(self.scale*v[1][1])
            # print(f' xmin= {xmin}\t ymin= {ymin}\t xmax= {xmax}\t ymax= {ymax}\t')
            area_topleft     =  (mapstartx+xmin, mapstarty+ymin)
            area_topright    =  (mapstartx+xmax, mapstarty+ymin)
            area_bottomright =  (mapstartx+xmax, mapstarty+ymax)
            area_bottomleft  =  (mapstartx+xmin, mapstarty+ymax)

            cv2.line(arr, (area_topleft), (area_topright), 255,1)
            cv2.line(arr, (area_topright), (area_bottomright), 255,1)
            cv2.line(arr, (area_bottomright), (area_bottomleft), 255,1)
            cv2.line(arr, (area_bottomleft), (area_topleft), 255,1)

            cv2.rectangle(arr_solid,(area_topleft),(area_bottomright),255,-1)

        # cv2.imshow("Resized image", arr_solid)
        # cv2.waitKey(0)  
        # plt.imshow(arr_solid)
        # plt.show() 
       
        cropped = arr[mapstarty:mapendy+1, mapstartx:mapendx+1]
        cropped_solid = arr_solid[mapstarty:mapendy+1, mapstartx:mapendx+1]

        # cv2.imshow("Resized image", cropped)
        # cv2.waitKey(0)
        # plt.imshow(arr_solid)
        # plt.show() 

        return arr,cropped,arr_solid,cropped_solid
    
    def isValidLocation(self,x,y):
        x = ceil(x)
        y = ceil(y)
        if self.map_validation[y][x] == 255:
            return False
        else: 
            return True


    def calculateDistanceMap(self):
        cropped = self.map_cropped
        cropped = cv2.bitwise_not(cropped)
        cropped = cropped.astype(np.uint8)

        full = self.map_full
        full = cv2.bitwise_not(full)
        full = full.astype(np.uint8)
        

        distmap_cropped = cv2.distanceTransform(cropped, cv2.DIST_L2, 5)/self.scale
        distmap_full = cv2.distanceTransform(full, cv2.DIST_L2, 5)/self.scale

        # plt.imshow(distmap_cropped)
        # plt.show()
        # plt.imshow(distmap_full)
        # plt.show() 
        return distmap_cropped, distmap_full
