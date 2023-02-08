class map():
    def __init__(self, directory = "../Labs/2223-pf/C2-lab.xml", scale = 10):
        self.directory = directory
        self.scalee = scale
        self.mapmax_x = mapmax_x
        self.mapmax_y = mapmax_y

    def getScale(self):
        return self.scalee

    def getDirectory(self):
        return self.directory

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

    def getAreas(self):
        array = self.parseXML(self.directory)
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

    def getMap(self):
        scale = 100

        mapstartx = scale*self.mapmax_x
        mapstarty = scale*self.mapmax_y
        mapendx = 2*scale*self.mapmax_x
        mapendy = 2*scale*self.mapmax_y

        topleft = (mapstartx,mapstarty)
        topright = (mapendx,mapstarty)
        bottomright = (mapendx, mapendy)
        bottomleft = (mapstartx, mapendy)

        arr = np.zeros((3*mapstarty,3*mapstartx,1), np.uint8)

        cv2.line(arr, (topleft), (topright),255,1)
        cv2.line(arr, (topright), (bottomright),255,1)
        cv2.line(arr, (bottomright), (bottomleft),255,1)       
        cv2.line(arr, (bottomleft), (topleft),255,1)
        
        for i,v in enumerate(self.areas): # v[0][0] = xmin v[0][1] = ymin
            # print(f'i = {i}\t v= {v}')
            xmin = int(scale*v[0][0])
            ymin = int(scale*v[0][1])
            xmax = int(scale*v[1][0])
            ymax = int(scale*v[1][1])
            # print(f' xmin= {xmin}\t ymin= {ymin}\t xmax= {xmax}\t ymax= {ymax}\t')
            area_topleft     =  (mapstartx+xmin, mapstarty+ymin)
            area_topright    =  (mapstartx+xmax, mapstarty+ymin)
            area_bottomright =  (mapstartx+xmax, mapstarty+ymax)
            area_bottomleft  =  (mapstartx+xmin, mapstarty+ymax)

            cv2.line(arr, (area_topleft), (area_topright), 255,1)
            cv2.line(arr, (area_topright), (area_bottomright), 255,1)
            cv2.line(arr, (area_bottomright), (area_bottomleft), 255,1)
            cv2.line(arr, (area_bottomleft), (area_topleft), 255,1)

        # cv2.imshow("Resized image", arr)
        # cv2.waitKey(0)  
        # plt.imshow(arr)
        # plt.show() 
       
        cropped = arr[mapstarty:mapendy+1, mapstartx:mapendx+1]
        # cv2.imshow("Resized image", cropped)
        # cv2.waitKey(0) ^
        # plt.imshow(cropped)
        # plt.show() 
        return arr,cropped,scale

    def getDistanceMap(self):
        cropped = self.map_cropped
        cropped = cv2.bitwise_not(cropped)
        cropped = cropped.astype(np.uint8)

        full = self.map
        full = cv2.bitwise_not(full)
        full = full.astype(np.uint8)
        

        distmap_cropped = cv2.distanceTransform(cropped, cv2.DIST_L2, 5)
        distmap_full = cv2.distanceTransform(full, cv2.DIST_L2, 5)

        # plt.imshow(distmap_cropped)
        # plt.show()
        # plt.imshow(distmap_full)
        # plt.show() 
        return distmap_cropped
