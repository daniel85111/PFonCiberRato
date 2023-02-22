from lxml import etree
import numpy as np
import matplotlib.pyplot as plt
import cv2

def parseXML(xmlFile):
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

def getAreas():
    array = parseXML("../Labs/2223-pf/C2-lab.xml")
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
    return areas

def get_startPos(xmlFile):
    """
    Parse the XML
    """
    dic = []
    x = []
    y = []
    ori = []
    with open(xmlFile) as fobj:
        xml = fobj.read()
    root = etree.fromstring(xml)
    
    for grid in root.getchildren():
        if(grid.tag=="Position"):
            dic.append(grid.attrib)
    
    for i,v in enumerate(dic):
        x.append(float(v["X"]))
        y.append(float(v["Y"]))
        ori.append(float(v["Dir"]))

    return (x,y,ori) 

def getMap(areas, mapmax_x, mapmax_y):
    arr = np.array([],dtype=np.uint8)
    scale = 10

    for l in range(scale*mapmax_y):
        # collum = []
        for c in range(scale*mapmax_x):
            sum = 0
            for j,k in enumerate(areas):

                if c >= scale*float(k[0][0]) and c+1 <= scale*float(k[1][0]) and l >= scale*float(k[0][1]) and l+1 <= scale*float(k[1][1]):
                    sum += 1
                    break

            if sum != 0:
                arr = np.concatenate((arr,[1]))
            else:
                arr = np.concatenate((arr,[0]))

        # line.append(collum)

    return arr,scale


if __name__ == "__main__":

    areas = getAreas()
    x,y,ori = get_startPos("../Labs/2223-pf/C2-grid.xml")
    # print(x)
    # print(y)
    # print(ori)
    # print(len(x))
    map, map_scale_factor = getMap(areas,28,14)
    print(len(map))

    b = map.reshape(10*14, 10*28)
    b = cv2.bitwise_not(b)
    b = b.astype(np.uint8)
    plt.imshow(b)
    plt.show()

    dist = cv2.distanceTransform(b, cv2.DIST_L1, 5)
    plt.imshow(dist)
    plt.show()
    # print(pos)


