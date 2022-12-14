from lxml import etree

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
    array = parseXML("../Labs/2223-pf/C1-lab.xml")
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


if __name__ == "__main__":

    areas = getAreas()
    print(areas[0])

