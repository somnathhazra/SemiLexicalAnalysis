import xml.etree.ElementTree as ET

# Parses the xml file to look for wheels and frames
# Input:	The xml file to be parsed
# Output:	2 lists of bounding box coordinates containing wheels and frames
def parse_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    list_wheel = []
    list_seat = []
    countW = 0
    countS = 0
    # iterate news items
    for object in root.findall('object'):
        name = object.find('name').text
        if name == "wheel":
            list_wheel.append([])
            list_wheel[countW].append(object.find('bndbox').find('xmin').text)
            list_wheel[countW].append(object.find('bndbox').find('ymin').text)
            list_wheel[countW].append(object.find('bndbox').find('xmax').text)
            list_wheel[countW].append(object.find('bndbox').find('ymax').text)
            countW = countW + 1
        elif name == "frame":
            list_seat.append([])
            list_seat[countS].append(object.find('bndbox').find('xmin').text)
            list_seat[countS].append(object.find('bndbox').find('ymin').text)
            list_seat[countS].append(object.find('bndbox').find('xmax').text)
            list_seat[countS].append(object.find('bndbox').find('ymax').text)
            countS = countS + 1
    return list_wheel, list_seat
