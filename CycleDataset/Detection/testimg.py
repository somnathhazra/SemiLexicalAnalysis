import cv2
from darkflow.net.build import TFNet
import os
import shutil
import xml.etree.ElementTree as ET

# Runs the trianed YOLOv2 network on the images in actual folder
# Generates annotated xml files in 'modified' folder
# Annotated files contain the bounding box dimensions
def run_yolo():
	# Setting the objectness threshold at 30%
    options = {
        'model': '../cfg/tiny-yolo-voc-2c.cfg',
        'load': 1200,
        'threshold': 0.30,
    }
    tfnet = TFNet(options)
    
    # Stores number of images where bicycle is identified and their list
    ctr = 0
    correct = []
	
    for i, file1 in enumerate(os.listdir("../testimg/actual")):
        file1c = file1
        file1 = "../testimg/actual/" + file1
        img = cv2.imread(file1, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        str1 = 'img{:04}.jpg'.format(i + 1)

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "testimg/actual"
        ET.SubElement(root, "file").text = str1
        ET.SubElement(root, "segmented").text = "0"
        doc1 = ET.SubElement(root, "size")
        ET.SubElement(doc1, "width").text = str(img.shape[1])
        ET.SubElement(doc1, "height").text = str(img.shape[0])
        ET.SubElement(doc1, "depth").text = str(img.shape[2])

        result = tfnet.return_predict(img)
        
        # If from result list object returned is a bicle with confidence score above 40%
        # -- mark the position of identified object in image
        # -- increment counter to mark correctly identified
        # -- store the annotation
        for j in range(len(result)):
            if result[j]['confidence'] >= 0.4 and result[j]['label']=='bicycle':
                ctr += 1
                correct.append(file1c)
                
                tl = (result[j]['topleft']['x'], result[j]['topleft']['y'])
                br = (result[j]['bottomright']['x'], result[j]['bottomright']['y'])
                label = result[j]['label']
                img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
                img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

                doc2 = ET.SubElement(root, "object")
                ET.SubElement(doc2, "name").text = label
                ET.SubElement(doc2, "pose").text = "Unspecified"
                ET.SubElement(doc2, "truncated").text = "0"
                ET.SubElement(doc1, "difficult").text = "0"
                doc3 = ET.SubElement(doc2, "bndbox")
                ET.SubElement(doc3, "xmin").text = str(result[j]['topleft']['x'])
                ET.SubElement(doc3, "ymin").text = str(result[j]['topleft']['y'])
                ET.SubElement(doc3, "xmax").text = str(result[j]['bottomright']['x'])
                ET.SubElement(doc3, "ymax").text = str(result[j]['bottomright']['y'])

        tree = ET.ElementTree(root)
        tree.write('../testimg/modified/img{:04}.xml'.format(i + 1))
		
		# Store the modified image with bounding box, and the annotation in the folder modified
        str1 = '../testimg/modified/' + str1
        cv2.imwrite(str1, img)
    print(correct)
    return ctr

if __name__ == "__main__":
	
	src = '../testimg/actual'
	dst = '../testimg/modified'
	srf = input('Input directory: ')
	srf = os.path.join('../testimg/', srf)
	try:
		print('Number of files:', len([name for name in os.listdir(srf) if os.path.isfile(os.path.join(srf, name))]))
	except:
		print('No such directory found.')
		exit()
	
	# Create copy of the input folder contents
	# Create a 'modified' folder to store the marked images and the annotations
	if os.path.isdir(src):
		shutil.rmtree(src)
	os.mkdir(src)
	if os.path.isdir(dst):
		shutil.rmtree(dst)
	os.mkdir(dst)
	
	for item in os.listdir(srf):
		s = os.path.join(srf, item)
		d = os.path.join(src, item)
		if os.path.isdir(s):
			shutil.copytree(s, d, symlinks, ignore)
		else:
			shutil.copy2(s, d)
	
	print('Bicycle detcted:',run_yolo())
	
	shutil.rmtree(src)
