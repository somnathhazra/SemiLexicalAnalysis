import cv2
from darkflow.net.build import TFNet
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageFilter

# Runs the trianed YOLOv2 network on the images in actual folder
# Generates annotated xml files in 'modified' folder
# Annotated files contain the bounding box dimensions
def run_yolo():
	# Setting the objectness threshold at 40%
	options = {
				'model': '../cfg/tiny-yolov2-trial6-2c.cfg',
				'load': 1200,
				'threshold': 0.40,
			}
	tfnet = TFNet(options)
	
	for i, file1 in enumerate(os.listdir("../testimg/actual")):
		file1c = file1
		file1 = "../testimg/actual/" + file1
		img = cv2.imread(file1, cv2.IMREAD_COLOR)
		try:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		except:
			print(file1)
		
		root = ET.Element("annotation")
		ET.SubElement(root, "folder").text = "testimg/actual"
		ET.SubElement(root, "file").text = file1c
		ET.SubElement(root, "segmented").text = "0"
		doc1 = ET.SubElement(root, "size")
		ET.SubElement(doc1, "width").text = str(img.shape[1])
		ET.SubElement(doc1, "height").text = str(img.shape[0])
		ET.SubElement(doc1, "depth").text = str(img.shape[2])
		
		result = tfnet.return_predict(img)
		for j in range(len(result)):
			# If the confidence is above 40% then generate the annotations
			if result[j]['confidence'] >= 0.4:
				tl = (result[j]['topleft']['x'], result[j]['topleft']['y'])
				br = (result[j]['bottomright']['x'], result[j]['bottomright']['y'])
				label = result[j]['label']
				img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
				img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
				
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
		tree.write('../testimg/modified/' + file1c.split(".")[0] + '.xml')
		cv2.imwrite('../testimg/modified/' + file1c, img)


# Blurs the previously identified parts to supress redetected
def blurringfn(file1, blur):
	file1c = file1
	file1 = "../testimg/actual/" + file1
	image = Image.open(file1)
	for i in blur:
		crop_image = image.crop((i[0],i[1],i[2],i[3]))
		blur_image = crop_image.filter(ImageFilter.GaussianBlur(radius = 20))
		image.paste(blur_image, (i[0],i[1],i[2],i[3]))
	image.save("../testimg/modified/blur-" + file1c)


# Runs the YOLOv2 network with reduced threshold
def run_yolo_custom(file1):
	# Runs YOLOv2 with higher threshold to get the blurring dimensions
	options = {
				'model': '../cfg/tiny-yolov2-trial6-2c.cfg',
				'load': 1200,
				'threshold': 0.40,
			}
	tfnet = TFNet(options)
	
	file1c = file1
	file1 = "../testimg/actual/" + file1
	img = cv2.imread(file1, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	root = ET.Element("annotation")
	ET.SubElement(root, "folder").text = "testimg/actual"
	ET.SubElement(root, "file").text = file1c
	ET.SubElement(root, "segmented").text = "0"
	doc1 = ET.SubElement(root, "size")
	ET.SubElement(doc1, "width").text = str(img.shape[1])
	ET.SubElement(doc1, "height").text = str(img.shape[0])
	ET.SubElement(doc1, "depth").text = str(img.shape[2])
	
	blur = []
	result = tfnet.return_predict(img)
	for j in range(len(result)):
		if (result[j]['confidence'] >= 0.4):
			tl = (result[j]['topleft']['x'], result[j]['topleft']['y'])
			br = (result[j]['bottomright']['x'], result[j]['bottomright']['y'])
			label = result[j]['label']
			img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
			img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
			
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
			
			if result[j]['label'] == "wheel":
				blur.append([int(result[j]['topleft']['x']), int(result[j]['topleft']['y']), int(result[j]['bottomright']['x']), int(result[j]['bottomright']['y'])])
			
	blurringfn(file1c, blur)
	file1 = "../testimg/modified/blur-" + file1c
	imgb = cv2.imread(file1, cv2.IMREAD_COLOR)
	imgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Setting the objectness threshold at 20%
	options = {
				'model': '../cfg/tiny-yolov2-trial6-2c.cfg',
				'load': 1200,
				'threshold': 0.20,
			}
	tfnet = TFNet(options)
	
	result = tfnet.return_predict(imgb)
	for j in range(len(result)):
		if (result[j]['confidence'] >= 0.4):
			tl = (result[j]['topleft']['x'], result[j]['topleft']['y'])
			br = (result[j]['bottomright']['x'], result[j]['bottomright']['y'])
			label = result[j]['label']
			img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
			img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
			
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
	tree.write('../testimg/modified/' + file1c.split(".")[0] + '.xml')
	cv2.imwrite('../testimg/modified/' + file1c, img)
	os.remove("../testimg/modified/blur-" + file1c)
