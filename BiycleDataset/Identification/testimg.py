import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom

CONF_THRESH, NMS_THRESH = 0.7, 0.5

# Loads the YOLOv3 network from configuration and weight files
# Output:	Pretrained YOLOv3 model
def load_yolo():
	net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", "yolov3_custom_last.weights")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	classes = []
	with open("obj.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes)*4, 3))
	return net, classes, colors, output_layers

# Loads the image from the given path
# Input:	Image path
# Output:	Image and its dimensions
def load_image(img_path):
	img = cv2.imread(img_path)
	height, width, channels = img.shape
	return img, height, width, channels

# Single pass through the network
# Input:	Image, Network
# Output:	Output of the network
def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

# Calculates bounding boxes, confidence scores, classes
# Input:	Outputs of the YOLOv3 network
# Output:	Boxes, Class ids, Confidence scores
def get_box_dimensions(outputs, height, width, call):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > (CONF_THRESH - float(call) * 0.1):
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

# Creates new image file pointing out the detected objects
# Creates corresponding csv file descibing detections
# Input:	All the detection details and image file path
def draw_labels(boxes, confs, colors, class_ids, classes, img, img_path, call):
	indexes = cv2.dnn.NMSBoxes(boxes, confs, (CONF_THRESH - float(call) * 0.1), NMS_THRESH)
	font = cv2.FONT_HERSHEY_PLAIN
	height, width, channels = cv2.imread(img_path).shape
	root = ET.Element("annotation")	
	ET.SubElement(root, "folder").text = "testimg/modified"
	ET.SubElement(root, "file").text = img_path.split("/")[-1]
	ET.SubElement(root, "segmented").text = "0"
	doc1 = ET.SubElement(root, "size")
	ET.SubElement(doc1, "width").text = str(width)
	ET.SubElement(doc1, "height").text = str(height)
	ET.SubElement(doc1, "depth").text = str(channels)
	
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x + 5, y + 15), font, 1, color, 1)
			
			doc2 = ET.SubElement(root, "object")
			ET.SubElement(doc2, "name").text = label
			ET.SubElement(doc2, "pose").text = "Unspecified"
			ET.SubElement(doc2, "truncated").text = "0"
			ET.SubElement(doc2, "difficult").text = "0"
			doc3 = ET.SubElement(doc2, "bndbox")
			ET.SubElement(doc3, "xmin").text = str(x)
			ET.SubElement(doc3, "ymin").text = str(y)
			ET.SubElement(doc3, "xmax").text = str(x+w)
			ET.SubElement(doc3, "ymax").text = str(y+h)
	tree = ET.ElementTree(root)
	tree.write("../testimg/modified/" + (img_path.split("/")[-1]).split(".")[0] + ".xml")
	cv2.imwrite("../testimg/modified/" + img_path.split("/")[-1], img)

# Function calling all other functions related to detection
# Input:	Image path, Call number to this fuction
def run_yolo(img_path, call):
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width, call)
	draw_labels(boxes, confs, colors, class_ids, classes, image, img_path, call)
