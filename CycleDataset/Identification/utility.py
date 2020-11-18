import logicChecker as lg
import testimg as tg
import math
import cv2
from csv import writer

# Wheel-wheel and wheel-frame distance range from training data
w1_w2_range = [0.23859128639156496, 0.669809639633901]
w_f_range = [0.13722042284020422, 0.42270145161622563]

def append_list_as_row(file_name, list_of_elem):
	# Open file in append mode
	with open(file_name, 'a+', newline='') as write_obj:
		# Create a writer object from csv module
		csv_writer = writer(write_obj)
		# Add contents of list as last row in the csv file
		csv_writer.writerow(list_of_elem)

# Computes Euclidean distance between 2 bounding boxes
# Input:	Coordinates of mid point of the two bounding boxes, width and height of image
# Output:	Normalised Euclidean distance by width and height of image
def compute_dist(x_mid_1, x_mid_2, y_mid_1, y_mid_2, width, height):
	return math.sqrt(((x_mid_1 - x_mid_2) / width) ** 2 + ((y_mid_1 - y_mid_2) / height) ** 2)

# Computes the level of overlap between two bounding boxes
# Input:	Coordinate positions of the two bounding boxes in the image
# Output:	Returns 1 of overlap is more than 50%; else returns 0
def overlapping_obj(coord1, coord2):
	area1 = 0
	for i in range(int(coord2[0]),int(coord2[2])):
		for j in range(int(coord2[1]), int(coord2[3])):
			if int(coord1[0]) <= i and i <= int(coord1[2]) and int(coord1[1]) <= j and j <= int(coord1[3]):
				area1 += 1
	area2 = (int(coord1[2]) - int(coord1[0])) * (int(coord1[3]) - int(coord1[1]))
	
	# If overlap is more than 50% then ignore
	if area1/area2 >= 0.5:
		return 1
	else:
		return 0

# Method to check bicycle
# Assumed that annotation file and image have same name
# Input:	Image name and annotation directory
# Output:	Returns 1 if it is bicycle; else returns 0
def compute_spatial_bi(directory, filename, callnum):
	if callnum > 3:
		append_list_as_row('../testimg/counterfactual.csv', [filename, 'No significant components detected'])
		return 0, []
	filenamec = filename
	filename_img = directory + "/" + filename
	height, width, channels = cv2.imread(filename_img).shape
	# Lists of wheel and frame coordinates detected
	coord_wo, coord_fo = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
	
	coord_w = []
	coord_f = []
	if len(coord_wo) > 0:
		coord_w.append(coord_wo[0])
		for i in range(1,len(coord_wo)):
			for j in range(len(coord_w)):
				if overlapping_obj(coord_w[j],coord_wo[i])==0:
					coord_w.append(coord_wo[i])
	if len(coord_fo) > 0:
		coord_f.append(coord_fo[0])
		for i in range(1,len(coord_fo)):
			for j in range(len(coord_f)):
				if overlapping_obj(coord_f[j],coord_fo[i])==0:
					coord_f.append(coord_fo[i])
	
	# If 2 wheels and 1 frame is detected
	if len(coord_w) >= 2 and len(coord_f) >= 1:
		count1 = 0
		count2 = 0
		ranges = []
		for i in range(len(coord_w)-1):
			for j in range(i+1, len(coord_w)):
				x_mid_w1 = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
				y_mid_w1 = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
				x_mid_w2 = (float(coord_w[j][0]) + float(coord_w[j][2])) / 2
				y_mid_w2 = (float(coord_w[j][1]) + float(coord_w[j][3])) / 2
				dist = compute_dist(x_mid_w1, x_mid_w2, y_mid_w1, y_mid_w2, width, height)
				# If distance between 2 wheels is within range
				# -- increment in-range wheels count
				if dist >= w1_w2_range[0] and dist <= w1_w2_range[1]:
					count1 += 1
					if count1 <= 1:
						ranges.append(str(dist))
		for i in range(len(coord_w)):
			for j in range(len(coord_f)):
				x_mid_w = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
				y_mid_w = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
				x_mid_f = (float(coord_f[j][0]) + float(coord_f[j][2])) / 2
				y_mid_f = (float(coord_f[j][1]) + float(coord_f[j][3])) / 2
				dist = compute_dist(x_mid_w, x_mid_f, y_mid_w, y_mid_f, width, height)
				# If distance between wheel and frame is within range
				# -- increment in-range wheel, frame count
				if dist >= w_f_range[0] and dist <= w_f_range[1]:
					count2 += 1
					if count2 <= 2:
						ranges.append(str(dist))
		
		# If in-range(wheels) occurs atleast once and in-range(wheel,frame) occurs atleast twice
		# -- it is a bicycle
		if count1 >= 1 and count2 >= 2:
			append_list_as_row('../testimg/explanations.csv', [filenamec, '2 wheels, 1 frame', 'Wheel-wheel range: ' + ranges[0] + '; Frame-wheel ranges: ' + ranges[1] + ', ' + ranges[2]])
			return 1, ranges
	
	
	# If 1 wheel and 1 frame is detected
	if len(coord_w) >= 1 and len(coord_f) >= 1:
		count = 0
		ranges = []
		for i in range(len(coord_w)):
			for j in range(len(coord_f)):
				x_mid_w = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
				y_mid_w = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
				x_mid_f = (float(coord_f[j][0]) + float(coord_f[j][2])) / 2
				y_mid_f = (float(coord_f[j][1]) + float(coord_f[j][3])) / 2
				dist = compute_dist(x_mid_w, x_mid_f, y_mid_w, y_mid_f, width, height)
				# If distance between wheel and frame is within range
				# -- increment in-range wheel, frame count
				if dist >= w_f_range[0] and dist <= w_f_range[1]:
					count += 1
					if count <= 1:
						ranges.append(str(dist))
		
		# If in-range(wheel,frame) occurs atleast once
		# -- it is a bicycle
		if count >= 1:
			append_list_as_row('../testimg/explanations.csv', [filenamec, '1 wheel, 1 frame', 'Frame-wheel range: ' + ranges[0]])
			return 2, ranges
	
	
	# If two wheels are detected
	if len(coord_w) >= 2:
		count = 0
		for i in range(len(coord_w)-1):
			for j in range(i+1, len(coord_w)):
				x_mid_w1 = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
				y_mid_w1 = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
				x_mid_w2 = (float(coord_w[j][0]) + float(coord_w[j][2])) / 2
				y_mid_w2 = (float(coord_w[j][1]) + float(coord_w[j][3])) / 2
				dist = compute_dist(x_mid_w1, x_mid_w2, y_mid_w1, y_mid_w2, width, height)
				# If distance between 2 wheels is within range
				# -- increment in-range wheels count
				if dist >= w1_w2_range[0] and dist <= w1_w2_range[1]:
					count += 1
		
		# If in-range(wheels) occurs atleast once
		# -- run YOLOv2 with reduced threshold
		if count >= 1:
			tg.run_yolo('../testimg/actual/' + filenamec, callnum + 1)
			coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
			# If a frame is found
			# -- run this logic again
			if len(coord_f) > 0:
				return compute_spatial_bi(directory, filenamec, callnum + 1)
			# Else it is not a bicycle
			else:
				# append_list_as_row('../testimg/counterfactual.csv', [filenamec, 'No significant components detected'])
				append_list_as_row('../testimg/counterfactual.csv', [filenamec, 'Only 2 wheels detected within range without frame'])
				return 0, []
	
	
	# If only 1 frame is detected
	# -- run YOLOv2 with reduced threshold
	if len(coord_f) >= 1:
		tg.run_yolo('../testimg/actual/' + filenamec, callnum + 1)
		coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
		# If a wheel is found
		# -- run this logic again
		if len(coord_w) >= 1:
			return compute_spatial_bi(directory, filenamec, callnum + 1)
		# Else it is not a bicycle
		else:
			append_list_as_row('../testimg/counterfactual.csv', [filenamec, 'Only 1 frame detected without any wheels'])
			return 0, []
	
	
	# If only 1 wheel is detected
	# -- run YOLOv2 with reduced threshold
	if len(coord_w) >= 1:
		tg.run_yolo('../testimg/actual/' + filenamec, callnum + 1)
		coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
		# If a frame is found
		# -- run this logic again
		if len(coord_f) > 0:
			return compute_spatial_bi(directory,filenamec, callnum + 1)
		# Else it is not a bicycle
		else:
			append_list_as_row('../testimg/counterfactual.csv', [filenamec, 'Only 1 wheel detected without frame'])
			return 0, []
	
	
	# Else it is not a bicycle
	append_list_as_row('../testimg/counterfactual.csv', [filenamec, 'No significant components detected'])
	return 0, []
