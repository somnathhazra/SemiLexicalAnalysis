import logicChecker as lg
import testimg as tg
import math
import cv2

# Wheel-wheel and wheel-frame distance range from training data
w1_w2_range = [0.23859128639156496, 0.669809639633901]
w_f_range = [0.13722042284020422, 0.42270145161622563]

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
	if callnum > 2:
		return 0
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
						ranges.append(dist)
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
						ranges.append(dist)
		
		# If in-range(wheels) occurs atleast once and in-range(wheel,frame) occurs atleast twice
		# -- it is a bicycle
		if count1 >= 1 and count2 >= 2:
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
						ranges.append(dist)
		
		# If in-range(wheel,frame) occurs atleast once
		# -- it is a bicycle
		if count >= 1:
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
			tg.run_yolo_custom(filenamec)
			coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
			# If a frame is found
			# -- run this logic again
			if len(coord_f) > 0:
				return compute_spatial_bi(directory,filenamec, callnum + 1)
			# Else it is not a bicycle
			else:
				return 0, []
	
	
	# If only 1 frame is detected
	# -- run YOLOv2 with reduced threshold
	if len(coord_f) >= 1:
		tg.run_yolo_custom(filenamec)
		coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
		# If a wheel is found
		# -- run this logic again
		if len(coord_w) >= 1:
			return compute_spatial_bi(directory,filenamec, callnum + 1)
		# Else it is not a bicycle
		else:
			return 0, []
	
	
	# If only 1 wheel is detected
	# -- run YOLOv2 with reduced threshold
	if len(coord_w) >= 1:
		tg.run_yolo_custom(filenamec)
		coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
		# If a frame is found
		# -- run this logic again
		if len(coord_f) > 0:
			return compute_spatial_bi(directory,filenamec, callnum + 1)
		# Else it is not a bicycle
		else:
			return 0, []
	
	
	# Else it is not a bicycle
	return 0, []

# Method to check unicycle
# Assumed that annotation file and image have same name
# Input:	Image name and annotation directory
# Output:	Returns 1 if it is unicycle; else returns 0
def compute_spatial_uni(directory, filename, callnum):
	filenamec = filename
	filename_img = directory + "/" + filename
	height, width, channels = cv2.imread(filename_img).shape
	# List of wheel and frame coordinates detected
	coord_wo, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
	
	# Removes multiple detection of wheels from list
	coord_w = []
	if len(coord_wo) > 0:
		coord_w.append(coord_wo[0])
		for i in range(1,len(coord_wo)):
			for j in range(len(coord_w)):
				if overlapping_obj(coord_w[j],coord_wo[i])==0:
					coord_w.append(coord_wo[i])
	
	# If atleast 1 frame is detected
	# -- not a unicycle
	if len(coord_f) > 0:
		return 0, []
	
	# If atleast 1 wheel is detected
	if len(coord_w) >= 1:
		# Run YOLOv2 with reduced threshold
		tg.run_yolo_custom(filenamec)
		coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
		# If atleast 1 frame is detected
		# -- it is not a unicycle
		if len(coord_f) > 0:
			return 0, []
		# If more than 1 wheels is detected
		if len(coord_w) > 1:
			count = 0
			ranges= []
			for i in range(len(coord_w)-1):
				for j in range(i+1, len(coord_w)):
					x_mid_w1 = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
					y_mid_w1 = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
					x_mid_w2 = (float(coord_w[j][0]) + float(coord_w[j][2])) / 2
					y_mid_w2 = (float(coord_w[j][1]) + float(coord_w[j][3])) / 2
					dist = compute_dist(x_mid_w1, x_mid_w2, y_mid_w1, y_mid_w2, width, height)
					
					# If distance between 2 wheels is within range
					# -- increment in-range wheels count
					ranges.append(dist)
					if dist >= w1_w2_range[0] and dist <= w1_w2_range[1]:
						count += 1
			
			# If in-range(wheels) occurs atleast once
			# -- it is not a unicycle	
			if count >= 1:
				return 0, []
			# Else it is a unicycle
			else:
				return 1, ranges
		# Else (only 1 wheel detected) it is a unicycle
		else:
			return 2, []
	
	# Else it is not a unicycle
	return 0, []

# Method to check tricycle
# Assumed that annotation file and image have same name
# Input:	Image name and annotation directory
# Output:	Returns 1 if it is tricycle; else returns 0
def compute_spatial_tri(directory, filename, callnum):
	if callnum > 2:
		return 0, []
	filenamec = filename
	filename_img = directory + "/" + filename
	height, width, channels = cv2.imread(filename_img).shape
	# List of wheel and frame coordinates detected
	coord_wo, coord_fo = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
	
	# Removes multiple detections from list
	coord_w = []
	coord_f = []
	if len(coord_wo) > 0:
		coord_w.append(coord_wo[0])
		for i in range(1,len(coord_wo)):
			flag = 0
			for j in range(len(coord_w)):
				if overlapping_obj(coord_w[j],coord_wo[i])==1:
					flag = 1
					break
			if flag==0:
				coord_w.append(coord_wo[i])
	if len(coord_fo) > 0:
		coord_f.append(coord_fo[0])
		for i in range(1,len(coord_fo)):
			for j in range(len(coord_f)):
				if overlapping_obj(coord_f[j],coord_fo[i])==1:
					flag = 1
					break
			if flag==0:
				coord_f.append(coord_fo[i])
	#print(filename,len(coord_f),len(coord_w))
	
	# If 3 wheels and 1 frame is detected
	if len(coord_w) >= 3 and len(coord_f) >= 1:
		count1 = 0
		count2 = 0
		ranges= []
		for i in range(len(coord_w)-1):
			for j in range(i+1, len(coord_w)):
				x_mid_w1 = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
				y_mid_w1 = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
				x_mid_w2 = (float(coord_w[j][0]) + float(coord_w[j][2])) / 2
				y_mid_w2 = (float(coord_w[j][1]) + float(coord_w[j][3])) / 2
				dist = compute_dist(x_mid_w1, x_mid_w2, y_mid_w1, y_mid_w2, width, height)
				# If distance between wheels is within range
				# -- increment in-range wheels count
				if dist >= w1_w2_range[0] and dist <= w1_w2_range[1]:
					count1 += 1
					if count1 <= 2:
						ranges.append(dist)
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
					if count2 <= 3:
						ranges.append(dist)
		
		# If in-range(wheels) occurs atleast twice and in-range(wheel,frame) occurs atleast thrice
		# -- it is a tricycle
		if count1 == 2 and count2 == 3:
			return 1, ranges
	
	
	# If 2 wheels and 2 frames are detected
	# (Some tricycle have 2 non overlapping frames)
	if len(coord_w) >= 2 and len(coord_f) >= 2:
		count1 = 0
		count2 = 0
		ranges= []
		for i in range(len(coord_w)-1):
			for j in range(i+1, len(coord_w)):
				x_mid_w1 = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
				y_mid_w1 = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
				x_mid_w2 = (float(coord_w[j][0]) + float(coord_w[j][2])) / 2
				y_mid_w2 = (float(coord_w[j][1]) + float(coord_w[j][3])) / 2
				dist = compute_dist(x_mid_w1, x_mid_w2, y_mid_w1, y_mid_w2, width, height)
				# If distance between wheels is within range
				# -- increment in-range wheels count
				if dist >= w1_w2_range[0] and dist <= w1_w2_range[1]:
					count1 += 1
					if count1 <= 1:
						ranges.append(dist)
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
						ranges.append(dist)
		
		# If in-range(wheels) occurs atleast once and in-range(wheel,frame) occurs atleast twice
		# -- it is a tricycle
		if count1 == 1 and count2 == 2:
			return 2, ranges
	
	
	# If 3 wheels or more are detected
	if len(coord_w) >= 3:
		count = 0
		ranges= []
		for i in range(len(coord_w)-1):
			for j in range(i+1, len(coord_w)):
				x_mid_w1 = (float(coord_w[i][0]) + float(coord_w[i][2])) / 2
				y_mid_w1 = (float(coord_w[i][1]) + float(coord_w[i][3])) / 2
				x_mid_w2 = (float(coord_w[j][0]) + float(coord_w[j][2])) / 2
				y_mid_w2 = (float(coord_w[j][1]) + float(coord_w[j][3])) / 2
				dist = compute_dist(x_mid_w1, x_mid_w2, y_mid_w1, y_mid_w2, width, height)
				# If distance between wheels is within range
				# -- increment in-range wheels count
				if dist >= w1_w2_range[0] and dist <= w1_w2_range[1]:
					count += 1
					if count <= 2:
						ranges.append(dist)
		
		# If in-range(wheels) occurs atleast twice
		# -- it is a tricycle
		if count >= 2:
			return 3, ranges
	
	# If 2 wheels or more and atleast 1 frame is detected
	if len(coord_w) >= 2 and len(coord_f) >= 1:
		count = 0
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
		
		# If in-range(wheel,frame) count occurs twice
		# -- run YOLOv2 with reduced threshold to look for another wheel
		if count == 2:
			tg.run_yolo_custom(filenamec)
			coord_w, coord_f = lg.parse_xml(directory + "/" + filename.split(".")[0] + ".xml")
			# If found another wheel
			# -- run this logic again
			if len(coord_w) > 0:
				return compute_spatial_tri(directory,filenamec, callnum + 1)
			# Else it is not a tricycle
			else:
				return 0, []
	
				
	# Else it is not a tricycle
	return 0, []
