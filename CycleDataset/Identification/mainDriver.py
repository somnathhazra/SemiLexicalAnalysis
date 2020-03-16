import os
import shutil
import utility
import testimg as tg
import pandas as pd

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
		shutil.cpytree(s, d, symlinks, ignore)
	else:
		shutil.copy2(s, d)

# Runs the trained YOLOv2 network on the test images
# The annotated xml files are generated in modified folder
tg.run_yolo()

if os.path.exists('../testimg/explanations.csv'):
	os.remove('../testimg/explanations.csv')

algo = input('Bicycle, Unicycle or Tricycle?: ')

if 'Bicycle' in algo:
	# Calls the bicycle logic function in utility passing the images from modified folder
	# Prints the count for which logic returned true
	count = 0
	f = []
	e = []
	o = []
	d1 = []
	d2 = []
	for filename in os.listdir(dst):
		if ".jpg" in filename or ".jpeg" in filename:
			result, ranges = utility.compute_spatial_bi(dst, filename, 1)
			if result != 0:
				f.append(filename)
				count += 1
				if result == 1:
					o.append('1 frame between 2 wheels')
					e.append('frame to wheels distance and wheel to wheel distance, within range')
					d1.append(str(ranges[0]))
					d2.append(str(ranges[1]) + ", " + str(ranges[2]))
				else:
					o.append('1 wheel, 1 frame')
					e.append('frame to wheel distance within range')
					d1.append('NA')
					d2.append(str(ranges[0]))
		
	dict1 = {'File': f, 'Objects detected': o, 'Explanation': e, 'Wheel-Wheel normalized distance': d1, 'Wheel-Frame normalized distance': d2}
	df = pd.DataFrame(dict1)
	df.to_csv('../testimg/explanations.csv', index = False)
			
	f.sort()
	print(f)
	print('Bicycle detcted:',count)

elif 'Unicycle' in algo:
	# Calls the unicycle logic function in utility passing the images from modified folder
	# Prints the count for which logic returned true
	count = 0
	f = []
	e = []
	o = []
	d1 = []
	d2 = []
	for filename in os.listdir(dst):
		if ".jpg" in filename or ".jpeg" in filename:
			result, ranges = utility.compute_spatial_uni(dst, filename, 1)
			if result != 0:
				f.append(filename)
				count += 1
				if result == 1:
					str_ranges = str(ranges[0])
					for i in range(1, len(ranges)):
						str_ranges = str_ranges + ", " + str(ranges[i])
					o.append('More than 1 wheel')
					e.append('No wheel to wheel distance within range')
					d1.append(str_ranges)
					d2.append('NA')
				else:
					o.append('1 wheel')
					e.append('Only 1 wheel and no frames detected')
					d1.append('NA')
					d2.append('NA')
	
	dict1 = {'File': f, 'Objects detected': o, 'Explanation': e, 'Wheel-Wheel normalized distance': d1, 'Wheel-Frame normalized distance': d2}
	df = pd.DataFrame(dict1)
	df.to_csv('../testimg/explanations.csv', index = False)
	
	f.sort()
	print(f)
	print('Unicycle detcted:',count)

elif 'Tricycle' in algo:
	# Calls the tricycle logic function in utility passing the images from modified folder
	# Prints the count for which logic returned true
	count = 0
	f = []
	e = []
	o = []
	d1 = []
	d2 = []
	for filename in os.listdir(dst):
		if ".jpg" in filename or ".jpeg" in filename:
			result, ranges = utility.compute_spatial_tri(dst, filename, 1)
			if result != 0:
				f.append(filename)
				count += 1
				if result == 1:
					o.append('1 frame between 3 wheels')
					e.append('frame to wheels distance and wheel to wheel distance, within range')
					d1.append(str(ranges[0]) + ", " + str(ranges[1]))
					d2.append(str(ranges[2]) + ", " + str(ranges[3]) + ", " + str(ranges[4]))
				elif result == 2:
					o.append('2 frames detected between wheels')
					e.append('frame to wheels distance and wheel to wheel distance, within range')
					d1.append(str(ranges[0]))
					d2.append(str(ranges[1]) + ", " + str(ranges[2]))
				else:
					o.append('3 wheels detected')
					e.append('wheel to wheel distance within range')
					d1.append(str(ranges[0]) + ", " + str(ranges[1]))
					d2.append('NA')
	
	dict1 = {'File': f, 'Objects detected': o, 'Explanation': e, 'Wheel-Wheel normalized distance': d1, 'Wheel-Frame normalized distance': d2}
	df = pd.DataFrame(dict1)
	df.to_csv('../testimg/explanations.csv', index = False)
	
	f.sort()
	print(f)
	print('Tricycle detcted:',count)

shutil.rmtree(src)
