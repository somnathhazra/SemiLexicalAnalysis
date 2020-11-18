import os
import shutil
import utility
import testimg as tg
import pandas as pd
from tqdm import tqdm

src = '../testimg/actual'
dst = '../testimg/modified'
srf = input('Input directory: ')
srf = os.path.join('../testimg/', srf)
try:
	print('Number of files:', len([name for name in os.listdir(srf) if os.path.isfile(os.path.join(srf, name))]))
except:
	print('No such directory found.')
	exit()

# Creating directories for saving the results of detection process
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

# Starting of the initial component detection process
l = os.listdir(src)
print('Starting detection process...')
for i in tqdm(range(len(l))):
	try:
		tg.run_yolo(src + "/" + l[i], 0)
	except:
		pass


count = 0
f = []

# Initializing the csv files for explaining positives and negatives
dict1 = {'File': [], 'Objects detected': [], 'Explanation': []}
df = pd.DataFrame(dict1)
df.to_csv('../testimg/explanations.csv', index = False)

dict2 = {'File': [], 'Explanation': []}
df = pd.DataFrame(dict2)
df.to_csv('../testimg/counterfactual.csv', index = False)

# Starting the deduction process after detecting the components
l = os.listdir(dst)
print('Stating deduction process...')
for i in tqdm(range(len(l))):
	if ".jpg" in l[i] or ".jpeg" in l[i]:
		try:
			result, ranges = utility.compute_spatial_bi(dst, l[i], 0)
		except:
			result = 0
		if result != 0:
			f.append(l[i])
			count += 1

# Printing the filenames that gave positive results and count of positives
f.sort()
print(f)
print('Detected: ', count)

# Removing unnecessary directory
shutil.rmtree(src)
