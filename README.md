# Semi Lexical Languages
This repository contains code for testing of Semi-Lexical Languages- a formal basis with imperfect tokens presented by the real world in an interpretable way. We combine the generalization capability of deep learning algorithms along with the interpretability of deduction systems. This repository mainly contains files for testing of integrated setup tested for dealing with two different scenarios as highlighted by the two directories. All codes are written in Python 3.6 and above.

## Component Experiments
This section perticularly involves retraining of YOLOv3 network (https://arxiv.org/abs/1804.02767), to identify components. The configuration file that we had used has been included in the directory. The weight file is to be replaced with a retrained weights file. The test directories are to be included in the [testimg] sub-directory which will aslo contain the output files from the algorithm. Furthur details are provided in the [Readme.txt] files inside the directory.

### Training
1. Trained on Google Colab
2. Requires GPU for training (Colab's GPU)
3. This repository contains the cfg and weights files: https://github.com/AlexeyAB/darknet
4. The training set was created using https://github.com/tzutalin/labelImg

### Testing
1. Tested on Ubuntu 18.04 system and CentOS remote server
2. Does not need GPU for testing
3. The cfg and weight files were loaded using open-cv
4. Execute mainDriver.py to run the experiments
5. Needs a retrained weights file and some test images inside a directory inside [testimg] directory
