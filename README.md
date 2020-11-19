# Semi Lexical Languages
This repository contains code for testing of Semi-Lexical Languages- a formal basis with imperfect tokens presented by the real world in an interpretable way. We combine the generalization capability of deep learning algorithms along with the interpretability of deduction systems. This repository mainly contains files for testing of integrated setup tested for dealing with two different scenarios as highlighted by the two directories. All codes are written in Python 3.6 and above.

## Component Experiments
This section perticularly involves retraining of YOLOv3 network (https://arxiv.org/abs/1804.02767), to identify components. The retraining was done in Google Colab with configuration and weight files from this repository: https://github.com/AlexeyAB/darknet. The configuration file that we had used has been included in the directory. The weight file is to be replaced with a retrained weights file. The weights and configuration files are loaded for testing with open-cv. The test directories are to be included in the [testimg] sub-directory which will aslo contain the output files from the algorithm. To run the experiments, execute the mainDriver.py script. All tests were conducted on a Ubuntu 18.04 system and a remote server running on CentOS.
