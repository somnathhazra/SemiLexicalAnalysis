# Program uses CNN to recognise handwritten digits from a visual Soduku board
# The digits are semi lexical hence some of them maybe recognized ambiguously
# The conflict caused by ambiguity is resolved using a constraint rules written in Sudoku.py
# The wrong examples are explained using euclidean distance from the pooling layer activations using DistanceUtility.py
# @Author : Briti Gangopadhyay
# @Institution : IIT Kharagpur

from keras.models import model_from_json
import numpy as np
import cv2
import Sudoku as sd
import DistanceUtility as ds
import csv
import os
import sys
import copy
import random as rd
from keras import Model
from keras.datasets import mnist
os.environ['KMP_WARNINGS'] = 'off'
# For setting random numbers and getting the same result everytime
rd.seed(32)
argumentList = sys.argv

# global variable suduko board
board = np.zeros((9, 9))
# dictionary containing all the conflicting pairs
conflict_pairs = {}
# Dictionary to store the confidence of the numbers [(0,1) : {1:22,7:78}]
semilexicalcell_pred = {}
# Maintaining a set of conflict tuples encountered
conflict_set = set()
cnn_pred_board = []
input_board = []
model_refined = 0
train_encoding = 0

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

def load_model():
	# load json and create model
	json_file = open('model_digit.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model_digit.h5")
	print("Loaded model from disk")
	return loaded_model

# Create a new model from the trained model
# This produces the output of the pooling layer

def create_max_pool_model(layer,model):
    model_refined = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    train_encoding = model_refined.predict(X_train)
    return model_refined, train_encoding


def inference(model, img):
	pr = model.predict_classes(img.reshape(1, 28, 28, 1).astype('float32') / 255)
	return pr


# Deduce all the conflicting pairs once a conflict is found
def check_valid_board(board):
	# Check row values for all conflicts first
	row_num = 0
	column_num = 0
	conflict_id = 1
	# Check for all the conflicting pairs in a row and there related conflicts
	for row in board:
		i = 0
		conflict_stack = []
		while i < len(row):
			j = i + 1
			while j < len(row):
				# if there is a conflict in the row and that connflict cell has not been encountered previously
				if board[row_num][i] == board[row_num][j] and (row_num, i) not in conflict_set:
					# Add the first conflicting pairs in a stack
					conflict_stack.append((row_num, i))
					conflict_stack.append((row_num, j))
					# Check for other conflicts of the conflicting pair
					# Adding the conflict id in the map
					if conflict_id not in conflict_pairs.keys():
						conflict_pairs[conflict_id] = set()
					# check for conflict until the conflict stack is empty
					while len(conflict_stack) != 0:
						ele = conflict_stack.pop()
						conflict_pairs.get(conflict_id).add(ele)
						conflict_set.add(ele)
						ele_row = ele[0]
						ele_col = ele[1]
						k = 0
						# Check for other conflict cells in row and column
						while k < len(row):
							if (board[ele_row][k] == board[ele_row][ele_col]) and (
									(ele_row, k) not in conflict_set) and (
									ele_row, k) not in conflict_stack:
								conflict_stack.append((ele_row, k))
							if (board[k][ele_col] == board[ele_row][ele_col]) and (
									(k, ele_col) not in conflict_set) and (
									k, ele_col) not in conflict_stack:
								conflict_stack.append((k, ele_col))
							k = k + 1
					conflict_id = conflict_id + 1
				j = j + 1
			i = i + 1
		row_num = row_num + 1

	# Check for column conflict this only applies if the ambiguities are swapped in a row
	board_transpose = board.T
	for column in board_transpose:
		i = 0
		while i < len(column):
			j = i + 1
			while j < len(column):
				if board_transpose[column_num][i] == board_transpose[column_num][j] and (
						i, column_num) not in conflict_set:
					# Check for other conflicts of the conflicting pair
					# Adding the conflict id in the map
					if conflict_id not in conflict_pairs.keys():
						conflict_pairs[conflict_id] = set()
					conflict_pairs.get(conflict_id).add((i, column_num))
					conflict_pairs.get(conflict_id).add((j, column_num))
					conflict_set.add((i, column_num))
					conflict_set.add((j, column_num))
					conflict_id = conflict_id + 1
				j = j + 1
			i = i + 1
		column_num = column_num + 1

	if len(conflict_pairs) == 0:
		return True
	return False


# Function to remove ambiguities
# And return the correct solution board
def create_correct_solution(board):
	key = []
	for conflict_id in conflict_pairs.keys():
		if conflict_id in key:
			continue
		# Case 1: unique row conflict
		if len(conflict_pairs.get(conflict_id)) == 3:
			set_number = {1, 2, 3, 4, 5, 6, 7, 8, 9}
			conflict_list = list(conflict_pairs.get(conflict_id))
			conflict_list.sort(key = lambda x: x[0])
			print(conflict_list)
			if conflict_list[0][0] != conflict_list[1][0] and conflict_list[0][1] != conflict_list[1][1]:
				conflict_ele = conflict_list[2]
			elif conflict_list[0][0] != conflict_list[2][0] and conflict_list[0][1] != conflict_list[2][1]:
				conflict_ele = conflict_list[1]
			else:
				conflict_ele = conflict_list[0]
			col_num = conflict_ele[1]
			for i in range(9):
				try:
					set_number.remove(board[i][col_num])
				except:
					print('The key' + str(board[i][col_num]) + ' has already been removed')
			board[conflict_ele[0]][conflict_ele[1]] = list(set_number)[0]
			key.append(conflict_id)
		# Case 2: Conflict in only column when row items are swapped
		elif len(conflict_pairs.get(conflict_id)) == 2:
			conflict_list = list(conflict_pairs.get(conflict_id))
			column_of_conflict = conflict_list[0][1]
			set_number = {1, 2, 3, 4, 5, 6, 7, 8, 9}
			for i in range(9):
				try:
					set_number.remove(board[i][column_of_conflict])
				except:
					pass
			conflict_tuples = None
			# Check which the common row between two tuple conflicts and if the number required is in the conflicted tuple
			for k , v in conflict_pairs.items():
				if list(v) != conflict_list and len(v) == 2:
					v = list(v)
					if conflict_list[0][0] == v[0][0]:
						conflict_tuples = (v[0],conflict_list[0])
					elif conflict_list[1][0] == v[0][0]:
						conflict_tuples = (v[0],conflict_list[1])
					elif conflict_list[0][0] == v[1][0]:
						conflict_tuples = (v[1],conflict_list[0])
					elif conflict_list[1][0] == v[1][0]:
						conflict_tuples = (v[1],conflict_list[1])
					# If one common row conflict tuple was found
					if conflict_tuples != None:
						# The conflict tuple has the missing number then swap the two pairs
						if board[conflict_tuples[0][0]][conflict_tuples[0][1]] == list(set_number)[0]:
							key.append(k)
							key.append(conflict_id)
							temp = board[conflict_tuples[0][0]][conflict_tuples[0][1]]
							board[conflict_tuples[0][0]][conflict_tuples[0][1]] = board[conflict_tuples[1][0]][conflict_tuples[1][1]]
							board[conflict_tuples[1][0]][conflict_tuples[1][1]] = temp
							break

		# Case 3 : When there is cyclic dependency and the board needs to be solved
		else:
			conflict_list = list(conflict_pairs.get(conflict_id))
			key.append(conflict_id)
			for conflict_item in conflict_list:
				board[conflict_item[0]][conflict_item[1]] = 0
			print(np.array(board))
	
	# Make the positions 0 for every conflict that could not be resolved
	for conflict_id in conflict_pairs.keys():
		if conflict_id in key:
			continue
		else:
			conflict_list = conflict_pairs.get(conflict_id)
			for ele in conflict_list:
				board[ele[0]][ele[1]] = 0
	board = sd.call_solve_sudoku(np.asarray(board))
	if not board:
		print("This board cannot be solved")
		return False
	return board


if __name__ == '__main__':
	'''#Part to fill up the board on the basis of global consistency
	model = load_model()
	model_refined, train_encoding = create_max_pool_model(5,model)
	path = './SudokoPerfectSolution/'
	with open(str(path + 'configuration_' + str(argumentList[1]) +'.csv'), newline='') as csvfile:
		conf_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		row_count = 0
		for row in conf_reader:
			col_count = 0
			input_board.append(row)
			for x in row:
				new_path = ''
				new_path = path + x + '.png'
				test_image = ds.read_image(new_path)
				# Computing global consistency, if globally consistent fill the board position else, 
				# Store confidence for cell in decreasing order
				support = ds.compute_support(test_image, 100,model_refined,train_encoding)
				predicted = int(inference(model, cv2.imread(new_path,0)))
				if support.get(predicted) > 80:
					board[row_count][col_count] = predicted
				else:
					board[row_count][col_count] = 0
					semilexicalcell_pred[(row_count,col_count)]=support
				col_count = col_count + 1
			row_count = row_count + 1
	print(semilexicalcell_pred)
	print(board)
	cnn_pred_board = copy.deepcopy(board)'''
	#Just dummy testing for the backtracking algorithm
	#semilexicalcell_pred = {(0, 0): {9: 68, 4: 21, 5: 10, 8: 1}, (0, 1): {3: 64, 5: 30, 7: 3, 9: 2, 0: 1}, (0, 4): {6: 75, 5: 11, 9: 10, 4: 4}, (1, 2): {6: 75, 5: 11, 9: 10, 4: 4}, (1, 3): {3: 76, 5: 24}, (2, 5): {9: 68, 4: 21, 5: 10, 8: 1}, (2, 7): {3: 76, 5: 24}, (2, 8): {6: 75, 5: 11, 9: 10, 4: 4}, (3, 2): {3: 76, 5: 24}, (3, 3): {9: 68, 4: 21, 5: 10, 8: 1}, (3, 6): {6: 75, 5: 11, 9: 10, 4: 4}, (4, 3): {6: 75, 5: 11, 9: 10, 4: 4}, (4, 6): {3: 76, 5: 24}, (5, 0): {6: 75, 5: 11, 9: 10, 4: 4}, (5, 5): {3: 76, 5: 24}, (6, 0): {3: 76, 5: 24}, (6, 5): {6: 75, 5: 11, 9: 10, 4: 4}, (7, 7): {6: 75, 5: 11, 9: 10, 4: 4}, (7, 8): {3: 76, 5: 24}, (8, 1): {6: 75, 5: 11, 9: 10, 4: 4}, (8, 4): {3: 76, 5: 24}}
	#board = [[0,0,5,2,0,9,7,8,1],[1,2,0,0,7,8,4,5,9],[9,7,8,1,5,0,2,0,0],[2,1,0,0,8,5,0,9,7],[5,4,7,0,9,1,0,2,8],[0,8,9,7,2,0,1,4,5],[0,5,1,8,4,0,9,7,2],[7,9,4,5,1,2,8,0,0],[8,0,2,9,0,7,5,1,4]]
	# print(semilexicalcell_pred)
	semilexicalcell_pred = {(0, 1): {9: 50, 4: 50}, (1, 1): {3: 50, 5: 50}, (1, 2): {9: 50, 4: 50}}
	board = [[6, 0, 1, 8, 7, 4, 5, 3, 2], [2, 0, 0, 1, 3, 9, 6, 8, 7], [8, 3, 7, 6, 2, 5, 4, 1, 9], [1, 6, 3, 9, 5, 7, 2, 4, 8], [9, 8, 2, 4, 1, 6, 7, 5, 3], [4, 7, 5, 3, 8, 2, 1, 9, 6], [3, 2, 8, 7, 4, 1, 9, 6, 5], [5, 1, 9, 2, 6, 3, 8, 7, 4], [7, 4, 6, 5, 9, 8, 3, 2, 1]]
	print('------------------Board before-------------------')
	print(board)
	val = check_valid_board(np.asarray(board))
	if not val:
		board = sd.call_solve_sudoku(np.asarray(board),semilexicalcell_pred)
	print('------------------Board after-------------------')
	print(board)
	'''if not val:
		new_board = create_correct_solution(board)
		if new_board:
			for row in new_board:
				print(row)

		dict_correct_cell = {}
		dict_incorrect_cell = {}
		for j in range(0,9):
			for k in range(0,9):
				if cnn_pred_board[j][k] == new_board[j][k]:
					if cnn_pred_board[j][k] in dict_correct_cell.keys():
						dict_correct_cell[cnn_pred_board[j][k]].append((j,k))
					else:
						dict_correct_cell[cnn_pred_board[j][k]] = [(j,k)]
				else:
					if new_board[j][k] in dict_incorrect_cell.keys():
						dict_incorrect_cell[new_board[j][k]].append((j,k))
					else:
						dict_incorrect_cell[new_board[j][k]]= [(j,k)]

		for k in dict_incorrect_cell.keys():
			for cell in dict_incorrect_cell[k]:
				test = ds.read_image('./SudokoPerfectSolution/'+input_board[cell[0]][cell[1]]+'.png')
				print('Support For :'+input_board[cell[0]][cell[1]]+'.png')
				ds.compute_support(test, 100)
				
				# print("input_board[cell[0]][cell[1]]", input_board[cell[0]][cell[1]])
				distance = []
				for corr_cell in dict_correct_cell[k]:
					# print("input_board[corr_cell[0]][corr_cell[1]]", input_board[corr_cell[0]][corr_cell[1]])
					distance.append((k, ds.compute_distance(ds.read_image('./SudokoPerfectSolution/'+input_board[corr_cell[0]][corr_cell[1]]+'.png'),
					ds.read_image('./SudokoPerfectSolution/'+input_board[cell[0]][cell[1]]+'.png'))))
				for corr_cell in dict_correct_cell[cnn_pred_board[cell[0]][cell[1]]]:
					distance.append((cnn_pred_board[cell[0]][cell[1]], ds.compute_distance(ds.read_image('./SudokoPerfectSolution/'+input_board[corr_cell[0]][corr_cell[1]]+'.png'), ds.read_image('./SudokoPerfectSolution/'+input_board[cell[0]][cell[1]]+'.png'))))
				distance.sort(key = lambda x: x[1])
				print(distance)'''