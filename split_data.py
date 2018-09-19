import random

def split_data(data, ratio):
	""" Split data in train, valid and test datasets.
	"""

	len_data = len(data['labels'])
	assert len_data == len(data['labels'])

	len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)
	print(len_train, len_valid, len_test)

	splited_data = {'train': dict(), 'valid': dict(), 'test': dict()}
	
	for key in data:
		splited_data['train'][key] = data[key][ : len_train]
		splited_data['valid'][key] = data[key][len_train : len_train + len_valid]
		splited_data['test'][key] = data[key][len_train + len_valid : ]

	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])

	return splited_data



def split_data_2(data, ratio, num_labels):
	""" Split data in train, valid and test datasets.
	And check that train set contains representatives of all classes.
	"""

	len_data = len(data['labels'])
	assert len_data == len(data['labels'])

	len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)
	print(len_train, len_valid, len_test)

	splited_data = {'train': dict(), 'valid': dict(), 'test': dict()}
	
	for key in data:
		splited_data['train'][key] = data[key][ : len_train]
		splited_data['valid'][key] = data[key][len_train : len_train + len_valid]
		splited_data['test'][key] = data[key][len_train + len_valid : ]

	train_labels = set(splited_data['train']['labels'])
	valid_labels = set(splited_data['valid']['labels'])
	test_labels = set(splited_data['test']['labels'])

	print('\n train_labels:', train_labels)
	print('\n valid_labels:', valid_labels)
	print('\n test_labels:', test_labels)
	
	if len(valid_labels - train_labels) > 0:
		print('No labels {0} in train data but in valid'.format(valid_labels - train_labels))
	if len(test_labels - train_labels) > 0:
		print('No labels {0} in train data but in test'.format(test_labels - train_labels))

	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])

	return splited_data	



def split_data_3(data, ratio):
	""" Split data in train, valid and test datasets.
	And check that train set contains representatives of all classes.

	with shuffle
	"""

	zip3 = list(zip(data['labels'], data['images'], data['filenames']))
	random.shuffle(zip3)
	print('mix ok')

	labels = { z[0] for z in zip3 }
	category = dict()
	for label in labels:
		category[label] = { z for z in zip3 if z[0] == label}

	for label in labels:
		print('Label {0}: {1} images'.format())

	"""
	data['labels']    = [x[0] for x in zip3]
	data['images']    = [x[1] for x in zip3]
	data['filenames'] = [x[2] for x in zip3]


	len_data = len(data['labels'])
	assert len_data == len(data['labels'])

	len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)
	print(len_train, len_valid, len_test)

	splited_data = {'train': dict(), 'valid': dict(), 'test': dict()}
	
	for key in data:
		splited_data['train'][key] = data[key][ : len_train]
		splited_data['valid'][key] = data[key][len_train : len_train + len_valid]
		splited_data['test'][key] = data[key][len_train + len_valid : ]

	train_labels = set(splited_data['train']['labels'])
	valid_labels = set(splited_data['valid']['labels'])
	test_labels = set(splited_data['test']['labels'])

	print('\n train_labels:', train_labels)
	print('\n valid_labels:', valid_labels)
	print('\n test_labels:', test_labels)
	
	if len(valid_labels - train_labels) > 0:
		print('No labels {0} in train data but in valid'.format(valid_labels - train_labels))
	if len(test_labels - train_labels) > 0:
		print('No labels {0} in train data but in test'.format(test_labels - train_labels))

	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])


	return splited_data		
	"""