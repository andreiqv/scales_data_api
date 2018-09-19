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

	for label in range(num_labels):
		if label not in splited_data['train']['labels']:
			print('label {0} not in train list'.format(label))


	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])

	return splited_data	