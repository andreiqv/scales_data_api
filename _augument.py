def augment_dataset(dataset):
	dataset.train_set = dataset.train_set.shuffle(60000).repeat(5).batch(128)

	def _random_distord(images, labels):
		rand = tf.random_uniform(shape=(1,), minval=0, maxval=2)
		toss = tf.less(rand, tf.constant([1.0], dtype=tf.float32))
		toss = tf.reshape(toss, [])

		def _flip_left():
			fliped_im = tf.image.flip_left_right(images)

			mirror = tf.constant([-1], dtype=tf.float32)
			mirror = tf.expand_dims(mirror, 0)
			mirror = tf.pad(mirror, [[0, 0], [0, tf.shape(labels)[1] - 1]], "CONSTANT", constant_values=1)
			mirror = tf.squeeze(mirror)
			mirror = tf.multiply(labels, mirror)  # [-0.34, 0.4, 0.28, 0,21, 1]

			basis = tf.constant([1], dtype=tf.float32)
			basis = tf.expand_dims(basis, 0)
			basis = tf.pad(basis, [[0, 0], [0, tf.shape(labels)[1] - 1]], "CONSTANT")  # [1, 0, 0 0 0]
			basis = tf.squeeze(basis)  # ?

			fliped_label = basis + mirror  # [0.6599, 0.4, 0.28, 0.21, 1]

			# eliminate fucking empty image coordinates problem
			# https://stackoverflow.com/questions/50538038/tf-data-dataset-mapmap-func-with-eager-mode
			new_labels = tf.multiply(fliped_label[:, 0], fliped_label[:, 4])
			new_labels = tf.expand_dims(new_labels, 1)
			new_labels = tf.concat([new_labels, fliped_label[:, 1:]], 1)

			return fliped_im, new_labels

		images, labels = tf.cond(toss, _flip_left, lambda: (images, labels))

		images = tf.image.random_hue(images, max_delta=0.05)
		images = tf.image.random_contrast(images, lower=0.3, upper=1.8)
		images = tf.image.random_brightness(images, max_delta=0.3)
		images = tf.image.random_saturation(images, lower=0.0, upper=2.0)

		images = tf.minimum(images, 1.0)
		images = tf.maximum(images, 0.0)

		return images, labels

	dataset.train_set = dataset.train_set.map(_random_distord)

	return dataset



class BulkImages:

    def __init__(self, path_list, image_size) -> None:
        self.path_list = path_list
        self.image_size = image_size
        self.load_images()
        super().__init__()

    def load_images(self):
        image_paths = []
        with open(self.path_list, "r") as pl:
            for line in pl:
                image_path = line.strip()
                image_paths.append(image_path)

        self.image_paths = np.array(image_paths)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset = dataset.map(self._parse_function)
        return dataset

    def _parse_function(self, image_path):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.image_size[1], self.image_size[0]],
                                               method=tf.image.ResizeMethod.BICUBIC)
        images = tf.cast(image_resized, tf.float32) / tf.constant(255.0)

        return images, image_path
        # return Image.open(image_path).resize(self.image_size, Image.BICUBIC)	