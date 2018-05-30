import tensorflow as tf
import os
from sklearn.cross_validation import train_test_split # por algun motivo me toma una version vieja de sklearn
import numpy as np
#parametros usados para entrenar la red
learning_rate = 0.5 # tasa de aprendizaje
num_steps = 50 # cantidad de pasos de entrenamiento
batch_size = 512 # cantidad de ejemplos por paso
display_step = 100 # cada cuanto imprime algo por pantalla
# Parametros para la construcción de la red
n_hidden_1 = 512 # numero de neuronas en la capa oculta 1
n_hidden_2 = 512 # numero de neuronas en la capa oculta 2
n_hidden_3 = 512 # numero de neuronas en la capa oculta 3
n_hidden_4 = 512 # numero de neuronas en la capa oculta 4
num_input = 6
num_classes = 2

# Definimos la red neuronal
def neural_net (x_dict):
	x = x_dict['passangers'] 
	layer_1 = tf.layers.dense(x, n_hidden_1)
	layer_2 = tf.layers.dense(layer_1, n_hidden_2)
	layer_3 = tf.layers.dense(layer_2, n_hidden_3)
	layer_4 = tf.layers.dense(layer_3, n_hidden_4)
	out_layer = tf.layers.dense(layer_4, num_classes)
	return out_layer

def model_fn (features, labels, mode):
	logits = neural_net(features)
	# Predicciones
	pred_classes = tf.argmax(logits, axis=1)
	pred_probas = tf.nn.softmax(logits)

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

	# Definimos nuestro error
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op,
	global_step=tf.train.get_global_step())
	
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
	
	estim_specs = tf.estimator.EstimatorSpec(
	mode=mode,
	predictions=pred_classes,
	loss=loss_op,
	train_op=train_op,
	eval_metric_ops={'accuracy': acc_op})
	return estim_specs

def processCsv(input_file, train = True):
	dir_path = os.path.dirname(os.path.realpath(__file__)) + '/Titanic/'
	filename = dir_path + input_file
	input_matrix = []
	correction = -1
	if(train):
		correction = 0
		input_survivors = []
	with open(filename) as inf:
		minAge = 100
		maxAge = 0
		minClass = 1
		maxClass = 3
		sibSPmin = 0
		sibSPmax = 0
		parchmin = 0
		parchmax = 0
    # Skip header
		next(inf)
		for line in inf:
			line_values = line.strip().split(",")
			if line_values[6 + correction] == '':
				# continue
				line_values[6 + correction] = 0
			survival = int(line_values[1])
			line_values[0] = int(line_values[0])
			if(train):
				line_values[1] = int(line_values[1])
			line_values[2 + correction] = int(line_values[2 + correction])
			if line_values[5 + correction] == 'male':
				line_values[5 + correction] = 0
			else:
				line_values[5 + correction] = 1
			line_values[6 + correction] = float(line_values[6 + correction])
			if(line_values[6 + correction] > maxAge):
				maxAge = line_values[6 + correction]
			elif line_values[6 + correction] < minAge:
				minAge = line_values[6 + correction]
			line_values[7 + correction] = int(line_values[7 + correction])
			if(line_values[7 + correction] > sibSPmax):
				sibSPmax = line_values[7 + correction]
			line_values[8 + correction] = int(line_values[8 + correction])
			if(line_values[8 + correction] > parchmax):
				parchmax = line_values[8 + correction]
			del line_values[12 + correction] # embark
			del line_values[11 + correction] # cabin
			del line_values[10 + correction] # fare
			del line_values[9 + correction] # ticket
			del line_values[4 + correction] # name
			del line_values[3 + correction] # last name
			if(train):
				del line_values[1] # survived
			del line_values[0] # id

			input_matrix.append(np.array(line_values))
			if(train):
				input_survivors.append(survival)
		for value in input_matrix:
			value[0] = normalize(value[0], maxClass, minClass)
			value[2] = normalize(value[2], maxAge, minAge)
			value[3] = normalize(value[3], sibSPmax, sibSPmin)
			value[4] = normalize(value[4], parchmax, parchmin)
	if(train):
		return input_matrix, input_survivors
	return input_matrix

def normalize(xi, maxX, minX):
	return ((xi - minX) * 1.0 / (maxX - minX) * 1.0)

train_1, train_2 = processCsv('train.csv', True)
trainX, testX, trainY, testY = train_test_split(train_1, train_2, test_size=0.33, random_state=42)

model = tf.estimator.Estimator(model_fn)

x_dict = {'passangers': np.array(trainX)}
y_value = np.array(trainY)
input_fn = tf.estimator.inputs.numpy_input_fn(
x=x_dict, y=y_value,
batch_size=batch_size, num_epochs=None, shuffle=True)
#Entrenamos el modelo
model.train(input_fn, steps=num_steps)

# Definimos la entrada para evaluar
x_dict = {'passangers': np.array(testX)}
y_value = np.array(testY)
input_fn = tf.estimator.inputs.numpy_input_fn(
x=x_dict, y=y_value,
batch_size=batch_size, shuffle=False)
# Usamos el método 'evaluate'del modelo
e = model.evaluate(input_fn)

print("Precisión en el conjunto de prueba:", e['accuracy'])

testX = processCsv('test.csv', False)

x_dict = {'passangers': np.array(testX)}
input_fn = tf.estimator.inputs.numpy_input_fn(
x=x_dict, num_epochs=1, shuffle=False)

initialP = 892
predictions = model.predict(input_fn)
for prediction in predictions:
	print(str(initialP) + ': {}'.format(prediction))
	initialP += 1