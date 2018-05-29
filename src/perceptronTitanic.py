import tensorflow as tf
import os
from sklearn.cross_validation import train_test_split # por algun motivo me toma una version vieja de sklearn
import numpy as np
#parametros usados para entrenar la red
learning_rate = 0.02 # tasa de aprendizaje
num_steps = 2000 # cantidad de pasos de entrenamiento
batch_size = 256 # cantidad de ejemplos por paso
display_step = 100 # cada cuanto imprime algo por pantalla
# Parametros para la construcción de la red
n_hidden_1 = 512 # numero de neuronas en la capa oculta 1
n_hidden_2 = 512 # numero de neuronas en la capa oculta 2
num_input = 6
num_classes = 2

# Definimos la red neuronal
def neural_net (x_dict):
	x = x_dict['passangers'] 
	layer_1 = tf.layers.dense(x, n_hidden_1)
	layer_2 = tf.layers.dense(layer_1, n_hidden_2)
	out_layer = tf.layers.dense(layer_2, num_classes)
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

def processTrainCsv(input_file):
	dir_path = os.path.dirname(os.path.realpath(__file__)) + '/Titanic/'
	filename = dir_path + input_file
	input_matrix = []
	input_survivors = []
	with open(filename) as inf:
        # Skip header
				next(inf)
				for line in inf:
					line_values = line.strip().split(",")
					if line_values[6] == '':
						continue
					survival = int(line_values[1])
					line_values[0] = int(line_values[0])
					line_values[1] = int(line_values[1])
					line_values[2] = int(line_values[2])
					if line_values[5] == 'male':
						line_values[5] = 0
					else:
						line_values[5] = 1
					line_values[6] = float(line_values[6])
					line_values[7] = int(line_values[7])
					line_values[8] = int(line_values[8])
					line_values[10] = float(line_values[10])
					del line_values[12] #embark
					del line_values[11] # cabin
					del line_values[9] # ticket
					del line_values[4] # name
					del line_values[3] # last name
					del line_values[1] # survived
					del line_values[0] # id
					input_matrix.append(np.array(line_values))
					input_survivors.append(survival)
	return input_matrix, input_survivors

train_1, train_2 = processTrainCsv('train.csv')
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