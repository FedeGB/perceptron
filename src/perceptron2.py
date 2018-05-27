""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.
Links:
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
# Importamos los datos de ejemplo de la biblioteca MNIST
from tensorflow.examples.tutorials.mnist import input_data
#le decimos que los descargue en /tmp/data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
import tensorflow as tf
#parámetros usados para entrenar la red
learning_rate = 0.1#tasa de aprendizaje
num_steps = 1000
#cantidad de pasos de entrenamiento
batch_size = 128
#cantidad de ejemplos por paso
display_step = 100#cada cuánto imprime algo por pantalla
# Parámetros para la construcción de la red
n_hidden_1 = 256 # número de neuronas en la capa oculta 1
n_hidden_2 = 256 # número de neuronas en la capa oculta 2
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST clases: digitos (0-9 digitos)
# Definimos la red neuronal
def neural_net (x_dict):
	# x_dic es un diccionario con los valores de entrada
	x = x_dict['images'] #en particular vendrán en el campo “image”
	# Conectamos x (la entrada) con la capa oculta 1: Conexión full
	layer_1 = tf.layers.dense(x, n_hidden_1)
	# Conectamos la capa oculta 1, con la capa oculta 2: Conexión full
	layer_2 = tf.layers.dense(layer_1, n_hidden_2)
	# Conectamos la capa oculta 2 con la capa de salida
	out_layer = tf.layers.dense(layer_2, num_classes)
	return out_layer

# Usamos la clase “TF Estimator Template”, para definir cómo será el entrenamiento
def model_fn (features, labels, mode):
	# Llamamos a la función anterior para construir la red
	logits = neural_net(features)
	# Predicciones
	pred_classes = tf.argmax(logits, axis=1)
	pred_probas = tf.nn.softmax(logits)
	# Si es de predicción devolvemos directamente un EstimatorSpec
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

	# Definimos nuestro error
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
	# sparse_softmax_cross_entropy_with_logits : Mide el error de probabilidad
	# en tareas de clasificación discretas en las que las clases son mutuamente
	# excluyentes (cada entrada está en exactamente una clase)
	# reduce_mean : Calcula la media de los elementos a través de las dimensiones
	# de un tensor
	# Definimos un optmizador, que trabaja por el método de descenso por gradiente
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op,
	global_step=tf.train.get_global_step())
	# Definimos cómo se evaluará la precisión del modelo
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
	# Finalmente devolvemos un objeto: “EstimatorSpec”, indicando todo lo que
	# calculamos para el entrenamiento: modo, predicción, error (loss), método de
	# entrenamiento y métricas
	estim_specs = tf.estimator.EstimatorSpec(
	mode=mode,
	predictions=pred_classes,
	loss=loss_op,
	train_op=train_op,
	eval_metric_ops={'accuracy': acc_op})
	return estim_specs


# Construimos un estimador, le decimos que use la función antes definida
model = tf.estimator.Estimator(model_fn)
#pasamos ahora todos los parámetros que necesita la función definida
input_fn = tf.estimator.inputs.numpy_input_fn(
x={'images': mnist.train.images}, y=mnist.train.labels,
batch_size=batch_size, num_epochs=None, shuffle=True)
#Entrenamos el modelo
model.train(input_fn, steps=num_steps)

# Evaluamos el modelo
# Definimos la entrada para evaluar
input_fn = tf.estimator.inputs.numpy_input_fn(
x={'images': mnist.test.images}, y=mnist.test.labels,
batch_size=batch_size, shuffle=False)
# Usamos el método 'evaluate'del modelo
e = model.evaluate(input_fn)
print("Precisión en el conjunto de prueba:", e['accuracy'])