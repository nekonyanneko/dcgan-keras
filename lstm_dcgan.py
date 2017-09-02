# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, UpSampling2D
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Reshape, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
from numpy import *
from PIL import Image
import argparse
import math

EPOCH = 500
PNG_CREATE = 20

length_of_sequences  = 50 # 入力データの次元
length_of_random_num = 100 # 乱数
in_out_neurons = 50 # 出力データの次元
hidden_neurons = 300 # 中間層の数
TRAIN_PATH = './R/data_lstm.txt'
TEST_PATH = './R/data_lstm_test.txt'
train_data = loadtxt(TRAIN_PATH)
test_data = loadtxt(TEST_PATH)
print 'train_input_data shape: ',train_data.shape
print 'test_input_data shape: ',test_data.shape

# 偽物を作る生成機
def generator_model():
	model = Sequential()
	'''
        model.add(Dense(input_dim=length_of_random_num, output_dim=length_of_sequences)) # input_dim >> 乱数
        #model.add(BatchNormalization())
        model.add(Activation('elu'))
	model.add(Reshape((in_out_neurons, 1)))
	model.add(GRU(hidden_neurons, batch_input_shape=(None, length_of_sequences, 1), return_sequences=True))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	model.add(GRU(hidden_neurons/2, input_shape=(hidden_neurons/2, 1), return_sequences=True))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	model.add(GRU(hidden_neurons/2, input_shape=(hidden_neurons/2, 1), return_sequences=False))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	model.add(Dense(in_out_neurons))
	model.add(Activation("linear"))
	model.add(Reshape((in_out_neurons, 1)))
	'''
	model.add(Dense(input_dim=length_of_random_num, output_dim=1024)) # input_dim >> 乱数
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(64*5))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Reshape((5, 64), input_shape=(64*5,)))
	model.add(UpSampling1D(size=2)) # generatorはアップサンプリング
	model.add(Conv1D(32, 5, padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(UpSampling1D(size=2))
	model.add(Conv1D(16, 5, padding='same'))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(in_out_neurons))
	model.add(Activation("tanh"))
	model.add(Reshape((in_out_neurons, 1)))
	
	print("Create Generator Model")
	model.summary()
	return model

# 偽物を検知する発見機
def discriminator_model():
	model = Sequential()
	'''
        model.add(GRU(hidden_neurons, batch_input_shape=(None, length_of_sequences, 1), return_sequences=True))
        model.add(GRU(hidden_neurons/2, input_shape=(hidden_neurons/2, 1), return_sequences=True))
        model.add(GRU(hidden_neurons/2, input_shape=(hidden_neurons/2, 1), return_sequences=False))
        model.add(Dense(in_out_neurons))
        model.add(Activation("linear"))
	#model.add(LeakyReLU(0.2))
	#model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
	print("Create Discriminator Model")
	'''
        model.add(Conv1D(32, 2, padding='same', batch_input_shape=(None, length_of_sequences, 1)))
	model.add(LeakyReLU(0.2))
	model.add(Conv1D(64, 4))
	model.add(LeakyReLU(0.2))
	model.add(Flatten())
	model.add(Dense(256))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	
	model.summary()
	return model

def generator_containing_discriminator(g, d):
	model = Sequential()
	model.add(g)
	d.trainable = False
	model.add(d)
	model.summary()
	return model

def train(BATCH_SIZE):
	####
	# Dataの読み込み
	####
	X_train = []
	y_train = []
	X_train = train_data[:,0:length_of_sequences]
	y_train = train_data[:,length_of_sequences:length_of_sequences+in_out_neurons]
	
	X_train = (X_train.astype(np.float32) - 100.0)/100.0
	
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	
	print X_train.shape
	print y_train.shape

	####
	# モデルの準備
	####
	d = discriminator_model() # discriminatorの作成
	g = generator_model()     # generatorの作成
	d_on_g  = generator_containing_discriminator(g, d) # discriminatorとgeneratorのCONCAT
	d_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # discriminatorの最適化パラメータ設定
	g_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # generatorの最適化パラメータ設定
	d_on_g_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # d_gの最適化パラメータ設定
	g.compile     (loss='binary_crossentropy', optimizer=g_optim)
	d_on_g.compile(loss='binary_crossentropy', optimizer=d_on_g_optim)
	d.trainable = True # 
	d.compile     (loss='binary_crossentropy', optimizer=d_optim)
	
	####
	# epoch数分のループ
	####
	for epoch in range(EPOCH):
		print("Epoch is", epoch)
		print("Number of batches", int( X_train.shape[0] / BATCH_SIZE ))
		
		####
		# batchのループ
		####
		for index in range(int( X_train.shape[0] / BATCH_SIZE )):
			noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, length_of_random_num)) # 乱数を生成(generatorへ食わすデータ)
			image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE] # 学習画像をBatch分、取得する
			generated_images = g.predict(noise, verbose=0) # noiseを元にgeneratorで偽画像データを生成
			#print image_batch[1]
			print generated_images[0]
			
			####
			# 評価
			####
			X = np.concatenate((image_batch, generated_images)) # 学習画像と生成画像のCONCAT
			y = [1] * BATCH_SIZE + [0] * BATCH_SIZE # 学習画像を1,生成画像を0として正解ラベルの生成
			d_loss = d.train_on_batch(X, y) # discriminatorで評価(偽を検出できるかどうか)
			#print d.predict(X)
			print("batch %d D_loss : %f" % (index, d_loss))
			noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100)) # 新しく100個の乱数を生成(偽画像生成の評価のため)
			d.trainable = False # 
			g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE) # generatorを評価(ちゃんと偽画像が生成できているかどうか)
			d.trainable = True # 
			print("batch %d G_loss : %f" % (index, g_loss))
			
			if index % 10 == 9:
				g.save_weights('generator', True)
				d.save_weights('discriminator', True)
				g.save('generator1')
				d.save('discriminator1')
	g.save('generator1')
	d.save('discriminator1')

def estimate():
	####
        # Dataの読み込み
        ####
        X_test = []
        y_test = []
        X_test = test_data[:,0:length_of_sequences]
        y_test = test_data[:,length_of_sequences:length_of_sequences+in_out_neurons]
	
	X_test = (X_test.astype(np.float32) - 100.0)/100.0
	y_test = (y_test.astype(np.float32) - 100.0)/100.0
	
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
	
        print X_test.shape
        print y_test.shape
	
	model = load_model('discriminator_max')
	predicted = model.predict(X_test)
	#print predicted
	predicted = model.predict(y_test)
	print predicted
	return

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--nice", dest="nice", action="store_true")
	parser.set_defaults(nice=False)
	args = parser.parse_args()
	print args
	return args

if __name__ == "__main__":
	args = get_args()
	if args.mode == "train":
		train(BATCH_SIZE=args.batch_size)
	elif args.mode == "estimate":
		estimate()
