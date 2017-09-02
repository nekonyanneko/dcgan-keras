# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

EPOCH = 30
PNG_CREATE = 20

# 偽物を作る生成機
def generator_model():
	model = Sequential()
	model.add(Dense(input_dim=100, output_dim=1024)) # input_dim >> 乱数
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(128*7*7))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
	model.add(UpSampling2D(size=(2, 2))) # generatorはアップサンプリング
	model.add(Conv2D(64, (5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(1, (5, 5), padding='same'))
	model.add(Activation('tanh'))
	model.summary()
	return model

# 偽物を検知する発見機
def discriminator_model():
	model = Sequential()
	model.add(
			Conv2D(64, (5, 5),
			padding='same',
			input_shape=(28, 28, 1))
			)
	model.add(LeakyReLU(0.2))
	model.add(Conv2D(128, (5, 5), subsample=(2, 2))) # subsampleでダウンサンプリング
	model.add(LeakyReLU(0.2))
	model.add(Flatten())
	model.add(Dense(1024))
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

def combine_images(generated_images):
	num = generated_images.shape[0]
	width = int(math.sqrt(num))
	height = int(math.ceil(float(num)/width))
	shape = generated_images.shape[1:3]
	image = np.zeros((height*shape[0], width*shape[1]),
					 dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		i = int(index/width)
		j = index % width
		image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
			img[:, :, 0]
	return image

def train(BATCH_SIZE):
	####
	# Dataの読み込み
	####
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	print ("データのサイズ")
	print ("	X_train: %s, y_train:%s" % (X_train.shape, y_train.shape))
	print ("	X_test:  %s, y_test: %s" % (X_test.shape, y_test.shape))
	X_train = (X_train.astype(np.float32) - 127.5)/127.5 # ここはイミフ正規化でもなく圧縮でもなく
	X_train = X_train[:, :, :, None] # channelは使用しない為None設定
	X_test  = X_test [:, :, :, None] # channelは使用しない為None設定

	####
	# モデルの準備
	####
	d = discriminator_model() # discriminatorの作成
	g = generator_model()     # generatorの作成
	d_on_g = generator_containing_discriminator(g, d) # discriminatorとgeneratorのCONCAT
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
			noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100)) # 100個の乱数を生成(generatorへ食わすデータ)
			image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE] # 学習画像をBatch分、取得する
			generated_images = g.predict(noise, verbose=0) # noiseを元にgeneratorで偽画像データを生成
			
			####
			# 指定回数の度に、状況をpngを作成
			####
			if index % PNG_CREATE == 0:
				image = combine_images(generated_images)
				image = image*127.5+127.5 # イミフの計算を元に戻す
				Image.fromarray(image.astype(np.uint8)).save(str(epoch)+"_"+str(index)+".png")
			
			####
			# 評価
			####
			X = np.concatenate((image_batch, generated_images)) # 学習画像と生成画像のCONCAT
			y = [1] * BATCH_SIZE + [0] * BATCH_SIZE # 学習画像を1,生成画像を0として正解ラベルの生成
			d_loss = d.train_on_batch(X, y) # discriminatorで評価(偽を検出できるかどうか)
			print("batch %d D_loss : %f" % (index, d_loss))
			noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100)) # 新しく100個の乱数を生成(偽画像生成の評価のため)
			d.trainable = False # 
			g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE) # generatorを評価(ちゃんと偽画像が生成できているかどうか)
			d.trainable = True # 
			print("batch %d G_loss : %f" % (index, g_loss))
			if index % 10 == 9:
				g.save_weights('generator', True)
				d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
	g = generator_model()
	g_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	g.compile(loss='binary_crossentropy', optimizer=g_optim)
	g.load_weights('generator')
	if nice:
		d = discriminator_model()
		d_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		d.compile(loss='binary_crossentropy', optimizer=g_optim)
		d.load_weights('discriminator')
		
		noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
		generated_images = g.predict(noise, verbose=1)
		d_pret = d.predict(generated_images, verbose=1)
		index  = np.arange(0, BATCH_SIZE*20)
		index.resize((BATCH_SIZE*20, 1))
		pre_with_index = list(np.append(d_pret, index, axis=1))
		pre_with_index.sort(key=lambda x: x[0], reverse=True)
		nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
		nice_images = nice_images[:, :, :, None]

		for i in range(BATCH_SIZE):
			idx = int(pre_with_index[i][1])
			nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
		image = combine_images(nice_images)
	else:
		noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
		generated_images = g.predict(noise, verbose=1)
		image = combine_images(generated_images)
	image = image*127.5+127.5
	Image.fromarray(image.astype(np.uint8)).save("generated_image.png")

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
	elif args.mode == "generate":
		generate(BATCH_SIZE=args.batch_size, nice=args.nice)
