from __future__ import division
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from skimage.exposure import equalize_adapthist
from tensorflow.keras.layers import LSTMCell, Layer, LSTM
from tensorflow import keras as k


def datagens(rescale=1./65536,validation_split=0.2,directory='Breast_Cancer_sinanotar/',target_size=(450, 450),batch_size=32,seed=33,color_mode="grayscale",augmentation=False):
    if augmentation:
        train_datagen=ImageDataGenerator(rescale=rescale,horizontal_flip=True,rotation_range=45, width_shift_range=0.1,
        height_shift_range=0.1,validation_split=validation_split)
    else:
        train_datagen=ImageDataGenerator(rescale=rescale,validation_split=validation_split)
    val_datagen=ImageDataGenerator(rescale=rescale,validation_split=validation_split)
    train_generator = train_datagen.flow_from_directory(
        directory,
        color_mode=color_mode,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=seed,
        subset="training")

    val_generator = val_datagen.flow_from_directory(
        directory,
        color_mode=color_mode,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=seed,
        shuffle=False,
        subset="validation")
    return train_generator, val_generator


from skimage.transform import pyramid_expand
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
import matplotlib.pyplot as plt
import numpy as np
import math

def visualize(imagen,alfas,alpha=0.7,h=450,w=450):
    #Falta mirar como ha hecho el resize
    if type(imagen)==str:
        im=load_img(imagen,grayscale=True)
        im=im.resize((h,w))
        ima=img_to_array(im)/255.
        ima=ima.reshape(1,h,w,1)
        
    elif type(imagen)==np.ndarray:
        im=array_to_img(imagen)
        ima=imagen.reshape(1,h,w,1)
    #Si es tipo imagen
    else:
        im=imagen
        im=im.resize((h,w))
        ima=img_to_array(im)/255.
        ima=ima.reshape(1,h,w,1)
    reshape_measure=int(math.sqrt(alfas.shape[1]))
    alfas=alfas.reshape((reshape_measure,reshape_measure))
    expanded_alfas=pyramid_expand(alfas,upscale=h/reshape_measure,sigma=20)
    fig,((ax1,ax2))=plt.subplots(2,1)
    ax1.imshow(im,cmap="gray")
    ax2.imshow(expanded_alfas,cmap="gray")
    plt.show()
    plt.imshow(im,cmap="gray")
    plt.imshow(expanded_alfas, alpha=alpha,cmap="gray")
    return expanded_alfas,im

def reshape_matmul(M3D,M2D):
    return tf.einsum('ijk,kl->ijl', M3D, M2D)

import cv2

#def clahe(data,clip_limit=0.03,bilateral=True):
#    if bilateral:
#        for i in range(len(data)):
#            for j in range(len(data[i][0])):    
#                data[i][0][j]=tf.expand_dims(equalize_adapthist(cv2.bilateralFilter(data[i][0][j], 15, 75, 75),clip_limit=clip_limit),axis=2)
#    else:
#        for i in range(len(data)):
#            for j in range(len(data[i][0])):    
#                data[i][0][j]=tf.expand_dims(equalize_adapthist(tf.squeeze(data[i][0][j]),clip_limit=clip_limit),axis=2)
#    return data

#def clahe(data,clip_limit=0.03):
#    for i in range(len(data)):
#        for j in range(len(data[i][0])):    
#            data[i][0][j]=tf.expand_dims(equalize_adapthist(tf.squeeze(data[i][0][j]),clip_limit=clip_limit),axis=2)
#    return data

def clahe(data,clip_limit=0.03):
    for i in range(len(data)):
        images=data[i][0]
        target=data[i][1]
        for j in range(len(images)):
            if (i==0 and j==0):
                x=tf.expand_dims(equalize_adapthist(tf.squeeze(data[i][0][j]),clip_limit=clip_limit),axis=2)
            else:
                x=np.append(x,tf.expand_dims(equalize_adapthist(tf.squeeze(data[i][0][j]),clip_limit=clip_limit),axis=2),axis=0)
        if i==0:
            y=target
        else:
            y=np.append(y,target,axis=0)
    a,b,c=data[0][0][0].shape
    x=tf.reshape(x,[-1,a,b,c])
    #x=x.array()
    #y=y.array()
    #y=tf.reshape(y,[-1,3])
    return x,y

def bilateral_clahe(data,clip_limit=0.03):
    for i in range(len(data)):
        images=data[i][0]
        target=data[i][1]
        for j in range(len(images)):
            if (i==0 and j==0):
                x=tf.expand_dims(equalize_adapthist(cv2.bilateralFilter(data[i][0][j], 5, 10, 10),clip_limit=clip_limit),axis=2)
            else:
                x=np.append(x,tf.expand_dims(equalize_adapthist(cv2.bilateralFilter(data[i][0][j], 5, 10, 10),clip_limit=clip_limit),axis=2),axis=0)
        if i==0:
            y=target
        else:
            y=np.append(y,target,axis=0)
    a,b,c=data[0][0][0].shape
    x=tf.reshape(x,[-1,a,b,c])
    #x=x.array()
    #y=y.array()
    #y=tf.reshape(y,[-1,3])
    return x,y


import cv2
import numpy as np
import math

def srbf(image_matrix, window_length=5,sigma_color=25,sigma_space=20):
    mask_image_matrix = np.zeros(
        (image_matrix.shape[0], image_matrix.shape[1]))
    image_matrix = image_matrix.astype(np.float64)#transfer the image_matrix to type int32，for uint cann't represent the negative number afterward
    
    def limit(x):
        x = 0 if x < 0 else x
        x = 1 if x > 1 else x
        return x
    limit_ufun = np.vectorize(limit, otypes=[np.double])
    def alfa(image_matrix):
        #Number of pixels in the image
        N=len(image_matrix)**2
        return math.sqrt(np.sum(np.square(image_matrix))/(2*N))
        
    def look_for_gaussion_table(delta):
        return delta_gaussion_dict[delta]
    def generate_bilateral_filter_distance_matrix(window_length,sigma):
        distance_matrix = np.zeros((window_length,window_length,1))
        left_bias = int(math.floor(-(window_length - 1) / 2))
        right_bias = int(math.floor((window_length - 1) / 2))
        for i in range(left_bias,right_bias+1):
            for j in range(left_bias,right_bias+1):
                #Aquí debo introducir el primer cambio.
                distance_matrix[i-left_bias][j-left_bias] = math.exp(-(i**2+j**2)/(2*(sigma**2))) 
        return distance_matrix
    #Segundo cambio, cambiar delta_gaussion_dict. No se puede emplear diccionario, ya que depende del valor de alfa.
    delta_gaussion_dict = {i: math.exp(-i ** 2 / (2 *(sigma_color**2))) for i in range(256)}
    look_for_gaussion_table_ufun = np.vectorize(look_for_gaussion_table, otypes=[np.float64])#to accelerate the process of get the gaussion matrix about color.key:color difference，value:gaussion weight
    bilateral_filter_distance_matrix = generate_bilateral_filter_distance_matrix(window_length,sigma_space)#get the gaussion weight about distance directly

    margin = int(window_length / 2)
    left_bias = math.floor(-(window_length - 1) / 2)
    right_bias = math.floor((window_length - 1) / 2)
    filter_image_matrix = image_matrix.astype(np.float64)

    for i in range(0 + margin, image_matrix.shape[0] - margin):
        for j in range(0 + margin, image_matrix.shape[1] - margin):
            if mask_image_matrix[i][j]==0:
                filter_input = image_matrix[i + left_bias:i + right_bias + 1,
                               j + left_bias:j + right_bias + 1]#get the input window
                #Tal vez las aproximaciones están pensadas para valores de intensidad entre 0 y 1.
                bilateral_filter_value_matrix=(image_matrix[i][j]/(filter_input**2+1))*np.exp(-(image_matrix[i][j]**2)/(2*filter_input**2+1))
#                 bilateral_filter_value_matrix = look_for_gaussion_table_ufun(np.abs(filter_input-image_matrix[i][j]))#get the gaussion weight about color
                bilateral_filter_matrix = np.multiply(bilateral_filter_value_matrix, bilateral_filter_distance_matrix)#multiply color gaussion weight  by distane gaussion weight to get the no-norm weigth matrix
                bilateral_filter_matrix = bilateral_filter_matrix/np.sum(bilateral_filter_matrix,keepdims=False,axis=(0,1))#normalize the weigth matrix
                filter_output = np.sum(np.multiply(bilateral_filter_matrix,filter_input),axis=(0,1)) #multiply the input window by the weigth matrix，then get the sum of channels seperately
                filter_image_matrix[i][j] = filter_output
    filter_image_matrix = limit_ufun(filter_image_matrix)#limit the range
    return filter_image_matrix


def srbf_generator(data):
    for i in range(len(data)):
        images=data[i][0]
        target=data[i][1]
        for j in range(len(images)):
            if (i==0 and j==0):
                x=srbf(data[i][0][j], 5, 25, 20)
            else:
                x=np.append(x,srbf(data[i][0][j], 5, 25, 20),axis=0)
        if i==0:
            y=target
        else:
            y=np.append(y,target,axis=0)
    a,b,c=data[0][0][0].shape
    x=tf.reshape(x,[-1,a,b,c])
    #x=x.array()
    #y=y.array()
    #y=tf.reshape(y,[-1,3])
    return x,y



class LSTM_Attention(Layer):
    def __init__(self, dim_feature=[196,512],dim_word=46,dim_hidden=256,
                 dropout=False, L2Attention=False, Gatted=False, L2dim=20, name="LSTM_Attention",**kwargs):
        #Inicializa los features que no dependen de la entrada
        self.L2Attention=L2Attention
        self.Gatted=Gatted
        self.L2dim=L2dim
        self.H=dim_hidden
        self.dropout=dropout
        self.weight_initializer = tf.keras.initializers.glorot_normal
        self.const_initializer = tf.keras.initializers.Zeros()
        super(LSTM_Attention, self).__init__(name=name)
        super(LSTM_Attention, self).__init__(**kwargs)
        
    def build(self,input_shape):
        Features_shape=input_shape[0]
        F_2=Features_shape[2]
        #Inicia los features que dependen de la entrada
        
        #Iniciar primer estado oculto y memoria. Se siquen los pasos establecidos en 
        #"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
        
        #features_mean = tf.reduce_mean(features, 1)

        self.w_h = self.add_weight('w_h', [F_2, self.H], initializer=self.weight_initializer)
        self.b_h = self.add_weight('b_h', [self.H], initializer=self.const_initializer)
        #h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)
        

        self.w_c = self.add_weight('w_c', [F_2, self.H], initializer=self.weight_initializer)
        self.b_c = self.add_weight('b_c', [self.H], initializer=self.const_initializer)
        #c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
        
        #Iniciar pesos primera Atención
        if self.L2Attention or self.Gatted:
            self.w_a_h_1= self.add_weight("w_a_h_1", shape=[self.H,self.L2dim])
            self.w_a_1= self.add_weight("w_a_1", shape=[F_2,self.L2dim])
            self.b_a_1=self.add_weight("b_a_1",shape=[self.L2dim])
            
            self.w_a_2= self.add_weight("w_a_2", shape=[self.L2dim,1])
            self.b_a_2=self.add_weight("b_a_2",shape=[1])
            
            if self.Gatted:
                self.w_a_h_g= self.add_weight("w_a_h_g", shape=[self.H,self.L2dim])
                self.w_a_g= self.add_weight("w_a_g", shape=[F_2,self.L2dim])
                self.b_a_g=self.add_weight("b_a_g",shape=[self.L2dim])
             
        else:
            self.w_a_h= self.add_weight("w_a_h", shape=[self.H,1])
            self.w_a= self.add_weight("w_a", shape=[F_2,1])
            self.b_a=self.add_weight("b_a",shape=[1])
        
        #Iniciar LSTM
        self.lstm_cell = LSTMCell(self.H,activation=tf.keras.activations.tanh, recurrent_activation=tf.keras.activations.sigmoid)  #From keras.    
        #self.lstm_cell = LSTM(self.H,activation=tf.keras.activations.tanh, recurrent_activation=tf.keras.activations.sigmoid)  #From keras.    
        #super(LSTM_Attention, self).build(input_shape)
    
    def call(self, inputs,training=None):
        #Crear primer estado oculto y memoria
        tf.random.set_seed(1)
        input_data=inputs[0]
        word_inputs=inputs[1]
        T=word_inputs.shape[1]
        alfas=[]
        hiddens=[]
        memory=[]
        features_mean=tf.reduce_mean(input_data,1)
        if len(inputs)==3:
            h=tf.cond(tf.equal(tf.reduce_sum(inputs[2][0]),tf.constant(0,dtype=tf.float32)),
                      lambda: tf.keras.activations.tanh(tf.matmul(features_mean, self.w_h) + self.b_h), lambda: inputs[2][0])
            c=tf.cond(tf.equal(tf.reduce_sum(inputs[2][1]),tf.constant(0,dtype=tf.float32)),
                      lambda: tf.keras.activations.tanh(tf.matmul(features_mean, self.w_c) + self.b_c), lambda: inputs[2][1])
        else:
            h=tf.keras.activations.tanh(tf.matmul(features_mean, self.w_h) + self.b_h)
            c=tf.keras.activations.tanh(tf.matmul(features_mean, self.w_c) + self.b_c)
        #Primear atención
        #lstm
        for t in range (T):#FALTA HACER Gatted
            if self.L2Attention or self.Gatted:
                Attention1=tf.keras.activations.tanh(tf.expand_dims(tf.matmul(h,self.w_a_h_1),axis=1)+reshape_matmul(input_data,self.w_a_1)+self.b_a_1) #(B,F_1,L2dim)
                
                if self.Gatted:
                    Attention2=k.activations.sigmoid(tf.expand_dims(tf.matmul(h,self.w_a_h_g),axis=1)+reshape_matmul(input_data,self.w_a_g)+self.b_a_g) #(B,F_1,L2dim)
                    Attention1=tf.math.multiply(Attention1,Attention2)
                Attention=reshape_matmul(Attention1,self.w_a_2)+self.b_a_2 #(B,F_1,1)
            else:
                Attention=tf.keras.activations.tanh(tf.expand_dims(tf.matmul(h,self.w_a_h),axis=1)+reshape_matmul(input_data,self.w_a)+self.b_a) #(B,F_1,1)
            alfa=tf.keras.activations.softmax(tf.reshape(Attention,[-1,Attention.shape[1]])) #(B,F_1)
            context = tf.reduce_sum(tf.math.multiply(input_data, tf.expand_dims(alfa, axis=2)), 1) #(B,F_2)
            alfas.append(alfa)
            _, (h, c) = self.lstm_cell(inputs=tf.concat( [word_inputs[:,t,:], context],1), states=[h, c])
            
            if self.dropout and training:
                
                hiddens.append(tf.nn.dropout(h,0.2))
            else:
                hiddens.append(h)
            memory.append(c)
        alfas=tf.transpose(tf.stack(alfas), (1, 0, 2))
        hiddens=tf.transpose(tf.stack(hiddens), (1, 0, 2))
            #Volver a crear context                                                                                       
        return hiddens,memory,alfas
    def get_config(self):
        config = super(LSTM_Attention, self).get_config()
        #config.update({"H": self.H})
        return config