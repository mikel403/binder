from __future__ import division
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.layers import Layer, LSTMCell
def reshape_matmul(M3D,M2D):
    return tf.einsum('ijk,kl->ijl', M3D, M2D)

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


class Horizontal_flip(Layer):
    def __init__(self,**kwargs):
        super(Horizontal_flip, self).__init__()
        super(Horizontal_flip, self).__init__(**kwargs)
    def call(self,inputs,training=None):
        if not training:
            output=inputs
        else:
            output=tf.image.random_flip_left_right(inputs)
        return output

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