# -*- coding: utf-8 -*-

import os
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Conv2D, BatchNormalization, Conv2DTranspose, Concatenate, LayerNormalization, PReLU, Activation, ReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

import soundfile as sf
import librosa
from random import seed
import numpy as np
import tqdm

from utils import reshape, transpose, ParallelModelCheckpoints
from data_loader_wham import *

seed(42)

class FULLSUB_model():
    '''
    Class to create and train the fullsubnet model
    '''
    
    def __init__(self, batch_size = 1,
                       length_in_s = 5,
                       fs = 16000,
                       norm = 'iLN',
                       numUnits = 16,
                       numDP = 1,
                       block_len = 512,
                       block_shift = 256,
                       max_epochs = 200,
                       lr = 1e-3):

        # defining default cost function
        self.cost_function = self.sisnr_cost
        self.model = None
        # defining default parameters
        self.fs = fs
        self.length_in_s = length_in_s
        self.batch_size = batch_size
        # number of the hidden layer size in the LSTM
        self.numUnits = numUnits
        # number of the DPRNN modules
        self.numDP = numDP
        # frame length and hop length in STFT
        self.block_len = block_len
        self.block_shift = block_shift
        self.lr = lr
        self.max_epochs = max_epochs
        # window for STFT: sine win
        win = np.sin(np.arange(.5,self.block_len-.5+1)/self.block_len*np.pi)
        self.win = tf.constant(win,dtype = 'float32')

        self.L = (16000*length_in_s-self.block_len)//self.block_shift + 1
        self.eps = 1e-9
        self.multi_gpu = False
        # iLN for instant Layer norm and BN for Batch norm
        self.input_norm = norm
        
    @staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        Static Method defining the cost function. 
        The negative signal to noise ratio is calculated here. The loss is 
        always calculated over the last dimension. 
        '''
        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True) + 1e-8)
        # using some more lines, because TF has no log10
        num = tf.math.log(snr + 1e-8) 
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10*(num / (denom))

        return loss
    
    @staticmethod
    def sisnr_cost(s_hat, s):
        '''
        Static Method defining the cost function. 
        The negative signal to noise ratio is calculated here. The loss is 
        always calculated over the last dimension. 
        '''
        def norm(x):
            return tf.reduce_sum(x**2, axis=-1, keepdims=True)

        s_target = tf.reduce_sum(
            s_hat * s, axis=-1, keepdims=True) * s / norm(s)
        upp = norm(s_target)
        low = norm(s_hat - s_target)
  
        return -10 * tf.log(upp /low) / tf.log(10.0)  
    
    def spectrum_loss(self, y_pred, y_true):
        '''
        spectrum MSE loss 
        '''
        pred_real,pred_imag = self.stftLayer(y_pred, mode='real_imag')
        pred_mag = tf.sqrt(pred_real**2 + pred_imag**2 + 1e-8)
        
        true_real,true_imag = self.stftLayer(y_true, mode='real_imag')
        true_mag = tf.sqrt(true_real**2 + true_imag**2 + 1e-8)
        
        loss_real = tf.reduce_mean((pred_real - true_real)**2,)
        loss_imag = tf.reduce_mean((pred_imag - true_imag)**2,)
        loss_mag = tf.reduce_mean((pred_mag - true_mag)**2,) 
        
        return loss_real + loss_imag + loss_mag
        
    def spectrum_loss_phasen(self, s_hat,s,gamma = 0.3):
        
        true_real,true_imag = self.stftLayer(s, mode='real_imag')
        hat_real,hat_imag = self.stftLayer(s_hat, mode='real_imag')

        true_mag = tf.sqrt(true_real**2 + true_imag**2+1e-9)
        hat_mag = tf.sqrt(hat_real**2 + hat_imag**2+1e-9)

        true_real_cprs = (true_real / true_mag )*true_mag**gamma
        true_imag_cprs = (true_imag / true_mag )*true_mag**gamma
        hat_real_cprs = (hat_real / hat_mag )* hat_mag**gamma
        hat_imag_cprs = (hat_imag / hat_mag )* hat_mag**gamma

        loss_mag = tf.reduce_mean((hat_mag**gamma - true_mag**gamma)**2,)         
        loss_real = tf.reduce_mean((hat_real_cprs - true_real_cprs)**2,)
        loss_imag = tf.reduce_mean((hat_imag_cprs - true_imag_cprs)**2,)

        return 0.7 * loss_mag + 0.3 * ( loss_imag + loss_real )
    
    def metricsWrapper(self):
        '''
        A wrapper function which returns the metrics used during training
        '''
        return [self.sisnr_cost]        
    
    @staticmethod    
    def sisnr_cost1(s, s_hat):
        def norm(x):
            return tf.reduce_sum(x**2, axis=-1, keepdims=True)
            
        s_target = tf.reduce_sum(
            s_hat * s, axis=-1, keepdims=True) / tf.sqrt(norm(s)) / tf.sqrt(norm(s_hat))
        
        upp = 1 + s_target
        low = 1 - s_target
  
        return -10 * tf.log(upp /low) / tf.log(10.0)        
    
    def lossWrapper2(self):
        '''
        CRM loss like fullsubnet
        '''
        def compress_cIRM(mask, K=10, C=0.1):
            
            #mask = tf.where(mask <= tf.constant(-100.), tf.constant(-100., shape = [8,311,257,2]), mask)
            mask1 = K * (1. - tf.exp(-C * mask)) / (1. + tf.exp(-C * mask))
            
            return mask1
        
        def lossFunction(y_true,y_pred):
            
            #y_true = tf.truediv(y_true, self.batch_gain + 1e-9)
            #y_pred = tf.truediv(y_pred, self.batch_gain + 1e-9)
            
            true_real, true_imag = self.stftLayer(y_true, mode='real_imag') # B T F
                       
            denominator = self.noisy_real[:,:,:,0]**2 + self.noisy_imag[:,:,:,0]**2 + 1e-9
            mask_real = (self.noisy_real[:,:,:,0] * true_real + self.noisy_imag[:,:,:,0] * true_imag) / denominator
            mask_imag = (self.noisy_real[:,:,:,0] * true_imag - self.noisy_imag[:,:,:,0] * true_real) / denominator
            complex_ratio_mask = tf.stack([mask_real, mask_imag], axis=-1)
            
            #pred_mask_real = self.output_mask[:,:,:,0]
            
            #pred_mask_imag = self.output_mask[:,:,:,1]
            complex_ratio_mask_comp = compress_cIRM(complex_ratio_mask, K=10, C=0.1)
            
            
            #loss = tf.reduce_mean((pred_mask_real - mask_real)**2,) + tf.reduce_mean((pred_mask_imag - mask_imag)**2,)
            
            pred_mask = self.output_mask 
            loss = tf.reduce_mean((pred_mask - complex_ratio_mask)**2,)
            
            return loss
            
            
        return lossFunction
            
    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''
        def lossFunction(y_true,y_pred):
            
            y_true = tf.truediv(y_true, self.batch_gain + 1e-9)
            y_pred = tf.truediv(y_pred, self.batch_gain + 1e-9)
            # calculating loss and squeezing single dimensions away
            loss = tf.squeeze(self.cost_function(y_pred,y_true)) 
            #mag_loss = tf.log(self.spectrum_loss(y_pred,y_true) + 1e-8)
            #mag_loss = tf.log(self.spectrum_loss_phasen(y_pred,y_true) + 1e-8)
            # calculate mean over batches
            #loss = tf.reduce_mean(loss)
            return loss 
        
        return lossFunction
    
    def lossWrapper1(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''
        def spectrum_loss_SD(s_hat, s, c = 0.3, Lam = 0.1):
            # The complex compressed spectrum MSE loss
            s = tf.truediv(s,self.batch_gain + 1e-9)
            s_hat= tf.truediv(s_hat,self.batch_gain + 1e-9)

            true_real,true_imag = self.stftLayer(s, mode='real_imag')
            hat_real,hat_imag = self.stftLayer(s_hat, mode='real_imag')
            
            true_mag = tf.sqrt(true_real**2 + true_imag**2 + 1e-9)
            hat_mag = tf.sqrt(hat_real**2 + hat_imag**2 + 1e-9)

            true_real_cprs = (true_real / true_mag )*true_mag**c
            true_imag_cprs = (true_imag / true_mag )*true_mag**c
            hat_real_cprs = (hat_real / hat_mag )* hat_mag**c
            hat_imag_cprs = (hat_imag / hat_mag )* hat_mag**c

            loss_mag = tf.reduce_mean((hat_mag**c - true_mag**c)**2,)         
            loss_real = tf.reduce_mean((hat_real_cprs - true_real_cprs)**2,)
            loss_imag = tf.reduce_mean((hat_imag_cprs - true_imag_cprs)**2,)
            
            return (1 - Lam) * loss_mag + Lam * ( loss_imag + loss_real )

        return spectrum_loss_SD    
       
    
    '''
    In the following some helper layers are defined.
    '''  
    def seg2frame(self, x):
        '''
        split signal x to frames
        '''
        frames = tf.signal.frame(x, self.block_len, self.block_shift)
        if self.win is not None:
            frames = self.win*frames
        return frames
    
    def stftLayer(self, x, mode ='mag_pha'):
        '''
        Method for an STFT helper layer used with a Lambda layer
        mode: 'mag_pha'   return magnitude and phase spectrogram
              'real_imag' return real and imaginary parts
        '''
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.block_len, self.block_shift)
        #print(frames.shape)
        if self.win is not None:
            frames = self.win*frames
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        #print(stft_dat.shape)
        # calculating magnitude and phase from the complex signal
        output_list = []
        if mode == 'mag_pha':
            mag = tf.math.abs(stft_dat)
            phase = tf.math.angle(stft_dat)
            output_list = [mag, phase]
        elif mode == 'real_imag':
            real = tf.math.real(stft_dat)
            imag = tf.math.imag(stft_dat)
            output_list = [real, imag]
            
        # returning magnitude and phase as list
        return output_list
    
    def fftLayer(self, x):
        '''
        Method for an fft helper layer used with a Lambda layer.
        The layer calculates the rFFT on the last dimension and returns
        the magnitude and phase of the STFT.
        '''
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(x)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]


    def ifftLayer(self, x,mode = 'mag_pha'):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information. 
        As input x a list with [mag,phase] is required.
        '''
        if mode == 'mag_pha':
        # calculating the complex representation
            s1_stft = (tf.cast(x[0], tf.complex64) * 
                        tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        elif mode == 'real_imag':
            s1_stft = tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)  
    
    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''
        # calculating and returning the reconstructed waveform
        '''
        if self.move_dc:
            x = x - tf.expand_dims(tf.reduce_mean(x,axis = -1),2)
        '''
        return tf.signal.overlap_and_add(x, self.block_shift)              
     
    def mk_mask_complex(self, x):
        '''
        complex ratio mask
        '''
        [noisy_real,noisy_imag,mask] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        mask_real = mask[:,:,:,0]
        mask_imag = mask[:,:,:,1]
        
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real
        
        return [enh_real,enh_imag]
    
    def mk_mask_mag(self, x):
        '''
        magnitude mask
        '''
        [noisy_real,noisy_imag,mag_mask] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        enh_mag_real = noisy_real * mag_mask
        enh_mag_imag = noisy_imag * mag_mask
        return [enh_mag_real,enh_mag_imag]
    
    def mk_mask_pha(self, x):
        '''
        phase mask
        '''
        [enh_mag_real,enh_mag_imag,pha_cos,pha_sin] = x
        
        enh_real = enh_mag_real * pha_cos - enh_mag_imag * pha_sin
        enh_imag = enh_mag_real * pha_sin + enh_mag_imag * pha_cos
        
        return [enh_real,enh_imag]
    
    def mk_mask_mag_pha(self, x):
        
        [noisy_real,noisy_imag,mag_mask,pha_cos,pha_sin] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        enh_mag_real = noisy_real * mag_mask
        enh_mag_imag = noisy_imag * mag_mask
        
        enh_real = enh_mag_real * pha_cos - enh_mag_imag * pha_sin
        enh_imag = enh_mag_real * pha_sin + enh_mag_imag * pha_cos
        
        return [enh_real,enh_imag]
    
        
    def build_FULLSUB_model(self, name = 'model0'):

        # input layer for time signal
        time_dat = Input(batch_shape=(self.batch_size, None))
        self.batch_gain = Input(batch_shape=(self.batch_size, 1))
        
        # calculate STFT
        real,imag = Lambda(self.stftLayer,arguments = {'mode':'real_imag'})(time_dat)
        
        # normalizing log magnitude stfts to get more robust against level variations
        real = Lambda(reshape,arguments={'axis':[self.batch_size,-1,self.block_len // 2 + 1,1]})(real)
        imag = Lambda(reshape,arguments={'axis':[self.batch_size,-1,self.block_len // 2 + 1,1]})(imag)
        #print(real)
        
        self.noisy_real = real
        self.noisy_imag = imag
        
        input_mag = tf.math.sqrt(real**2 + imag**2 +1e-9)
        
        
        
        input_complex_spec = input_mag # B T F C
        
        input_complex_spec = Lambda(transpose, arguments={'axis':[0,3,1,2]})(input_complex_spec)
        input_complex_spec = Lambda(reshape, arguments={'axis':[self.batch_size, -1, 257]})(input_complex_spec)
        #print(input_complex_spec) # 8,?,257
        
        
        # for tflite
        # fb_gru_out_1 = keras.layers.GRU(units=32, return_sequences=True, recurrent_activation = 'sigmoid',reset_after = False)(input_complex_spec)
        # fb_gru_out_2 = keras.layers.GRU(units=32, return_sequences=True, recurrent_activation = 'sigmoid',reset_after = False)(fb_gru_out_1)
        
        # for training
        fb_gru_out_1 = keras.layers.CuDNNGRU(units=32, return_sequences=True)(input_complex_spec)
        fb_gru_out_2 = keras.layers.CuDNNGRU(units=32, return_sequences=True)(fb_gru_out_1)
        
        fb_gru_out_fc = keras.layers.Dense(units = 257,)(fb_gru_out_2)
        #print(fb_gru_out_fc) # 8, ?, 257
        
        fb_gru_out_fc = ReLU()(fb_gru_out_fc)
        fb_gru_out_fc = Lambda(transpose, arguments={'axis':[0,2,1]})(fb_gru_out_fc)
        fb_gru_out_fc = Lambda(reshape, arguments={'axis':[self.batch_size, self.block_len // 2 + 1, 1, -1]})(fb_gru_out_fc)
        
        #print(fb_gru_out_fc) # 8, 257, 1, ?
        
        # pad
        input_mag = Lambda(transpose, arguments={'axis':[0,3,2,1]})(input_mag) # B T F C  ==> B C F T
        #print(input_mag)
        noisy_mag_pad = tf.pad(input_mag, [[0,0],[0,0],[4,4],[0,0]], "REFLECT")
        #print(noisy_mag_pad) # 8, 1, 265, ?
        
        
        noisy_mag_unfolded = tf.extract_image_patches(noisy_mag_pad, [1,1,9,1], [1,1,1,1], [1,1,1,1], padding='VALID')
        #print(noisy_mag_unfolded)
        
        noisy_mag_unfolded = Lambda(reshape, arguments={'axis':[self.batch_size, 1, self.block_len // 2 + 1, 9, -1]})(noisy_mag_unfolded) # 8 1 F 9 T
        noisy_mag_unfolded = Lambda(transpose, arguments={'axis':[0,2,3,4,1]})(noisy_mag_unfolded)
        noisy_mag_unfolded = Lambda(reshape, arguments={'axis':[self.batch_size, self.block_len // 2 + 1, 9, -1]})(noisy_mag_unfolded)
        
        #print(noisy_mag_unfolded) # 8 257 9 399
        
        sb_input = Concatenate(axis = 2)([noisy_mag_unfolded, fb_gru_out_fc]) # 8 257 10 399
        
        # 新加 效果不好
        #sb_input = Lambda(transpose, arguments={'axis':[0,3,1,2]})(sb_input) # 8 399 257 10
        #sb_input = LayerNormalization(axis = [-1,-2], epsilon = 1e-9, name = 'sb_input_norm')(sb_input)
        #sb_input = Lambda(transpose, arguments={'axis':[0,2,3,1]})(sb_input)
        # 新加结束
        
        sb_input = Lambda(transpose, arguments={'axis':[0,1,3,2]})(sb_input)
        sb_input = Lambda(reshape, arguments={'axis':[self.batch_size*(self.block_len // 2 + 1), -1, 10]})(sb_input)
        #print(sb_input) # 2056, 399, 10  注意B T F的输入格式是没有问题的
        
        
        
        # subband for tflite
        # sb_gru_out_1 = keras.layers.GRU(units=16, return_sequences=True, recurrent_activation = 'sigmoid',reset_after = False)(sb_input)
        # sb_gru_out_2 = keras.layers.GRU(units=16, return_sequences=True, recurrent_activation = 'sigmoid',reset_after = False)(sb_gru_out_1)
        
        # subband for training
        sb_gru_out_1 = keras.layers.CuDNNGRU(units=16, return_sequences=True)(sb_input)
        sb_gru_out_2 = keras.layers.CuDNNGRU(units=16, return_sequences=True)(sb_gru_out_1)
        #print(sb_gru_out_2) # 2056, 399, 16
        
        sb_mask = keras.layers.Dense(units = 2,)(sb_gru_out_2)
        #print(sb_mask) # B*F T 2
        
        sb_mask = Lambda(reshape, arguments={'axis':[self.batch_size, (self.block_len // 2 + 1), -1, 2]})(sb_mask)
        sb_mask = Lambda(transpose, arguments={'axis':[0,2,1,3]})(sb_mask)
        
        #print(sb_mask)
        
        enh_spec = Lambda(self.mk_mask_complex)([real,imag,sb_mask])
        
        
        enh_frame = Lambda(self.ifftLayer,arguments = {'mode':'real_imag'})(enh_spec)
        enh_frame = enh_frame * self.win
        enh_time = Lambda(self.overlapAddLayer, name = 'enhanced_time')(enh_frame)
        
        self.model = Model([time_dat, self.batch_gain], enh_time)
        #self.model = Model(time_dat, enh_time)
        self.model.summary()

        self.model_inference = Model(time_dat, enh_time)

        return self.model
        
    def compile_model(self):
        '''
        Method to compile the model for training
        '''
        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(),optimizer=optimizerAdam)

    def train(self, runName, data_generator):
        '''
        Method to train the model. 
        '''
        self.compile_model()
        
        # create save path if not existent
        savePath = './models_'+ runName+'/' 
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        csv_logger = CSVLogger(savePath+ 'training_' +runName+ '.log')
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=10**(-10), cooldown=1)
        # create callback for early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, 
            patience=30,  mode='auto', baseline=None)
        # create model check pointer to save the best model

        checkpointer = ModelCheckpoint(savePath+runName+'model_{epoch:02d}_{val_loss:02f}.h5',
                                       monitor='val_loss',
                                       save_best_only=False,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # create data generator for training data

        self.model.fit_generator(data_generator.generator(batch_size = self.batch_size,validation = False), 
                                 validation_data = data_generator.generator(batch_size =self.batch_size,validation = True),
                                 epochs = self.max_epochs, 
                                 steps_per_epoch = data_generator.train_length//self.batch_size,
                                 validation_steps = 100,
                                 #use_multiprocessing=True,
                                 callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping])
        # clear out garbage
        tf.keras.backend.clear_session()
    
    def test(self, noisy, out = None, weights_file = None):
        '''
        Method to test a trained model on a single file or a dataset.
        '''
        if weights_file:
            self.model.load_weights(weights_file)
            
        if os.path.exists(noisy):
            if os.path.isdir(noisy):
                file_list = librosa.util.find_files(noisy,ext = 'wav')
                if not os.path.exists(out):
                    os.mkdir(out)
                for f in tqdm.tqdm(file_list):
                    self.enhancement_single(f, output_f = os.path.join(out,os.path.split(f)[-1]), plot = False)
            if os.path.isfile(noisy):
                self.enhancement_single(noisy, output_f = out, plot = True)
        else:
            raise ValueError('The noisy file does not exist!')
            
    def enhancement_single(self, noisy_f, output_f = './enhance_s.wav', plot = True):
        '''
        Method to enhance a single file and plot figure
        '''
        if not self.model:
            raise ValueError('The FULLSUB model is not defined!')
             
        noisy_s = sf.read(noisy_f,dtype = 'float32')[0]
        
        enh_s = self.model_inference.predict(np.array([noisy_s]))
        
        enh_s = enh_s[0]
        
        #enh_s = enh_s/np.max(np.abs(enh_s))
        
        sf.write(output_f,enh_s,16000)
        return noisy_s,enh_s    

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--cuda", type = int, default = 5, help = 'which GPU to use')
    parser.add_argument("--mode", type = str, default = 'test', help = 'train or test')
    parser.add_argument("--bs", type = int, default = 8, help = 'batch size')
    parser.add_argument("--lr", type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument("--max_epochs", type = int, default = 200, help = 'Maximum number of epochs')
    parser.add_argument("--name", type = str, default = 'experiment_1', help = 'the experiment name')
    parser.add_argument("--second", type = int, default = 5, help = 'length in second of every sample')
    # parser.add_argument("--ckpt", type=str, default = 'models_exp_6-3/exp_6-3model_05_-17.004533.h5', help = 'the location of the weights')
    parser.add_argument("--ckpt", type=str, default = 'models_exp0-1/exp0-1model_109_-10.551251.h5', help = 'the location of the weights')
    parser.add_argument("--train_dir", type=str, default = TRAIN_DIR, help = 'the location of training data')
    # parser.add_argument("--val_dir", type=str, default = VAL_DIR, help = 'the location of val data')
    # parser.add_argument("--rir_dir", type=str, default = RIR_DIR, help = 'the location of rir data')
    parser.add_argument("--win_length", type=int, default = 512, help = 'window length of STFT')
    parser.add_argument("--hop_length", type=int, default = 256, help = 'hop length of STFT')   
    parser.add_argument("--test_dir", type=str, default = 'test_gucheng.wav', help = 'the floder of noisy speech or a single file')
    parser.add_argument("--output_dir", type=str, default = 'test_gucheng_enhanced_0-1.wav', help = 'the floder of enhanced speech or a single file')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    
    if args.mode == 'train':
        dg = data_generator(train_dir = args.train_dir,
##                            RIR_dir = args.rir_dr,
                            length_per_sample = args.second,
                            n_fft = args.win_length,
                            n_hop = args.hop_length)
        fullsub = FULLSUB_model(batch_size = args.bs, 
                            length_in_s = args.second, 
                            lr = args.lr, 
                            max_epochs = args.max_epochs, 
                            block_len = args.win_length,
                            block_shift = args.hop_length)
        fullsub.build_FULLSUB_model()
        fullsub.train(runName = args.name, data_generator = dg)
        
    elif args.mode == 'test':
        # batch size = 1 in test
        fullsub = FULLSUB_model(batch_size = 1, 
                            length_in_s = args.second, 
                            lr = args.lr,
                            block_len = args.win_length,
                            block_shift = args.hop_length)
        model = fullsub.build_FULLSUB_model()
        fullsub.test(noisy = args.test_dir, out = args.output_dir, weights_file = args.ckpt)
    else:
        raise ValueError('Running mode only support train or test!')
