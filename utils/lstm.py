
# Standard library imports
import os
import pickle
import argparse

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model
from tensorflow.keras.optimizers import Adam  # Leave as is to avoid issues

# Local application imports
from .process import divide_into_months, divide_into_periods, get_data


##########################################################
######### LSTM-BASED MODEL AND TRAINING ROUTINES #########
##########################################################


def create_lstm_model(config):
    '''
    LSTM-based model for sequence-to-sequence models where both input and output sequences are of the same lenghts
    and with shape T x N_features . Number of input and output features may differ.
    Args:
        config : a class having the required model data as attributes.
    '''
    inputs = layers.Input((config.MAX_LEN,config.INPUT_FEATURES))
    x = inputs
    for ib in range(config.NUM_LAYERS):
        x = layers.LSTM(config.EMBED_DIM, return_sequences=True)(x)
        x = layers.Dropout(config.DROPOUT)(x)
    output = layers.Conv1D(filters=config.OUTPUT_FEATURES, kernel_size=1)(x)
    model = keras.Model(inputs, output)
    model.summary()
    return model


def run_lstm(batch_size, adam, num_layers, embed_dim, week, dropout, calc_folder, verbose, epochs=50, basefolder='.'):

    if verbose:
        verbose_level=1
    else:
        verbose_level=0
    # 1. Get data
    n_training_data=8000 # 80-20 slpit
    monthly_aggregation=False
    period_aggregation=week
    if week==1000:
        monthly_aggregation=True
        period_aggregation=None
    if monthly_aggregation and period_aggregation is not None:
        raise ValueError('Choose just one between monthly and period aggregation')
    if not os.path.exists(calc_folder):
        os.mkdir(calc_folder)
    drivers_raw, ig_raw = get_data(basefolder=basefolder, plot=False)

    # 2. Preprocess into aggregated values
    drivers={}
    ig={}
    if monthly_aggregation:
        for site in ['tropical','boreal','temperate']:
            n_samples, n_times, n_feat = drivers_raw[site].shape
            #
            temp=divide_into_months(drivers_raw[site], mode='mean')
            n_months=temp.shape[1]//n_feat
            drivers[site]=temp.reshape((n_samples,n_feat,n_months))
            drivers[site]=np.transpose(drivers[site], (0, 2, 1))
            #
            temp=divide_into_months(ig_raw[site], mode='mean')
            n_months=temp.shape[1]//n_feat
            ig[site]=temp.reshape((n_samples,n_feat,n_months))
            ig[site]=np.transpose(ig[site], (0, 2, 1))
    elif period_aggregation is not None:
        for site in ['tropical','boreal','temperate']:
            drivers[site]=divide_into_periods(drivers_raw[site],
                                                window=period_aggregation,
                                                mode='mean')
            ig[site]=divide_into_periods(ig_raw[site],
                                        window=period_aggregation,
                                        mode='mean')
    else:
        raise NotImplementedError('error aggregation mode')

    # 3. Rescale
    drivers_rescaled={}
    ig_rescaled={}
    for site in ['tropical','boreal','temperate']:
        #
        scaler_module.create_scaler(ig[site][:n_training_data,...],
                                    mode=f'input_{site}',
                                    calc_folder=calc_folder)
        #
        scaler_module.create_scaler(drivers[site][:n_training_data,...],
                                    mode=f'output_{site}',
                                    calc_folder=calc_folder)
        #
        ig_rescaled[site]=scaler_module.scale_data(ig[site],
                                                        mode=f'input_{site}',
                                                        calc_folder=calc_folder)
        #
        drivers_rescaled[site]=scaler_module.scale_data(drivers[site],
                                                mode=f'output_{site}',
                                                calc_folder=calc_folder)

    # 4. Check dimensions
    for site in ['tropical','boreal','temperate']:
        print('raw dim: ',drivers_raw[site].shape, ig_raw[site].shape)
        print('aggregated dim: ',drivers[site].shape, ig[site].shape)
        print('aggregated and standardized dim: ',drivers_rescaled[site].shape, ig_rescaled[site].shape)
        print()

    # 5. LSTM paramters 
    max_len=ig_rescaled['tropical'].shape[1]
    input_features=ig_rescaled['tropical'].shape[2]
    output_features=drivers_rescaled['tropical'].shape[2]
    class Config_lstm:
        # data dimensions
        MAX_LEN = max_len
        INPUT_FEATURES = input_features
        OUTPUT_FEATURES = output_features
        # training
        BATCH_SIZE = batch_size
        ADAM = adam
        EPOCHS = epochs
        # lstm block parameters
        EMBED_DIM = embed_dim
        NUM_LAYERS = num_layers
        DROPOUT = dropout
    config_lstm = Config_lstm()
    save_lstm_card(config_lstm, calc_folder, additional={'PERIOD': week, 'MODEL': 'lstm'})

    # 5. Run and save models
    histories={}
    for site in ['tropical','boreal','temperate']:
        model=create_lstm_model(config_lstm)
        optimizer = Adam(config_lstm.ADAM)
        model.compile(loss = 'mse',
                    optimizer = optimizer,
                    metrics = 'mse' )
        save_path = f'{calc_folder}/{site}/{{epoch}}.ckpt'
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                        save_weights_only=True,
                                                        period=5)
        res = model.fit(ig_rescaled[site][:n_training_data,:,:],
                            drivers_rescaled[site][:n_training_data,:,:],
                            batch_size=config_lstm.BATCH_SIZE,
                            epochs=config_lstm.EPOCHS,
                            verbose=verbose_level,
                            validation_data=(ig_rescaled[site][n_training_data:,:,:],drivers_rescaled[site][n_training_data:,:,:]),
                            callbacks=[save_callback])
        histories[f'{site}'] = res.history
    pickle.dump(histories, open(f'{calc_folder}/histories.p','wb'))

    return 


def load_lstm(batch_size, adam, num_layers, embed_dim, week, dropout, calc_folder, verbose, epoch_to_load, basefolder='.'):
    # Verbose setup
    verbose_level = 1 if verbose else 0

    # Data parameters
    n_training_data = 8000
    monthly_aggregation = False
    period_aggregation = week
    if week == 1000:
        monthly_aggregation = True
        period_aggregation = None
    if monthly_aggregation and period_aggregation is not None:
        raise ValueError('Choose just one between monthly and period aggregation')
    if not os.path.exists(calc_folder):
        raise FileNotFoundError(f"Calculation folder {calc_folder} does not exist!")

    # 1. Get data
    drivers_raw, ig_raw = get_data(basefolder=basefolder, plot=False)

    # 2. Preprocess into aggregated values
    drivers = {}
    ig = {}
    sites = ['tropical','boreal','temperate']

    if monthly_aggregation:
        for site in sites:
            n_samples, n_times, n_feat = drivers_raw[site].shape
            temp = divide_into_months(drivers_raw[site], mode='mean')
            n_months = temp.shape[1] // n_feat
            drivers[site] = np.transpose(temp.reshape((n_samples, n_feat, n_months)), (0, 2, 1))

            temp = divide_into_months(ig_raw[site], mode='mean')
            n_months = temp.shape[1] // n_feat
            ig[site] = np.transpose(temp.reshape((n_samples, n_feat, n_months)), (0, 2, 1))
    elif period_aggregation is not None:
        for site in sites:
            drivers[site] = divide_into_periods(drivers_raw[site], window=period_aggregation, mode='mean')
            ig[site] = divide_into_periods(ig_raw[site], window=period_aggregation, mode='mean')
    else:
        raise NotImplementedError('Error: aggregation mode not set')

    # 3. Rescale
    drivers_rescaled = {}
    ig_rescaled = {}
    for site in sites:
        ig_rescaled[site] = scaler_module.scale_data(ig[site], mode=f'input_{site}', calc_folder=calc_folder)
        drivers_rescaled[site] = scaler_module.scale_data(drivers[site], mode=f'output_{site}', calc_folder=calc_folder)

    # 4. LSTM parameters
    max_len = ig_rescaled['tropical'].shape[1]
    input_features = ig_rescaled['tropical'].shape[2]
    output_features = drivers_rescaled['tropical'].shape[2]

    class Config_lstm:
        MAX_LEN = max_len
        INPUT_FEATURES = input_features
        OUTPUT_FEATURES = output_features
        BATCH_SIZE = batch_size
        ADAM = adam
        EPOCHS = 0  # not used for loading
        EMBED_DIM = embed_dim
        NUM_LAYERS = num_layers
        DROPOUT = dropout

    config_lstm = Config_lstm()

    # 5. Load models
    models = {}
    for site in sites:
        model = create_lstm_model(config_lstm)
        weights_path = f'{calc_folder}/{site}/{epoch_to_load}.ckpt'
        if not os.path.exists(f'{weights_path}.index'):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        model.load_weights(weights_path)
        optimizer = Adam(config_lstm.ADAM)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        models[site] = model

    return models, ig_rescaled, drivers_rescaled


def save_lstm_card(config, savefolder, additional={}):
    with open(f'{savefolder}/card','w') as f:
        f.write(f'BATCH_SIZE: {config.BATCH_SIZE}\n')
        f.write(f'ADAM: {config.ADAM}\n')
        f.write(f'NUM_LAYERS: {config.NUM_LAYERS}\n')
        f.write(f'DROPOUT: {config.DROPOUT}\n')
        f.write(f'EMBED_DIM: {config.EMBED_DIM}\n')
        f.write(f'INPUT_FEATURES: {config.INPUT_FEATURES}\n')
        f.write(f'MAX_LEN: {config.MAX_LEN}\n')
        f.write(f'OUTPUT_FEATURES: {config.OUTPUT_FEATURES}\n')
        for add in additional.keys():
            f.write(f'{add}: {additional[add]}\n')
    return


def load_card(savefolder):
    with open(f'{savefolder}/card','r') as f:
        lines=f.readlines()
        data={}
        for line in lines:
            lsplit=line.split(':')
            data[lsplit[0]]=lsplit[1].strip()
    return data


###################################################################################
################ CLASS TO HANDLE NORMALIZATION AND SCALING OF DATA ################
###################################################################################


class scaler_module():
  '''
  Usage:
    scaler_module.create_scaler(data, mode, calc_folder) creates a scaler from the data and saves it.
    scaler = scaler_module.get_scaler(mode, calc_folder) loads the saved scaler
    scaled = scaler_module.scale_data(data, mode, calc_folder) scales the data with the saved scaler.
    unscaled = scaler_module.unscale_data(data, mode, calc_folder) un scales the data with the save scaler.
    Alternatively a scaler can be given as input in the last too commands (not recommended for readability may be used for testing).
  '''

  @staticmethod
  def create_scaler(data, mode, calc_folder):
      n_samples, n_times, n_feat = data.shape
      scaler = StandardScaler()
      scaler.fit(data.reshape((n_samples,-1)))
      pickle.dump(scaler, open(f'{calc_folder}/{mode}_scaler.p','wb'))
      print(f'scaler saved to {calc_folder}/{mode}_scaler.p. Use it for unscaling.')
      return

  @staticmethod
  def scale_data(data, scaler=None, mode=None, calc_folder=None):
      n_samples, n_times, n_feat = data.shape
      if scaler is None:
          if mode is None or calc_folder is None:
            raise ValueError('Scaler is not provided, but mode and calc_folder are not set')
          scaler=scaler_module.get_scaler(mode, calc_folder)
      else:
          if mode is not None and calc_folder is not None:
            raise ValueError('Scaler is provided, but mode and calc_folder are also set')
      scaled=scaler.transform(data.reshape((n_samples,-1)))
      scaled=scaled.reshape((n_samples, n_times, n_feat))
      return scaled

  @staticmethod
  def unscale_data(data, scaler=None, mode=None, calc_folder=None):
      n_samples, n_times, n_feat = data.shape
      if scaler is None:
          if mode is None or calc_folder is None:
            raise ValueError('Scaler is not provided, but mode and calc_folder are not set')
          scaler=scaler_module.get_scaler(mode, calc_folder)
      else:
          if mode is not None and calc_folder is not None:
            raise ValueError('Scaler is provided, but mode and calc_folder are also set')
      unscaled=scaler.inverse_transform(data.reshape((n_samples,-1)))
      unscaled=unscaled.reshape((n_samples, n_times, n_feat))
      return unscaled

  @staticmethod
  def get_scaler(mode, calc_folder):
      scaler=pickle.load(open(f'{calc_folder}/{mode}_scaler.p','rb'))
      return scaler
  



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, required=True,
                        help='Batch size')
    parser.add_argument('-ad', '--adam', type=float, required=True,
                        help='Learning rate')
    parser.add_argument('-nl', '--num_layers', type=int, required=True,
                        help='Number of LSTM layers')
    parser.add_argument('-ed', '--embed_dim', type=int, required=True,
                        help='Internal dimension parameter')
    parser.add_argument('-w', '--week', type=int, required=True,
                        help='Period over which to average')
    parser.add_argument('-cf', '--calc_folder', type=str, required=True,
                        help='calculation_folder')
    parser.add_argument('-dr', '--dropout', type=float, required=True,
                        help='Dropout')
    parser.add_argument('-d', '--dry_run', 
                        help='A dry run', action='store_true')
    parser.add_argument('-v', '--verbose', 
                        help='Verbose level', action='store_true')
    args = parser.parse_args()
    batch_size=args.batch_size
    adam=args.adam
    num_layers=args.num_layers
    embed_dim=args.embed_dim
    week=args.week
    calc_folder=args.calc_folder
    dropout=args.dropout
    verbose=args.verbose
    for var, n_var in zip([batch_size, adam, num_layers, embed_dim, week, dropout, calc_folder],
                         ['batch_size', 'adam', 'num_layers', 'embed_dim', 'week', 'dropout', 'calc_folder']):
        print(f'{n_var}: {var}')        
    if not args.dry_run:
        run_lstm(batch_size, adam, num_layers, embed_dim, week, dropout, calc_folder, verbose)


