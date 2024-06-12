import copy

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch import nn
from torch import optim
import numpy as np
import time
from modules import *
from utils import *
from torch.nn.functional import normalize
from torch.optim import lr_scheduler
from sklearn.preprocessing import normalize as skr_normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'processed_weather_data'
modules_dir = 'saved_modules'

# Load Data
controlled_group_id_list = [0]
cross_validation_id_list = [1, 2, 3, 4]

batch_size = 128
feature_encoder_hid_dim = 100
feature_decoder_hid_dim = 100
window_size = 24
num_epochs = 300

# This python file is responsible for capturing the latent feature of raw time-series feature vectors via reconstruction

for controlled_group_id in controlled_group_id_list:
    for cross_validation_id in cross_validation_id_list:
        print('controlled_group_id: {}, cross_validation_id: {}'.format(controlled_group_id, cross_validation_id))

        # (num_counties, num_days, num_hours_each_day, num_features)
        X_train = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_train.pt')
        Y_train = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Y_train.pt')
        X_val = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_val.pt')
        Y_val = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Y_val.pt')
        X_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_test.pt')
        Y_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Y_test.pt')

        # num_hours = X_train.shape[1] * X_train.shape[2]
        num_counties = X_train.shape[0]
        num_features = X_train.shape[3]

        X_train = feature_l2_norm(X_train)
        X_val = feature_l2_norm(X_val)
        X_test = feature_l2_norm(X_test)

        training_data_anom = Load_Dataset(torch.flatten(X_train, end_dim=1), torch.flatten(Y_train, end_dim=1))  # (num_counties * num_days, window_size, #features)
        training_dataloader_anom = DataLoader(training_data_anom, batch_size=batch_size, shuffle=True)

        val_data_anom = Load_Dataset(torch.flatten(X_val, end_dim=1), torch.flatten(Y_val, end_dim=1))
        val_dataloader_anom = DataLoader(val_data_anom, batch_size=batch_size, shuffle=True)

        testing_data_anom = Load_Dataset(torch.flatten(X_test, end_dim=1), torch.flatten(Y_test, end_dim=1))
        testing_dataloader_anom = DataLoader(testing_data_anom, batch_size=batch_size, shuffle=True)

        # Initialize Loss
        loss_recon_fn = nn.MSELoss()
        loss_recon_fn = loss_recon_fn.to(device)

        # Initialize Model
        time_conv = ConvLayer(n_features=num_features).double()
        time_conv = time_conv.to(device)
        feature_encoder = Feature_Encoder(num_features, hid_dim=feature_encoder_hid_dim, n_layers=1, dropout=0.2).double()
        feature_encoder = feature_encoder.to(device)
        feature_decoder = Feature_Decoder(window_size=window_size, in_dim=feature_encoder_hid_dim, hid_dim=feature_decoder_hid_dim, out_dim=num_features, n_layers=1, dropout=0.2).double()
        feature_decoder = feature_decoder.to(device)

        # Initialize Optimizer
        optimizer = optim.Adam(list(time_conv.parameters())
                               + list(feature_encoder.parameters())
                               + list(feature_decoder.parameters()), lr=1e-4)

        min_val_loss = np.inf
        best_time_conv = copy.deepcopy(time_conv)
        best_feature_encoder = copy.deepcopy(feature_encoder)
        best_feature_decoder = copy.deepcopy(feature_decoder)
        best_epoch = 0

        for epoch in range(num_epochs):

            t = time.time()

            time_conv.train()
            feature_encoder.train()
            feature_decoder.train()

            train_loss = []
            val_loss = []

            for data in training_dataloader_anom:
                feature, _ = data
                feature = feature.to(device)

                x = time_conv(feature)  # (b, n, k)

                _, h_end = feature_encoder(x)
                h_end = h_end.view(x.shape[0], -1)
                recons = feature_decoder(h_end)

                loss_recon = loss_recon_fn(recons, x)

                optimizer.zero_grad()

                loss_recon.backward()

                optimizer.step()

                train_loss.append(loss_recon.item())

            for data in val_dataloader_anom:
                feature, _ = data
                feature = feature.to(device)

                x = time_conv(feature)

                _, h_end = feature_encoder(x)
                h_end = h_end.view(x.shape[0], -1)
                recons = feature_decoder(h_end)

                loss_recon = loss_recon_fn(recons, x)
                val_loss.append(loss_recon.item())

                if loss_recon.item() <= min_val_loss:
                    min_val_loss = loss_recon.item()
                    best_epoch = epoch

                    best_time_conv = copy.deepcopy(time_conv)
                    best_feature_encoder = copy.deepcopy(feature_encoder)
                    best_feature_decoder = copy.deepcopy(feature_decoder)

            print('Epoch: {:04d}'.format(epoch),
                  'train_loss: {:.10f}'.format(np.mean(train_loss)),
                  'val_loss: {:.10f}'.format(np.mean(val_loss)),
                  'time: {:.4f}s'.format(time.time() - t))

        print('Best epoch is {}'.format(best_epoch))

        torch.save(best_time_conv, modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'time_conv.pt')
        torch.save(best_feature_encoder, modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'feature_encoder.pt')
        torch.save(best_feature_decoder, modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'feature_decoder.pt')

        time_conv = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'time_conv.pt', map_location=device)
        feature_encoder = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'feature_encoder.pt', map_location=device)
        feature_decoder = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'feature_decoder.pt', map_location=device)

        testing_loss = []
        for data in testing_dataloader_anom:
            feature, _ = data
            feature = feature.to(device)

            x = time_conv(feature)
            _, h_end = feature_encoder(x)
            h_end = h_end.view(x.shape[0], -1)
            recons = feature_decoder(h_end)

            loss_recon = loss_recon_fn(recons, x)
            testing_loss.append(loss_recon.item())

        print('testing_loss: {:.10f}'.format(np.mean(testing_loss)))