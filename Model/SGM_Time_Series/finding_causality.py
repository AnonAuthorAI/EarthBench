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

DAG_encoder_hid_dim = 150
DAG_encoder_out_dim = 150
DAG_decoder_hid_dim = 150

lr_decay = 200
gamma = 1.0

k_max = 3
num_epochs = 50

# non-acyclic tolerance
h_tol = 1e-6

tau_A = 0.0
use_A_connect_loss = 0.0
use_A_positiver_loss = 0.0

graph_threshold = 1e-3  # cut trivial casual relations, ~ 1/num_counties

lr = 1e-4

# This python file is responsible for discovering the causality (i.e., DAG) among all counties in each hour based on their features

# Road Data
for controlled_group_id in controlled_group_id_list:
    for cross_validation_id in cross_validation_id_list:
        print('----------------------------------------------------------------------------------------------------------------------------------------')
        print('controlled_group_id: {}, cross_validation_id: {}'.format(controlled_group_id, cross_validation_id))

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

        all_X = torch.concat((X_train, X_val, X_test), 1)
        all_Y = torch.concat((Y_train, Y_val, Y_test), 1)

        training_data_DAG = Load_Dataset(torch.flatten(all_X, start_dim=1, end_dim=2).permute(1, 0, 2),torch.flatten(all_Y, start_dim=1, end_dim=2).permute(1, 0, 2))
        training_dataloader_DAG = DataLoader(training_data_DAG, batch_size=window_size, shuffle=True)
        # {For the coding simplicity, let DAG learning observe the ground-truth X_test features
        # instead of the predicted X_test features by prediction model, is based on,
        # the assumption that the predicted is very close to the ground-truth, then the causal structures are isomorphism}

        # {The most reasonable way is all_X only contains X_train and X_val,
        # then let DAG learning observe predicted X_test and get the learned DAG step by step, i.e., sequentially}

        time_conv = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'time_conv.pt', map_location=device)
        feature_encoder = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'feature_encoder.pt', map_location=device)
        feature_decoder = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'feature_decoder.pt', map_location=device)
        adj_A = np.zeros((num_counties, num_counties))

        DAG_encoder = MLP_Encoder(n_feat_dim=feature_encoder_hid_dim, n_hid=DAG_encoder_hid_dim, n_out=DAG_encoder_out_dim, adj_A=adj_A, batch_size=window_size).double()
        DAG_encoder = DAG_encoder.to(device)
        # window_size here stands for how many timestamps share an adjacency matrix
        DAG_decoder = MLP_Decoder(n_in_z=DAG_encoder_out_dim, n_out=feature_encoder_hid_dim, data_variable_size=num_counties, batch_size=window_size, n_hid=DAG_decoder_hid_dim).double()
        DAG_decoder = DAG_decoder.to(device)

        optimizer = optim.Adam(list(DAG_encoder.parameters()) + list(DAG_decoder.parameters()), lr=1e-4)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

        best_ELBO_graph_seq = []

        best_ELBO_loss = np.inf

        best_ELBO_epoch = 0

        min_num_edegs = np.inf

        best_k = 0

        c_A = 1

        lambda_A = 0.78

        # for non-acyclic measurement
        hA_old = 1
        hA_new = 1

        for k in range(k_max):
            print('k: {}, lambda_A: {}'.format(k, lambda_A))
            while c_A < 1e+4:  # while loop is responsible for updating the lambda, if the current lambda_A is well-preserved
                for epoch in range(num_epochs):
                    nll_train = []
                    kl_train = []
                    mse_train = []
                    hA_train = []

                    ELBO_graph_seq = []
                    NLL_graph_seq = []
                    MSE_graph_seq = []

                    t = time.time()

                    # training
                    DAG_encoder.train()
                    DAG_decoder.train()
                    scheduler.step()

                    optimizer, lr = update_optimizer(optimizer, lr, c_A)

                    for data in training_dataloader_DAG:  # one batch share a graph, where the batch_size is pre-defined as the window_size

                        print(data[0].shape)

                        # --- get raw data, (counties, features) per hour
                        feature, _ = data
                        feature = feature.to(device)
                        x = time_conv(feature)
                        # print(x.shape)  # (batch_size, 238, 45)
                        # ---

                        # --- get the latent representation of raw features
                        h = torch.zeros((x.shape[0], num_counties, feature_encoder_hid_dim))
                        h = h.to(device)

                        for i in range(num_counties):
                            h_cat = x[:,i,:].unsqueeze(1)
                            # print(h_cat.shape)  # obtain the hourly record in each batch (b, 1, k), (128, 1, 45)
                            _, h_end = feature_encoder(h_cat)
                            h_end = h_end.view(x.shape[0], -1)
                            # print(h_end.shape)  # obtain the encoded hourly record in each batch (b, 1, h), (128, 1, 100)
                            h[:,i,:] = h_end

                        # print(h.shape)  # (batch_size, 238, 100)
                        # ---

                        # --- build DAG graphs on h
                        h = Variable(h).double()
                        # print(h.shape)  # [window_size, 238, 100]
                        optimizer.zero_grad()

                        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = DAG_encoder(h)  # logits is of size: [num_sims, z_dims]
                        edges = logits
                        # print(origin_A.shape)  #[238, 238]

                        dec_x, output, adj_A_tilt_decoder = DAG_decoder(edges, origin_A, adj_A_tilt_encoder, Wa)

                        target = h
                        preds = output
                        variance = 0.

                        # reconstruction accuracy loss
                        loss_nll = nll_gaussian(preds, target, variance)

                        # KL loss
                        loss_kl = kl_gaussian_sem(logits)

                        # ELBO loss:
                        loss = loss_kl + loss_nll

                        # add A loss
                        one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
                        sparse_loss = tau_A * torch.sum(torch.abs(one_adj_A))

                        # other loss term
                        if use_A_connect_loss:
                            connect_gap = A_connect_loss(one_adj_A, graph_threshold, z_gap)
                            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

                        if use_A_positiver_loss:
                            positive_gap = A_positive_loss(one_adj_A, z_positive)
                            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

                        # compute h(A)
                        hA = h_A(origin_A, num_counties)  # the last hA
                        loss += lambda_A * hA + 0.5 * c_A * hA * hA + 100. * torch.trace(origin_A * origin_A) + sparse_loss  # +  0.01 * torch.sum(variance * variance)

                        loss.backward()
                        optimizer.step()

                        hA_train.append(hA.item())
                        mse_train.append(F.mse_loss(preds, target).item())
                        nll_train.append(loss_nll.item())
                        kl_train.append(loss_kl.item())

                        graph = origin_A.cpu().data.clone().numpy()  # one batch share a graph, where the batch_size is pre-defined as the window_size
                        graph = graph / graph.sum(axis=0)
                        graph[np.abs(graph) < graph_threshold] = 0
                        ELBO_graph_seq.append(graph)

                    nll_loss = np.mean(nll_train)
                    kl_loss = np.mean(kl_train)
                    elbo_loss = nll_loss + kl_loss
                    mse_loss = np.mean(mse_train)

                    print('Epoch: {:04d} |'.format(epoch),
                          'nll_train: {:.10f} |'.format(nll_loss),
                          'kl_train: {:.10f} |'.format(kl_loss),
                          'ELBO_loss: {:.10f} |'.format(elbo_loss),
                          'mse_train: {:.10f} |'.format(mse_loss),
                          'Avg. Non-acyclic measurement: {} |'.format(np.mean(hA_train)),
                          'time: {:.4f}s'.format(time.time() - t))

                    num_elbo_edges = get_num_edges(ELBO_graph_seq)
                    print('Total num of edges of all graphs: {}'.format(num_elbo_edges))

                    hA_new = np.mean(hA_train)

                    if nll_loss <= 1e-3 and kl_loss <= 1e-3 and elbo_loss <= 1e-3 and mse_loss <= 1e-5 and hA_new <= 1e-3:
                        # find a loss-tolerate candidate graph sequence
                        if num_elbo_edges < min_num_edegs:
                            # effective number of edges gets distillates
                            best_k = k
                            min_num_edegs = num_elbo_edges
                            best_ELBO_epoch = epoch
                            best_ELBO_loss = elbo_loss
                            best_ELBO_graph_seq = ELBO_graph_seq
                            print('\t--------------------------------------------------------------------')
                            print('\t| Find a suitable graph sequence at k={} c_A={}'.format(best_k, c_A))
                            print('\t| loss: {}, {}, {}, {}'.format(nll_loss, kl_loss, elbo_loss, hA_new))
                            print('\t| Num of edges: {}'.format(num_elbo_edges))
                            print('\t| Current best epoch: {}'.format(best_ELBO_epoch))
                            np.save(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'best_ELBO_graph_seq.npy', np.array(best_ELBO_graph_seq))
                            print('\t| Current best gets saved')
                            print('\t--------------------------------------------------------------------')

                if hA_new > 0.5 * hA_old:  # the non-acyclic measurement in the new round increases
                    c_A *= 10               # at most, give it 1e+5/10 times of trial
                    print('c_A trial percentage: {}/{}'.format(c_A, 10000))
                    print('The new hA did not decrease as half as the old')
                else:
                    break  # break while loop and finish the epoch iteration. Then update lambda for the next epoch iteration

            hA_old = hA_new
            lambda_A += c_A * hA_new

            if hA_new <= h_tol:  # graph is almost DAG, break the for-loop, i.e., k iteration
                break

        print('Best k iteration: {}'.format(best_k),
              'Best ELBO epoch: {}'.format(best_ELBO_epoch),
              "ELBO loss: {}".format(best_ELBO_loss),
              "Num of hours: {}".format(len(best_ELBO_graph_seq)),
              "Num of edges: {}".format(get_num_edges(best_ELBO_graph_seq)))
        np.save(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'best_ELBO_graph_seq.npy', np.array(best_ELBO_graph_seq))