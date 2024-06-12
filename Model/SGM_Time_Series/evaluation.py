from modules import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'processed_weather_data'
modules_dir = "saved_modules"

# Load Data
controlled_group_id_list = [0]
cross_validation_id_list = [1, 2, 3, 4]

for controlled_group_id in controlled_group_id_list:
    for cross_validation_id in cross_validation_id_list:

        X_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_test.pt')
        Y_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Y_test.pt')
        X_test = feature_l2_norm(X_test)

        # ---------------------------------Forecasting Performance-------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------

        print('Evaluating forecasting performance on testing data ...')

        X_test_RNN, Y_test_RNN = generate_sequential_data(torch.flatten(X_test, start_dim=1, end_dim=2).permute(1,0,2), period=24)
        # print(Y_test_RNN.shape)  # torch.Size([2904, 24, 238, 45])
        num_test_samples = X_test_RNN.shape[0]
        # print(num_test_samples)  # 2904

        DYNDAG = np.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'best_ELBO_graph_seq.npy')  # (num_days, num_nodes, num_nodes)
        X_train_DYNDAG, X_val_DYNDAG, X_test_DYNDAG = get_training_val_test_DYNDAG(DYNDAG, num_training_samples=8808, num_val_samples=2904, num_test_samples=2904, num_counties=238, window_size=24)

        parallelization_mode = False

        dcrnn_model = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'best_dcrnn_model.pt', map_location=device)

        test_combined = Combine_Dataset(X_test_RNN, Y_test_RNN, X_test_DYNDAG)
        test_dataloader_RNN = DataLoader(test_combined, batch_size=1, shuffle=False)

        testing_batch = 0
        testing_loss = []
        detection_list = []
        dcrnn_model.eval()

        for x, y, dyn_adj in test_dataloader_RNN:

            testing_batch += 1
            # print('\ttesting batch: {}'.format(testing_batch))

            x = torch.flatten(x.permute(1, 0, 2, 3), start_dim=2).to(device)  # after flatten (1, 0, 2*3) (batch_size, period, num_nodes*num_features)
            print(x.shape)
            y = torch.flatten(y.permute(1, 0, 2, 3), start_dim=2).to(device)
            print(y.shape)

            dyn_adj = dyn_adj.to(device)  # (batch, num_nodes, num_nodes)

            output = dcrnn_model(x, dyn_adj, y, parallelization_mode, batches_seen=0)

            # output = 0.1*output + 0.9*x # persistence forecasting

            # print(output.shape)  # [24, 1, 10710], 1 is batch size

            loss = masked_mae_loss(y, output)

            testing_loss.append(loss.item())

            output = output.detach().cpu().permute(1, 0, 2).view(1, 24, 238, 45)
            output = torch.flatten(output, start_dim=0, end_dim=1)  # (24, 238, 45)

            if testing_batch == 1:
                for idx in range(output.shape[0]):
                    detection_list.append(torch.unsqueeze(output[idx], dim=0))
            else:
                detection_list.append(torch.unsqueeze(output[-1], dim=0))
            # print(len(detection_list))

        print('Forecasting testing_loss: {:.10f}'.format(np.mean(testing_loss)))

        detection_list = torch.stack(detection_list, dim=0)
        # print(detection_list.shape)  # (2927, 238, 45)
        torch.save(detection_list, modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_prediction.pt')


        # ---------------------------------Anomaly Detection Performance-------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        print('Evaluating anomaly detection performance on testing data ...')

        X_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_test.pt')
        Y_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Y_test.pt')
        X_test = feature_l2_norm(X_test)

        # --- X_predict get one from X_test, because the predicted Y_test skip one day for being the trigger in X_test
        detection_list = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_prediction.pt')
        X_test = torch.flatten(X_test, start_dim=1, end_dim=2).permute(1,0,2)  # should be (2952, 238, 45)
        X_prediction = torch.cat((X_test[:24], detection_list[:,0,:,:], torch.unsqueeze(X_test[-1], dim=0)), 0)
        # print(X_prediction.shape)  # (2952, 238, 45)
        X_prediction = X_prediction.permute(1,0,2).view(238, -1, 24, 45)
        # print(X_prediction.shape)  # (238, 123, 24, 45)

        # X_predict go through feature decoder
        time_conv = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'time_conv.pt', map_location=device)
        Feature_Encoder = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Feature_Encoder.pt', map_location=device)
        Feature_Decoder = torch.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Feature_Decoder.pt', map_location=device)

        generation_scores = torch.zeros((1, 45))  # an empty placeholder, feature generation scores
        generation_scores = generation_scores.to(device)

        testing_data_anom = Load_Dataset(torch.flatten(X_prediction, end_dim=1), torch.flatten(Y_test, end_dim=1))
        testing_dataloader_anom = DataLoader(testing_data_anom, batch_size=8, shuffle=False)

        loss_recon_fn = nn.MSELoss()
        loss_recon_fn = loss_recon_fn.to(device)
        testing_loss = []
        for data in testing_dataloader_anom:
            feature, _ = data
            feature = feature.to(device)

            x = time_conv(feature)
            _, h_end = Feature_Encoder(x)
            h_end = h_end.view(x.shape[0], -1)
            recons = Feature_Decoder(h_end)

            loss_recon = loss_recon_fn(recons, x)
            # testing_loss.append(loss_recon.item())
            testing_loss.append(loss_recon.item() if not math.isnan(loss_recon.item()) else 0.0)

            score = torch.sqrt((recons - x) ** 2)
            # generation_scores = torch.cat((generation_scores, torch.flatten(score, end_dim=1)))
            generation_scores = torch.cat((generation_scores, torch.flatten(torch.nan_to_num(score, nan=0.0), end_dim=1)))
        print('Reconstruction testing_loss: {:.10f}'.format(np.mean(testing_loss)))

        row_exclude = 0
        generation_scores = torch.cat((generation_scores[:row_exclude],generation_scores[row_exclude+1:]))  # remove the placeholder
        # print(generation_scores.shape)  # should be aligned with X_test

        np_scores = generation_scores.cpu().detach().numpy()
        # print(np_scores.shape)

        y = Y_test.view(-1,1).cpu().detach().numpy()
        from sklearn.metrics import roc_auc_score

        # Get the sign of each element
        signs = np.sign(np_scores)
        # Square the absolute value of each element
        squared_abs = np.abs(np_scores) ** 2
        # Multiply the signs by the squared absolute values
        result = signs * squared_abs

        hourly_scores = np.mean(result, 1)  # (num_counties * num_days_in_testing * num_hours_a_day, 1)
        roc_test = roc_auc_score(y, hourly_scores)
        # print(roc_test)
        print('Anomaly Detection AUC_ROC: {:.10f}'.format(roc_test))