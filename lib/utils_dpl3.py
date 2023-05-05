import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from time import time
from scipy.sparse.linalg import eigs


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = (32 - distance)/32.0
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(float(row[0])), int(float(row[1])), float(row[2])#int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = (32 - distance)/32.0
            return A, distaneA

# def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
#     '''
#     Parameters
#     ----------
#     distance_df_filename: str, path of the csv file contains edges information
#     num_of_vertices: int, the number of vertices
#     Returns
#     ----------
#     A: np.ndarray, adjacency matrix
#     '''
#     if 'npy' in distance_df_filename:
#
#         adj_mx = np.load(distance_df_filename)
#
#         return adj_mx, None
#
#     else:
#
#         import csv
#
#         A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
#                      dtype=np.float32)
#
#         distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
#                             dtype=np.float32)
#
#         # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
#         if id_filename:
#
#             with open(id_filename, 'r') as f:
#                 id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
#
#             with open(distance_df_filename, 'r') as f:
#                 f.readline()  # 略过表头那一行
#                 reader = csv.reader(f)
#                 for row in reader:
#                     if len(row) != 3:
#                         continue
#                     i, j, distance = int(row[0]), int(row[1]), float(row[2])
#                     A[id_dict[i], id_dict[j]] = 1
#                     distaneA[id_dict[i], id_dict[j]] = distance
#             return A, distaneA
#
#         else:  # distance file中的id直接从0开始
#
#             with open(distance_df_filename, 'r') as f:
#                 f.readline()
#                 reader = csv.reader(f)
#                 for row in reader:
#                     if len(row) != 3:
#                         continue
#                     i, j, distance = int(row[0]), int(row[1]), float(row[2])
#                     A[i, j] = 1
#                     distaneA[i, j] = distance
#             return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(float(row[0])), int(float(row[1])), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A
    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)
    '''

    assert (A-A.transpose()).sum() == 0  # 首先确保A是一个对称矩阵

    D = np.diag(np.sum(A, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - A  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    return L


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - W  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    lambda_max = eigs(L, k=1, which='LR')[0].real  # 求解拉普拉斯矩阵的最大奇异值

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N) # 为邻居矩阵加上自连接
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix


def norm_Adj(W):
    '''
    compute  normalized Adj matrix
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def trans_norm_Adj(W):
    '''
    compute  normalized Adj matrix
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    W = W.transpose()
    N = W.shape[0]
    W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    trans_norm_Adj = np.dot(D, W)

    return trans_norm_Adj


def compute_val_loss(net, val_loader, criterion, sw, epoch):
    '''
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.transpose(-1,
                                                      -2)  # decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1) ->(B, N, T, F)

            # labels = labels.unsqueeze(-1)  # (B，N，T，1)
            # predict_length = labels.shape[2]
            predict_length = labels.shape[-1]  # T
            labels = labels.transpose(-1, -2)
            # encode
            encoder_output = net.encode(encoder_inputs)
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
            decoder_pump_inputs = decoder_inputs[:, :, :, 2:]  # 2 features
            decoder_input_list = [decoder_start_inputs]
            # 按着时间步进行预测
            # for step in range(predict_length):
            #     decoder_inputs = torch.cat(decoder_input_list, dim=2)
            #     predict_output = net.decode(decoder_inputs[:,:,:,-2:], encoder_output,encoder_inputs[:,:,-1:,:2])
            #     if step < predict_length - 1:
            #         decoder_input_list = [decoder_start_inputs,
            #                               torch.cat((predict_output, decoder_pump_inputs[:, :, 1:step + 2, :]), dim=3)]
            for step in range(predict_length):
                # decoder_inputs = torch.cat(decoder_input_list, dim=2)
                # if step==0:# added
                #     predict_output = net.decode(decoder_inputs[:,:,:,-2:], encoder_output,encoder_inputs[:,:,-1:,:2])#decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                # else: # added
                #     predict_output = net.decode(decoder_inputs[:, :, :, -2:], encoder_output,
                #                                 torch.cat((encoder_inputs[:,:,-1:,:2],predict_output[:, :, :, :]),axis=-2))
                # if step < predict_length - 1:
                #     decoder_input_list = [decoder_start_inputs,
                #                           torch.cat((predict_output, decoder_pump_inputs[:, :, 1:step + 2, :]), dim=3)]
                if step == 0:  # added
                    predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output, encoder_inputs[:, :, -1:,
                                                                                            :2])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                else:  # added
                    predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :2], predict_output[:, :, :, :]),
                                           axis=-2))

            c_r = criterion(predict_output[:,:,:,1], labels[:,:,:,1])
            h_r = criterion(predict_output[:,:,:,0], labels[:,:,:,0])
            a = torch.nn.Softmax(dim=1)
            weight_s = a(labels[:, :, :, 1])
            num_of_vertices =  labels[:, :, :, 0].shape[1]
            num_of_vertices_w =num_of_vertices
            # loss = criterion(predict_output[:,:,:,0], labels[:,:,:,0]) +  5* criterion(predict_output[:,:,:,1], labels[:,:,:,1])  # 计算误差
            # loss = criterion(num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 0]),
            #           num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 0])) \
            # + 5 * criterion(num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 1]),
            #                 num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 1]))
            # num_of_vertices_w = num_of_vertices
            decoder_time_pumps = torch.count_nonzero(decoder_inputs[:, :, :, 3] + 1, dim=1)
            weight_cf = torch.ones_like(predict_output[:, :, :, 0])
            if (decoder_time_pumps == 2).nonzero(as_tuple=False).shape[0] != 0:
                # print('here')
                for i in (decoder_time_pumps == 2).nonzero(as_tuple=False):
                    weight_cf[i[0], :, i[1]:i[1]+5]=5
            spatial_weighted_pred1 = ( num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 0]))
            spatial_weighted_pred2 = ( num_of_vertices_w * torch.mul(weight_s,predict_output[:,:, :, 1]))

            spatial_weighted_taget1 =(num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 0]))
            spatial_weighted_taget2 =(num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 1]))
            loss = criterion(torch.mul(weight_cf,predict_output[:, :, :, 0]),torch.mul(weight_cf,  labels[:, :, :, 0])) \
                   + 5* criterion(torch.mul(weight_cf,predict_output[:, :, :, 1]),torch.mul(weight_cf, labels[:, :, :, 1]))

            # loss = criterion( predict_output[:, :, :, 0],labels[:, :, :, 0]) \
            #        + 5 * criterion( predict_output[:, :, :, 1],
            #                         labels[:, :, :, 1])

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f' % ( \
                    batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item()))
                # print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type, index_adj,nrow,ncol):
    '''
    for transformerGCN
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        target_comp = []

        input = []  # 存储所有batch的input

        start_time = time()

        for batch_index, batch_data in enumerate(data_loader):
            # if batch_index ==87 or batch_index ==88:

                encoder_inputs, decoder_inputs, labels = batch_data

                encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

                decoder_inputs = decoder_inputs.transpose(-1,
                                                          -2)  # decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1) ->(B, N, T, F)

                # labels = labels.unsqueeze(-1)  # (B, N, T, 1)

                # predict_length = labels.shape[2]  # T
                predict_length = labels.shape[-1]  # T
                labels = labels.transpose(-1, -2)
                # encode
                encoder_output = net.encode(encoder_inputs)
                input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

                # decode
                # decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_input_list = [decoder_start_inputs]
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
                #
                # # 按着时间步进行预测
                # # for step in range(predict_length):
                # #     decoder_inputs = torch.cat(decoder_input_list, dim=2)
                # #     predict_output = net.decode(decoder_inputs, encoder_output)
                # #     decoder_input_list = [decoder_start_inputs, predict_output]
                # for step in range(predict_length):
                #     decoder_inputs = torch.cat(decoder_input_list, dim=2)
                #     predict_output = net.decode(decoder_inputs, encoder_output)
                #     if step < predict_length-1:
                #         decoder_input_list = [decoder_start_inputs, torch.cat((predict_output,decoder_pump_inputs[:,:,1:step+2,:]),dim=3)]
                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
                decoder_pump_inputs = decoder_inputs[:, :, :, 2:]  # 2 features
                decoder_input_list = [decoder_start_inputs[:,:,:,-2:]]
                # 按着时间步进行预测
                # for step in range(predict_length):
                #     decoder_inputs = torch.cat(decoder_input_list, dim=2)
                #     predict_output = net.decode(decoder_inputs[:, :, :, 2:], encoder_output, encoder_inputs[:, :, -1:, :2])
                #     # predict_output = net.decode(decoder_inputs, encoder_output)
                #     if step < predict_length - 1:
                #         decoder_input_list = [decoder_start_inputs,
                #                               torch.cat((predict_output, decoder_pump_inputs[:, :, 1:step + 2, :]), dim=3)]
                for step in range(predict_length):
                #     decoder_inputs = torch.cat(decoder_input_list, dim=2)
                #     if step == 0:  # added
                #         predict_output = net.decode(decoder_inputs[:, :, :, -2:], encoder_output, encoder_inputs[:, :, -1:,
                #                                                                                 :2])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                #     else:  # added
                #         predict_output = net.decode(decoder_inputs[:, :, :, -2:], encoder_output,
                #                                     torch.cat((encoder_inputs[:, :, -1:, :2], predict_output[:, :, :, :]),
                #                                               axis=-2))
                #     if step < predict_length - 1:
                #         decoder_input_list = [decoder_start_inputs,
                #                               torch.cat((predict_output, decoder_pump_inputs[:, :, 1:step + 2, :]), dim=3)]
                    if step == 0:  # added
                        predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output,
                                                    encoder_inputs[:, :, -1:,
                                                    :2])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    else:  # added
                        predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output,
                                                    torch.cat((encoder_inputs[:, :, -1:, :2], predict_output[:, :, :, :]),
                                                              axis=-2))

                prediction.append(predict_output.detach().cpu().numpy())
                target_comp.append(labels.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, :, 0], _min[0, 0, :, 0])

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        # data_target_tensor = np.transpose(data_target_tensor, (0, 1, 3, 2))
        # data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        target_comp = np.concatenate(target_comp, 0)#np.transpose(target_comp, (0, 1, 3, 2))
        target_comp = re_max_min_normalization(target_comp, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])


        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert target_comp.shape[0] == prediction.shape[0] #data_target_tensor.shape
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(target_comp[:, :, i, :].flatten(), prediction[:, :, i, :].flatten())
            rmse = mean_squared_error(target_comp[:, :, i, :].flatten(), prediction[:, :, i, :].flatten()) ** 0.5
            mape = masked_mape_np(target_comp[:, :, i, :], prediction[:, :, i, :], 0)
            # mae = mean_absolute_error(data_target_tensor[:, :, i, :].flatten(), prediction[:, :, i, :].flatten())
            # rmse = mean_squared_error(data_target_tensor[:, :, i, :].flatten(), prediction[:, :, i, :].flatten()) ** 0.5
            # mape = masked_mape_np(data_target_tensor[:, :, i, :], prediction[:, :, i, :], 0)
            # mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            # rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            # mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        # mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        # rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        # mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        mae = mean_absolute_error(target_comp.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(target_comp.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(target_comp.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

    #     time_view = 20
    #     pred_wholemap = np.nan * np.ones((nrow,ncol,prediction.shape[2]))
    #     tag_wholemap = np.nan * np.ones((nrow, ncol,prediction.shape[2]))
    #     pred = prediction[time_view,:,:,1]
    #     target = target_comp[time_view,:,:,1] #data_target_tensor
    #     for key in index_adj.keys():
    #         pred_wholemap[index_adj[key][0],index_adj[key][1],:] = pred[key,:]
    #         tag_wholemap[index_adj[key][0], index_adj[key][1],:] = target[key,:]
    #     np.savetxt('71n_ASTGNN_no_tempo_attention_w_STemb_contaim1_pred.txt',
    #                pred,
    #                delimiter=',') #  pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
    #     np.savetxt('71n_ASTGNN_no_tempo_attention_w_STemb_contaim1_true.txt',
    #                target,
    #                delimiter=',') # tag_wholemap.reshape((tag_wholemap.shape[0] * tag_wholemap.shape[1], tag_wholemap.shape[2]))
    #     # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
    #     # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
    #     from matplotlib import pyplot as plt
    #     plt.ion()
    #     fig, ax = plt.subplots(3, int(pred_wholemap.shape[2]//5), figsize=(20, 3))
    #     for i in range(0,pred_wholemap.shape[2]//5):
    #         im1 = ax[0][i].imshow(pred_wholemap[ :,:, 40+i],vmin=0,vmax=3)#15 ,vmin=0,vmax=2
    #         im2 = ax[1][i].imshow(tag_wholemap[ :,:, 40+i],vmin=0,vmax=3)#,vmin=0,vmax=1 ,vmin=0,vmax=2
    #         im3 = ax[2][i].imshow(
    #             pred_wholemap[:, :, 40+i] -  tag_wholemap[:, :, 40+i], vmin=-3, vmax=3,cmap = 'coolwarm')
    #         # im3 = ax[np.mod(i, 5)][i // 5].imshow(
    #         #    pred_wholemap[:, :,  i], vmin=0, vmax=10)
    #         # im3 = ax[np.mod(i, 5)][i // 5].imshow(
    #         #      tag_wholemap[:, :, i] , vmin=0, vmax=10)#,cmap='PuBu'
    #         # im3 = ax[np.mod(i,5)][i//5].imshow(
    #         #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-2,vmax=2,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
    #     fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
    #     fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
    #     fig.colorbar(im3, ax=ax[2], shrink=0.5, pad=0.01)
    #     plt.ioff()
    #     plt.show()
    # # # # # # # # # # # # # # # # # # # # # return  tag_wholemap,pred_wholemap
    # # # # # # # # # # # # # # # # # # # # #
    #     encoder_inputs = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view][0], 1, 2), 0)
    #     decoder_input_view =torch.unsqueeze( torch.transpose(data_loader.dataset[time_view][1], 1, 2), 0)
    #     decoder_pump_inputs = decoder_input_view[:, :, :, 2:]
    #     encoder_output = net.encode(encoder_inputs)
    #     for step in range(predict_length):
    #         if step == 0:  # added
    #             predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output,
    #                                         encoder_inputs[:, :, -1:,
    #                                         :2])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
    #         else:  # added
    #             predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output,
    #                                         torch.cat((encoder_inputs[:, :, -1:, :2], predict_output[:, :, :, :]),
    #                                                   axis=-2))
    #
    #     time_view1 = time_view+pred_wholemap.shape[-1]
    #     encoder_input_view = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][0], 1,2), 0)
    #     encoder_input_view_pump = encoder_input_view[:,:,:,-2:]
    #     # predict_output = torch.unsqueeze(predict_output,0)
    #     encoder_input_view_pump1 = torch.cat((predict_output[:,:,-encoder_input_view_pump.shape[-2]:,:],encoder_input_view_pump),axis=-1) #
    #     encoder_output = net.encode(encoder_input_view_pump1)
    #     decoder_input_view = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][1], 1,2), 0)
    #     decoder_pump_inputs = decoder_input_view[:, :, :, 2:]
    #     for step in range(predict_length):
    #         if step == 0:  # added
    #             predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output,
    #                                         encoder_input_view_pump1[:, :, -1:,
    #                                         :2])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
    #         else:  # added
    #             predict_output = net.decode(decoder_pump_inputs[:, :, 0:step + 1, -2:], encoder_output,
    #                                         torch.cat((encoder_input_view_pump1[:, :, -1:, :2], predict_output[:, :, :, :]),
    #                                                   axis=-2))
    #     labels = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][2], 1, 2),0)#[:,:,-encoder_input_view_pump.shape[-2]:,:]
    #
    #     # prediction =re_max_min_normalization(torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][0], 1,2), 0)[:,:,:,:2].cpu().numpy(), _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
    #     prediction1 = re_max_min_normalization(predict_output.cpu().numpy(), _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
    #
    #     target_comp1 = re_max_min_normalization(labels.cpu().numpy(), _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
    #
    #     pred_wholemap = np.nan * np.ones((nrow, ncol, prediction1.shape[2]))
    #     tag_wholemap = np.nan * np.ones((nrow, ncol, prediction1.shape[2]))
    #     pred = prediction1[ 0,:, :, 1]
    #     target = target_comp1[0, :, :, 1]  # data_target_tensor
    #     np.savetxt('71n_ASTGNN_no_tempo_attention_w_STemb_contaim1_ext_pred.txt',
    #                pred,
    #                delimiter=',') #  pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
    #     np.savetxt('71n_ASTGNN_no_tempo_attention_w_STemb_contaim1_ext_true.txt',
    #                target,
    #                delimiter=',')
    #     for key in index_adj.keys():
    #         pred_wholemap[index_adj[key][0], index_adj[key][1], :] = pred[key, :]
    #         tag_wholemap[index_adj[key][0], index_adj[key][1], :] = target[key, :]
    #     # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
    #     # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
    #     from matplotlib import pyplot as plt
    #     fig, ax = plt.subplots(3, int(pred_wholemap.shape[2] // 5), figsize=(20, 3))
    #     plt.ion()
    #     for i in range(0, pred_wholemap.shape[2]//5):
    #         im1 = ax[0][i].imshow(pred_wholemap[ :,:, 40+i],vmin=0,vmax=5)#15 ,vmin=0,vmax=2
    #         im2 = ax[1][i].imshow(tag_wholemap[ :,:, 40+i],vmin=0,vmax=5)#,vmin=0,vmax=1 ,vmin=0,vmax=2
    #         im3 = ax[2][i].imshow(
    #             pred_wholemap[:, :, 40+i] -  tag_wholemap[:, :, 40+i], vmin=-3, vmax=3,cmap = 'coolwarm')
    #         # im3 = ax[np.mod(i, 4)][i // 4].imshow(
    #         #     tag_wholemap[:, :, i], vmin=0, vmax=2)
    #         # im3 = ax[np.mod(i, 5)][i // 5].imshow(
    #         #      pred_wholemap[:, :, i] , vmin=0, vmax=5)#,cmap='PuBu'
    #         # im3 = ax[np.mod(i,5)][i//5].imshow(
    #         #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-3,vmax=3,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
    #     fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
    #     fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
    #     fig.colorbar(im3, ax=ax[2], shrink=0.5, pad=0.01)
    #     plt.ioff()
    #
    #     plt.show()

        #
        # label_view = data_loader.dataset[time_view][2]
        # feature_id = 0
        # view_feature = decoder_input_view#torch.transpose(encoder_input_view, 2,3)[0,:,:,:]
        # f1_wholemap = np.nan * np.ones((nrow, ncol, view_feature.shape[2]))
        # f2_wholemap = np.nan * np.ones((nrow, ncol, view_feature.shape[2]))
        # f3_wholemap = np.nan * np.ones((nrow, ncol, view_feature.shape[2]))
        # f4_wholemap = np.nan * np.ones((nrow, ncol, view_feature.shape[2]))
        #
        # f1 = view_feature[ :, 0, :].cpu().detach().numpy()
        # f2 = view_feature[:, 1, :].cpu().detach().numpy()
        # f3 = view_feature[ :, 2, :].cpu().detach().numpy()
        # f4 = view_feature[:, 3, :].cpu().detach().numpy()
        #
        # # target = data_target_tensor[700, :, :, 0]
        # for key in index_adj.keys():
        #     f1_wholemap[index_adj[key][0], index_adj[key][1], :] = f1[key, :]
        #     f2_wholemap[index_adj[key][0], index_adj[key][1], :] = f2[key, :]
        #     f3_wholemap[index_adj[key][0], index_adj[key][1], :] = f3[key, :]
        #     f4_wholemap[index_adj[key][0], index_adj[key][1], :] = f4[key, :]
        # # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
        # # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
        # fig, ax = plt.subplots(4, 10, figsize=(20, 4)) #pred_wholemap.shape[2]
        # for i in range(10):#pred_wholemap.shape[2]):
        #     im1 = ax[0][i].imshow(f1_wholemap[:, :, i+10],vmin=-1,vmax=1)
        #     im2 = ax[1][i].imshow(f2_wholemap[:, :, i+10],vmin=-1,vmax=-0.8)
        #     im3 = ax[2][i].imshow(f3_wholemap[:, :, i+10],vmin=-1,vmax=1)
        #     im4 = ax[3][i].imshow(f4_wholemap[:, :, i+10],vmin=-1,vmax=0)
        #
        # fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
        # fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
        # fig.colorbar(im3, ax=ax[2], shrink=0.5, pad=0.01)
        # fig.colorbar(im4, ax=ax[3], shrink=0.5, pad=0.01)
        # plt.show()


def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_decoder_x= file_data['train_decoder_x']
    #train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    val_decoder_x = file_data['val_decoder_x']
    #val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_decoder_x = file_data['test_decoder_x']
    #test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0:2, :], _min[:, :, 0:2, :]) #0:2 as two target to predict
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0:2, :], _min[:, :, 0:2, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0:2, :], _min[:, :, 0:2, :])

    #  ------- train_loader -------
    # train_decoder_input_start = train_x[:, :, 0:1, -1:]
    train_decoder_input_start = train_x[:, :, 0:2, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input_p = np.concatenate((train_decoder_input_start, train_target_norm[:, :,:, :-1]), axis=3)
    # train_decoder_input_p = np.concatenate((train_decoder_input_start, train_target_norm[:, :,:-1]), axis=2) # axis=2 # (B, N, T)
    # train_decoder_input = np.concatenate((np.expand_dims(train_decoder_input_p, axis=-2), train_decoder_x), axis=-2) #-> (B, N, F,T)
    train_decoder_input = np.concatenate((train_decoder_input_p, train_decoder_x), axis=-2)
    #  (B,N,T,F(2)) in pump and drawdown

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)-> (B, N, F,T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#shuffle

    #  ------- val_loader -------
    # val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = val_x[:, :, 0:2, -1:]
    # val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    # val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T) -> (B, N, F,T)
    val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :,:, :-1]), axis=3)
    # val_decoder_input = np.concatenate((np.expand_dims(val_decoder_input_p,axis=-2),val_decoder_x),axis=-2)
    val_decoder_input = np.concatenate((val_decoder_input_p,val_decoder_x), axis=-2)
    #  (B,N,T,F(2)) in pump and drawdown

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T) -> (B, N, F,T)
    #  (B,N,T,F(2)) in pump and drawdown
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    # test_decoder_input_start = test_x[:, :, 0:1, -1:]
    test_decoder_input_start = test_x[:, :, 0:2, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    # test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)
    test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :,:, :-1]), axis=3)  # (B, N, T)
    # test_decoder_input = np.concatenate((np.expand_dims(test_decoder_input_p,axis=-2),test_decoder_x),axis=-2)
    test_decoder_input = np.concatenate((test_decoder_input_p, test_decoder_x), axis=-2)
    #  (B,N,T,F(2)) in pump and drawdown

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  #(B, N, T) -> (B, N, F,T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor,test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor,train_timestamp, val_loader, val_target_tensor, val_timestamp,test_loader, test_target_tensor,test_timestamp,  _max, _min


def load_graphdata_normY_channel1_o(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)
    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0:2, :], _min[:, :, 0:2, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0:2, :], _min[:, :, 0:2, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :,0:2, :], _min[:, :, 0:2, :])

    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min
