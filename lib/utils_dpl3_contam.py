import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from time import time
from scipy.sparse.linalg import eigs
import torch.nn as nn

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

def compute_val_loss(net, val_loader, criterion, sw,decoder_dim, epoch):
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
            pump_dim = decoder_dim#int(decoder_inputs.shape[-1]/2)
            decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features
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
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output, encoder_inputs[:, :, -1:,
                                                                                            :pump_dim])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                else:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                                                                                        predict_output ),axis=-2)) #torch.cat((predict_output[:, :, :, :2],predict_output[:, :, :, 3:4]),axis=-1)

            c_r = criterion(predict_output[:,:,:,1], labels[:,:,:,1])
            h_r = criterion(predict_output[:,:,:,0], labels[:,:,:,0])
            # c_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
            # h_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
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
            decoder_time_pumps = torch.count_nonzero(decoder_inputs[:, :, :, pump_dim] + 1, dim=1)
            weight_cf = torch.ones_like(predict_output[:, :, :, 0])
            if (decoder_time_pumps == 2).nonzero(as_tuple=False).shape[0] != 0:
                # print('here')
                for i in (decoder_time_pumps == 2).nonzero(as_tuple=False):
                    weight_cf[i[0], :, i[1]:i[1]+5]=5
            spatial_weighted_pred1 = ( num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 0]))
            spatial_weighted_pred2 = ( num_of_vertices_w * torch.mul(weight_s,predict_output[:,:, :, 1]))

            spatial_weighted_taget1 =(num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 0]))
            spatial_weighted_taget2 =(num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 1]))
            # loss = criterion(torch.mul(weight_cf,predict_output[:, :, :, 0]),torch.mul(weight_cf,  labels[:, :, :, 0])) \
            #        + 5* criterion(torch.mul(weight_cf,predict_output[:, :, :, 1]),torch.mul(weight_cf, labels[:, :, :, 1]))
            loss = criterion(predict_output[:, :, :, 0], labels[:, :, :, 0]) \
                   + 5* criterion( predict_output[:, :, :, 1],
                                   labels[:, :, :, 1]) \
                   # + 5*criterion(predict_output[:, :, :, 2],
                   #                 labels[:, :, :, 2])
                    # +criterion(predict_output[:, :, :, 2], labels[:, :, :, 2]) \

            # loss = criterion( predict_output[:, :, :, 0],labels[:, :, :, 0]) \
            #        + 5 * criterion( predict_output[:, :, :, 1],
            #                         labels[:, :, :, 1])

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f' % ( \
                    batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item()))#l_hr1: %.4f, h_r1.item()))
                # print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f, c_r1: %.4f' % ( \
                #     batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item(), c_r1.item()))#l_hr1: %.4f, h_r1.item()))
                # print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss

def compute_val_loss_pdemask(net, val_loader, criterion, sw,decoder_dim, epoch,mask,mask_output,DEVICE):
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

            # x = encoder_inputs[:,:,:,2]
            # y = encoder_inputs[:,:,:,3]
            # t = encoder_inputs[:,:,:,4]
            log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :, 3]

            dim_encode = encoder_inputs[:, :, :, 4].shape[2]
            dim_decode = decoder_inputs[:, :, :, 4].shape[2]
            encoder_time = torch.tensor(
                [[list(range(dim_encode))] * encoder_inputs.shape[1]] * encoder_inputs.shape[0]).cuda()
            decoder_time = torch.tensor(
                [[list(range(dim_decode))] * decoder_inputs.shape[1]] * decoder_inputs.shape[0]).cuda()
            # encode
            encoder_output = net.encode(encoder_inputs[:, :, :, [0, 1, 5, 6]], encoder_inputs[:, :, :, 2],
                                        encoder_inputs[:, :, :, 3], encoder_time, mask)  # [:,:,:,[0,1,5,6]]
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
            pump_dim = decoder_dim  # int(decoder_inputs.shape[-1]/2)
            decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features
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
                # (self, trg, encoder_output, encoder_input, x, y, t)
                if step == 0:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output,
                                                 encoder_inputs[:, :, -1:,
                                                 :pump_dim], decoder_pump_inputs[:, :, 0:step + 1, 0],
                                                 decoder_pump_inputs[:, :, 0:step + 1, 1], decoder_time[:, :,
                                                                                           0:step + 1],mask)  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                else:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output,
                                                 torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                            predict_output), axis=-2), \
                                                 decoder_pump_inputs[:, :, 0:step + 1, 0],
                                                 decoder_pump_inputs[:, :, 0:step + 1,
                                                 1], decoder_time[:, :, 0:step + 1],mask
                                                 )  # torch.cat((predict_output[:, :, :, :2],predict_output[:, :, :, 3:4]),axis=-1)

            c_r = criterion(predict_output[:, :, :, 1]*mask_output, labels[:, :, :, 1]*mask_output)
            h_r = criterion(predict_output[:, :, :, 0]*mask_output, labels[:, :, :, 0]*mask_output)
            # c_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
            # h_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
            a = torch.nn.Softmax(dim=1)
            weight_s = a(labels[:, :, :, 1])
            num_of_vertices = labels[:, :, :, 0].shape[1]
            num_of_vertices_w = num_of_vertices
            # loss = criterion(predict_output[:,:,:,0], labels[:,:,:,0]) +  5* criterion(predict_output[:,:,:,1], labels[:,:,:,1])  # 计算误差
            # loss = criterion(num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 0]),
            #           num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 0])) \
            # + 5 * criterion(num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 1]),
            #                 num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 1]))
            # num_of_vertices_w = num_of_vertices
            decoder_time_pumps = torch.count_nonzero(decoder_inputs[:, :, :, pump_dim] + 1, dim=1)
            weight_cf = torch.ones_like(predict_output[:, :, :, 0])
            if (decoder_time_pumps == 2).nonzero(as_tuple=False).shape[0] != 0:
                # print('here')
                for i in (decoder_time_pumps == 2).nonzero(as_tuple=False):
                    weight_cf[i[0], :, i[1]:i[1] + 5] = 5
            spatial_weighted_pred1 = (num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 0]))
            spatial_weighted_pred2 = (num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 1]))

            spatial_weighted_taget1 = (num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 0]))
            spatial_weighted_taget2 = (num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 1]))
            # loss = criterion(torch.mul(weight_cf,predict_output[:, :, :, 0]),torch.mul(weight_cf,  labels[:, :, :, 0])) \
            #        + 5* criterion(torch.mul(weight_cf,predict_output[:, :, :, 1]),torch.mul(weight_cf, labels[:, :, :, 1]))

            # from torch.autograd import Variable
            # pt_x_collocation = Variable(torch.from_numpy(log1.cpu().detach().numpy()).float(), requires_grad=True).to(DEVICE)
            # pt_y_collocation = Variable(torch.from_numpy(lat1.cpu().detach().numpy()).float(), requires_grad=True).to(DEVICE)
            # pt_t_collocation = Variable(torch.from_numpy(decoder_time.cpu().detach().numpy()).float(), requires_grad=True).to(DEVICE)
            # pt_all_zeros = torch.from_numpy(np.zeros((pt_x_collocation.shape[0],pt_x_collocation.shape[1],\
            #                                           pt_x_collocation.shape[2]))).float().requires_grad_(True).to(DEVICE)#Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(DEVICE)
            #
            #
            # def f(x, y, t, net):
            #     u = net(encoder_inputs[:,:,:,[0,1,5,6]], decoder_inputs[:,:,:,[0,1,5,6]],x,y,encoder_time,t)  # the dependent variable u is given by the network based on independent variables x,t
            #     ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
            #     u_x = torch.autograd.grad(u[:,:,:,0], x, create_graph=True,grad_outputs=torch.ones_like(x))[0]
            #     u_y = torch.autograd.grad(u[:, :, :, 0], y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #                               )[0]
            #     u_t = torch.autograd.grad(u[:, :, :, 0], t, create_graph=True, grad_outputs=torch.ones_like(t))[0]
            #     W = decoder_inputs[:,:,:,5]
            #     dim_space = u_x.shape[1]
            #     dh_lx1 = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space,1,bias=False).cuda()
            #     dh_ly1 = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space, 1, bias=False).cuda()
            #     dh1x = dh_lx1*u_x#(torch.unsqueeze(u_x.transpose(1,2),,-1))
            #     dh1y = dh_ly1*u_y#(torch.unsqueeze(u_y,-1))
            #
            #     u2_x = torch.autograd.grad(torch.squeeze(dh1x,-1), x, create_graph=True, grad_outputs=torch.ones_like(x))[0].detach()
            #     u2_y = \
            #     torch.autograd.grad(torch.squeeze(dh1y, -1), y, create_graph=True, grad_outputs=torch.ones_like(y))[0].detach()
            #
            #     g_e = nn.Linear(2,1,bias=False).cuda()
            #     p_ddh_lx_w_t = torch.concat((\
            #     torch.unsqueeze(W, -1),\
            #     torch.unsqueeze(u_t, -1)),axis=-1)
            #     torch.squeeze(g_e(p_ddh_lx_w_t), -1)
            #     pde_h =u2_x+u2_y+torch.squeeze(g_e(p_ddh_lx_w_t),-1) #+u2_y* list(g_e.parameters())[0].T[0,:]
            #
            #     # c_x = torch.autograd.grad(u[:,:,:,1], x, create_graph=True,grad_outputs=torch.ones_like(x))[0]
            #     # c_y = torch.autograd.grad(u[:, :, :, 1], y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #     #                           )[0]
            #     # # dc_lxx = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space,1,bias=False).cuda()
            #     # # dc_lyy = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()
            #     # # dc1xx = dc_lxx*c_x#(torch.unsqueeze(u_x.transpose(1,2),,-1))
            #     # # dc1yy = dc_lyy*c_y#(torch.unsqueeze(u_y,-1))
            #     # #
            #     # # dc_lxy = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space,1,bias=False).cuda()
            #     # # dc_lyx = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()
            #     # # dc1xy = dc_lxy*c_x#(torch.unsqueeze(u_x.transpose(1,2),,-1))
            #     # # dc1yx = dc_lyx*c_y#(torch.unsqueeze(u_y,-1))
            #     #
            #     # diffusion_l1 = nn.Linear(dim_space*2,10).cuda()#
            #     # diffusion_l2 = nn.Linear(10, dim_space * 6).cuda()#
            #     # diffu_hidden = diffusion_l1(torch.concat((u_x.transpose(1,2),u_y.transpose(1,2)),-1))
            #     # diffusion_coef_and_velocity = diffusion_l2(diffu_hidden)
            #     #
            #     # dc1xx= diffusion_coef_and_velocity[:,:,0:dim_space].transpose(1,2)*c_x
            #     # dc1yy = diffusion_coef_and_velocity[:, :, dim_space:2*dim_space].transpose(1, 2) * c_y
            #     # dc1xy= diffusion_coef_and_velocity[:,:,2*dim_space:3*dim_space].transpose(1,2)*c_x
            #     # dc1yx = diffusion_coef_and_velocity[:, :, 3*dim_space:4*dim_space].transpose(1, 2) * c_y
            #     # dv1x = diffusion_coef_and_velocity[:, :, 4 * dim_space:5 * dim_space].transpose(1, 2) #* u[:,:,:,1]
            #     # dv1y = diffusion_coef_and_velocity[:, :, 5 * dim_space:6 * dim_space].transpose(1, 2) #* u[:,:,:,1]
            #     #
            #     # c_xx = torch.autograd.grad(dc1xx, x, create_graph=True,grad_outputs=torch.ones_like(x))[0].detach()
            #     # # print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
            #     # # print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
            #     # # print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
            #     # c_xy = torch.autograd.grad(dc1xy, y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #     #                           )[0].detach()
            #     # c_yy = torch.autograd.grad(dc1yy, y, create_graph=True,grad_outputs=torch.ones_like(y))[0].detach()
            #     # c_yx = torch.autograd.grad(dc1yx, x, create_graph=True, grad_outputs=torch.ones_like(x)\
            #     #                           )[0].detach()
            #     # v_x = torch.autograd.grad(dv1x, x, create_graph=True, grad_outputs=torch.ones_like(x)\
            #     #                           )[0].detach()
            #     # v_y = torch.autograd.grad(dv1y, y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #     #                           )[0].detach()
            #     # g_qc = nn.Linear(1,1,bias=False).cuda()
            #     # qC =  u[:,:,:,0]*u[:,:,:,1]
            #     # qC_term = torch.squeeze(g_qc(torch.unsqueeze(qC,-1)),-1)
            #     #
            #     # c_t = torch.autograd.grad(u[:, :, :, 1], t, create_graph=True, grad_outputs=torch.ones_like(t))[0].detach()
            #     #
            #     # pde_c = c_xx+c_xy+c_yy+c_yx-v_x-c_x-v_y-c_y+qC_term-c_t
            #
            #     # pde_h = u_x - 2 * u_t - u
            #     # pde_c = u_x - 2 * u_t - u
            #     return pde_h#,pde_c
            #
            # pde_h = f(pt_x_collocation,pt_y_collocation, pt_t_collocation, net)
            # loss_hpdf = criterion(pde_h, pt_all_zeros)

            loss = criterion(predict_output[:, :, :, 0]*mask_output, labels[:, :, :, 0]*mask_output) \
                   + 5 * criterion(predict_output[:, :, :, 1]*mask_output,
                                   labels[:, :, :, 1]*mask_output) \
                # + 5*criterion(predict_output[:, :, :, 2],
            #                 labels[:, :, :, 2])
            # +criterion(predict_output[:, :, :, 2], labels[:, :, :, 2]) \

            # loss = criterion( predict_output[:, :, :, 0],labels[:, :, :, 0]) \
            #        + 5 * criterion( predict_output[:, :, :, 1],
            #                         labels[:, :, :, 1])

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f' % ( \
                    batch_index + 1, val_loader_length, loss.item(), c_r.item(),
                    h_r.item()))  # l_hr1: %.4f, h_r1.item()))
                # print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f, c_r1: %.4f' % ( \
                #     batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item(), c_r1.item()))#l_hr1: %.4f, h_r1.item()))
                # print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' % (time() - start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss
def compute_val_loss_pde(net, val_loader, criterion, sw,decoder_dim, epoch,DEVICE):
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

            # x = encoder_inputs[:,:,:,2]
            # y = encoder_inputs[:,:,:,3]
            # t = encoder_inputs[:,:,:,4]
            log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]

            dim_encode = encoder_inputs[:,:,:,4].shape[2]
            dim_decode = decoder_inputs[:,:,:,4].shape[2]
            encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
            decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
            # encode
            encoder_output = net.encode(encoder_inputs[:, :, :, [0,1,5,6]], encoder_inputs[:, :, :, 2], encoder_inputs[:, :, :, 3],encoder_time) #[:,:,:,[0,1,5,6]]
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
            pump_dim = decoder_dim#int(decoder_inputs.shape[-1]/2)
            decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features
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
                # (self, trg, encoder_output, encoder_input, x, y, t)
                if step == 0:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, encoder_inputs[:, :, -1:,
                                                                                            :pump_dim],decoder_pump_inputs[:, :, 0:step + 1, 0],decoder_pump_inputs[:, :, 0:step + 1, 1],decoder_time[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                else:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                                                                                        predict_output ),axis=-2), \
                         decoder_pump_inputs[:, :, 0:step + 1, 0], decoder_pump_inputs[:, :, 0:step + 1,
                                                                    1], decoder_time[:, :, 0:step + 1]
                                                 ) #torch.cat((predict_output[:, :, :, :2],predict_output[:, :, :, 3:4]),axis=-1)

            c_r = criterion(predict_output[:,:,:,1], labels[:,:,:,1])
            h_r = criterion(predict_output[:,:,:,0], labels[:,:,:,0])
            # c_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
            # h_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
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
            decoder_time_pumps = torch.count_nonzero(decoder_inputs[:, :, :, pump_dim] + 1, dim=1)
            weight_cf = torch.ones_like(predict_output[:, :, :, 0])
            if (decoder_time_pumps == 2).nonzero(as_tuple=False).shape[0] != 0:
                # print('here')
                for i in (decoder_time_pumps == 2).nonzero(as_tuple=False):
                    weight_cf[i[0], :, i[1]:i[1]+5]=5
            spatial_weighted_pred1 = ( num_of_vertices_w * torch.mul(weight_s, predict_output[:, :, :, 0]))
            spatial_weighted_pred2 = ( num_of_vertices_w * torch.mul(weight_s,predict_output[:,:, :, 1]))

            spatial_weighted_taget1 =(num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 0]))
            spatial_weighted_taget2 =(num_of_vertices_w * torch.mul(weight_s, labels[:, :, :, 1]))
            # loss = criterion(torch.mul(weight_cf,predict_output[:, :, :, 0]),torch.mul(weight_cf,  labels[:, :, :, 0])) \
            #        + 5* criterion(torch.mul(weight_cf,predict_output[:, :, :, 1]),torch.mul(weight_cf, labels[:, :, :, 1]))

            # from torch.autograd import Variable
            # pt_x_collocation = Variable(torch.from_numpy(log1.cpu().detach().numpy()).float(), requires_grad=True).to(DEVICE)
            # pt_y_collocation = Variable(torch.from_numpy(lat1.cpu().detach().numpy()).float(), requires_grad=True).to(DEVICE)
            # pt_t_collocation = Variable(torch.from_numpy(decoder_time.cpu().detach().numpy()).float(), requires_grad=True).to(DEVICE)
            # pt_all_zeros = torch.from_numpy(np.zeros((pt_x_collocation.shape[0],pt_x_collocation.shape[1],\
            #                                           pt_x_collocation.shape[2]))).float().requires_grad_(True).to(DEVICE)#Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(DEVICE)
            #
            #
            # def f(x, y, t, net):
            #     u = net(encoder_inputs[:,:,:,[0,1,5,6]], decoder_inputs[:,:,:,[0,1,5,6]],x,y,encoder_time,t)  # the dependent variable u is given by the network based on independent variables x,t
            #     ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
            #     u_x = torch.autograd.grad(u[:,:,:,0], x, create_graph=True,grad_outputs=torch.ones_like(x))[0]
            #     u_y = torch.autograd.grad(u[:, :, :, 0], y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #                               )[0]
            #     u_t = torch.autograd.grad(u[:, :, :, 0], t, create_graph=True, grad_outputs=torch.ones_like(t))[0]
            #     W = decoder_inputs[:,:,:,5]
            #     dim_space = u_x.shape[1]
            #     dh_lx1 = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space,1,bias=False).cuda()
            #     dh_ly1 = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space, 1, bias=False).cuda()
            #     dh1x = dh_lx1*u_x#(torch.unsqueeze(u_x.transpose(1,2),,-1))
            #     dh1y = dh_ly1*u_y#(torch.unsqueeze(u_y,-1))
            #
            #     u2_x = torch.autograd.grad(torch.squeeze(dh1x,-1), x, create_graph=True, grad_outputs=torch.ones_like(x))[0].detach()
            #     u2_y = \
            #     torch.autograd.grad(torch.squeeze(dh1y, -1), y, create_graph=True, grad_outputs=torch.ones_like(y))[0].detach()
            #
            #     g_e = nn.Linear(2,1,bias=False).cuda()
            #     p_ddh_lx_w_t = torch.concat((\
            #     torch.unsqueeze(W, -1),\
            #     torch.unsqueeze(u_t, -1)),axis=-1)
            #     torch.squeeze(g_e(p_ddh_lx_w_t), -1)
            #     pde_h =u2_x+u2_y+torch.squeeze(g_e(p_ddh_lx_w_t),-1) #+u2_y* list(g_e.parameters())[0].T[0,:]
            #
            #     # c_x = torch.autograd.grad(u[:,:,:,1], x, create_graph=True,grad_outputs=torch.ones_like(x))[0]
            #     # c_y = torch.autograd.grad(u[:, :, :, 1], y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #     #                           )[0]
            #     # # dc_lxx = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space,1,bias=False).cuda()
            #     # # dc_lyy = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()
            #     # # dc1xx = dc_lxx*c_x#(torch.unsqueeze(u_x.transpose(1,2),,-1))
            #     # # dc1yy = dc_lyy*c_y#(torch.unsqueeze(u_y,-1))
            #     # #
            #     # # dc_lxy = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()#nn.Linear(dim_space,1,bias=False).cuda()
            #     # # dc_lyx = nn.Parameter(torch.randn((dim_space,1), requires_grad=True)).cuda()
            #     # # dc1xy = dc_lxy*c_x#(torch.unsqueeze(u_x.transpose(1,2),,-1))
            #     # # dc1yx = dc_lyx*c_y#(torch.unsqueeze(u_y,-1))
            #     #
            #     # diffusion_l1 = nn.Linear(dim_space*2,10).cuda()#
            #     # diffusion_l2 = nn.Linear(10, dim_space * 6).cuda()#
            #     # diffu_hidden = diffusion_l1(torch.concat((u_x.transpose(1,2),u_y.transpose(1,2)),-1))
            #     # diffusion_coef_and_velocity = diffusion_l2(diffu_hidden)
            #     #
            #     # dc1xx= diffusion_coef_and_velocity[:,:,0:dim_space].transpose(1,2)*c_x
            #     # dc1yy = diffusion_coef_and_velocity[:, :, dim_space:2*dim_space].transpose(1, 2) * c_y
            #     # dc1xy= diffusion_coef_and_velocity[:,:,2*dim_space:3*dim_space].transpose(1,2)*c_x
            #     # dc1yx = diffusion_coef_and_velocity[:, :, 3*dim_space:4*dim_space].transpose(1, 2) * c_y
            #     # dv1x = diffusion_coef_and_velocity[:, :, 4 * dim_space:5 * dim_space].transpose(1, 2) #* u[:,:,:,1]
            #     # dv1y = diffusion_coef_and_velocity[:, :, 5 * dim_space:6 * dim_space].transpose(1, 2) #* u[:,:,:,1]
            #     #
            #     # c_xx = torch.autograd.grad(dc1xx, x, create_graph=True,grad_outputs=torch.ones_like(x))[0].detach()
            #     # # print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
            #     # # print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
            #     # # print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
            #     # c_xy = torch.autograd.grad(dc1xy, y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #     #                           )[0].detach()
            #     # c_yy = torch.autograd.grad(dc1yy, y, create_graph=True,grad_outputs=torch.ones_like(y))[0].detach()
            #     # c_yx = torch.autograd.grad(dc1yx, x, create_graph=True, grad_outputs=torch.ones_like(x)\
            #     #                           )[0].detach()
            #     # v_x = torch.autograd.grad(dv1x, x, create_graph=True, grad_outputs=torch.ones_like(x)\
            #     #                           )[0].detach()
            #     # v_y = torch.autograd.grad(dv1y, y, create_graph=True, grad_outputs=torch.ones_like(y)\
            #     #                           )[0].detach()
            #     # g_qc = nn.Linear(1,1,bias=False).cuda()
            #     # qC =  u[:,:,:,0]*u[:,:,:,1]
            #     # qC_term = torch.squeeze(g_qc(torch.unsqueeze(qC,-1)),-1)
            #     #
            #     # c_t = torch.autograd.grad(u[:, :, :, 1], t, create_graph=True, grad_outputs=torch.ones_like(t))[0].detach()
            #     #
            #     # pde_c = c_xx+c_xy+c_yy+c_yx-v_x-c_x-v_y-c_y+qC_term-c_t
            #
            #     # pde_h = u_x - 2 * u_t - u
            #     # pde_c = u_x - 2 * u_t - u
            #     return pde_h#,pde_c
            #
            # pde_h = f(pt_x_collocation,pt_y_collocation, pt_t_collocation, net)
            # loss_hpdf = criterion(pde_h, pt_all_zeros)

            loss = criterion(predict_output[:, :, :, 0], labels[:, :, :, 0]) \
                   + 5*criterion( predict_output[:, :, :, 1],
                                   labels[:, :, :, 1]) \
                   # + 5*criterion(predict_output[:, :, :, 2],
                   #                 labels[:, :, :, 2])
                    # +criterion(predict_output[:, :, :, 2], labels[:, :, :, 2]) \

            # loss = criterion( predict_output[:, :, :, 0],labels[:, :, :, 0]) \
            #        + 5 * criterion( predict_output[:, :, :, 1],
            #                         labels[:, :, :, 1])

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f' % ( \
                    batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item()))#l_hr1: %.4f, h_r1.item()))
                # print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f, c_r1: %.4f' % ( \
                #     batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item(), c_r1.item()))#l_hr1: %.4f, h_r1.item()))
                # print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss

def compute_val_loss_mlti(net, val_loader, criterion,params, sw, epoch,DEVICE):
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
            decoder_pump_inputs = decoder_inputs[:, :, :, 3:]  # 2 features
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
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -3:], encoder_output, encoder_inputs[:, :, -1:,
                                                                                            :3])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                else:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -3:], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :3], \
                                                                                                                        predict_output ),axis=-2)) #torch.cat((predict_output[:, :, :, :2],predict_output[:, :, :, 3:4]),axis=-1)

            c_r = criterion(predict_output[:,:,:,1], labels[:,:,:,1])
            h_r = criterion(predict_output[:,:,:,0], labels[:,:,:,0])
            c_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
            # h_r1 = criterion(predict_output[:, :, :, 2], labels[:, :, :, 2])
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
            # loss = criterion(torch.mul(weight_cf,predict_output[:, :, :, 0]),torch.mul(weight_cf,  labels[:, :, :, 0])) \
            #        + 5* criterion(torch.mul(weight_cf,predict_output[:, :, :, 1]),torch.mul(weight_cf, labels[:, :, :, 1]))
            loss = params[0].to(DEVICE)*criterion(predict_output[:, :, :, 0], labels[:, :, :, 0]) \
                   + params[1].to(DEVICE)* criterion( predict_output[:, :, :, 1],
                                   labels[:, :, :, 1]) \
                   + params[2].to(DEVICE)*criterion(predict_output[:, :, :, 2],
                                   labels[:, :, :, 2])
                    # +criterion(predict_output[:, :, :, 2], labels[:, :, :, 2]) \

            # loss = criterion( predict_output[:, :, :, 0],labels[:, :, :, 0]) \
            #        + 5 * criterion( predict_output[:, :, :, 1],
            #                         labels[:, :, :, 1])

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f, c_r1: %.4f, w1: %.4f, w2: %.4f, w3: %.4f' % ( \
                    batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item(), c_r1.item(),params[0],\
                    params[1],params[2]))#l_hr1: %.4f, h_r1.item()))
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
                dinput_sh = int(decoder_inputs.shape[-1]/2)
                decoder_pump_inputs = decoder_inputs[:, :, :, dinput_sh:]  # 2 features
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
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                                                    encoder_inputs[:, :, -1:,
                                                    :dinput_sh])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    else:  # added
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                                                    torch.cat((encoder_inputs[:, :, -1:, :dinput_sh], predict_output[:, :, :, :]),
                                                              axis=-2)) #predict_output[:, :, :, [0,1,3]]

                prediction.append(predict_output.detach().cpu().numpy())
                target_comp.append(labels.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, :, 0], _min[0, 0, :, 0])

        max_sh = int(_max.shape[2]/2)
        _max1 = _max[0, 0, 0:max_sh, 0]#np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 0:3:2, 0]),axis=0)
        _min1 = _min[0, 0, 0:max_sh, 0]#np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 0:3:2, 0]),axis=0)

        # _max2 = np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 2:3, 0]),axis=0) #0:3:2
        # _min2 = np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 2:3, 0]),axis=0) #0:3:2

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        # prediction = re_max_min_normalization(prediction,_max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        prediction = re_max_min_normalization(prediction, _max1, _min1)
        # data_target_tensor = np.transpose(data_target_tensor, (0, 1, 3, 2))
        # data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        target_comp = np.concatenate(target_comp, 0)#np.transpose(target_comp, (0, 1, 3, 2))
        # target_comp = re_max_min_normalization(target_comp, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        # target_comp = re_max_min_normalization(target_comp, _max2, _min2)
        target_comp = re_max_min_normalization(target_comp, _max1, _min1)


        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        tg_list = list(range(max_sh))
        for i in range(prediction_length):
            assert target_comp.shape[0] == prediction.shape[0] #data_target_tensor.shape
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten())
            rmse = mean_squared_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten()) ** 0.5
            mape = masked_mape_np(target_comp[:, :, i, tg_list], prediction[:, :, i, :], 0)
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
        mae = mean_absolute_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

        # time_view = 0#55#55
        # ind_c =1
        # if ind_c==1:
        #     vmax_p = 10
        #     v_error = 1.5#0.5
        # else:
        #     vmax_p = 3
        #     v_error = 0.5#0.3
        # workdir  ='results\\'
        # time_view = 0#55
        # # scenario = 'difficult_no_emb'
        # # scenario = 'difficult_inductive'#t
        # scenario = 'difficult'#_inductive'
        # # scenario = 'easy_inductive'
        # monitor_sys = '71n'
        # train_test = 'test'
        # ind_c = 1
        # if ind_c==1:
        #     vmax_p = 10
        #     v_error = 1#0.5
        # else:
        #     vmax_p = 3
        #     v_error = 0.5#0.3
        # pred_wholemap = np.nan * np.ones((nrow,ncol,prediction.shape[2]))
        # tag_wholemap = np.nan * np.ones((nrow, ncol,prediction.shape[2]))
        # pred = prediction[time_view,:,:,ind_c]
        # target = target_comp[time_view,:,:,ind_c] #data_target_tensor
        # np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_pred.npz',
        #            prediction)
        # np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_true.npz',
        #            target_comp)
        # pred_wholemap = np.nan * np.ones((nrow,ncol,prediction.shape[2]))
        # tag_wholemap = np.nan * np.ones((nrow, ncol,prediction.shape[2]))
        # pred = prediction[time_view,:,:,ind_c]
        # target = target_comp[time_view,:,:,ind_c] #data_target_tensor
        # for key in index_adj.keys():
        #     pred_wholemap[index_adj[key][0],index_adj[key][1],:] = pred[key,:]
        #     tag_wholemap[index_adj[key][0], index_adj[key][1],:] = target[key,:]
        # # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
        # # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
        # # np.savetxt('71n_no_temporal_atte_ASTGNN_w_STemb_contaim2-d'+str(time_view)+'_'+str(ind_c)+'pred.txt',
        # #            pred,
        # #            delimiter=',') # no_emb_no_temporal_atte_ pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
        # # np.savetxt('71n_no_temporal_atte_ASTGNN_w_STemb_contaim2-d'+str(time_view)+'_'+str(ind_c)+'true.txt',
        # #            target,
        # #            delimiter=',')#no_emb_no_temporal_atte_
        # from matplotlib import pyplot as plt
        # import matplotlib as mpl
        # plt.ion()
        # label_size = 20
        # tick_font_size = 14
        # plt.rcParams['axes.labelsize'] = label_size
        # plt.rcParams['axes.titlesize'] = label_size
        # mpl.rcParams['xtick.labelsize'] = tick_font_size
        # mpl.rcParams['ytick.labelsize'] = tick_font_size
        # plt.rcParams["font.family"] = "Calibri"
        # params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        # plurial_param = 10
        # fig, ax = plt.subplots(3, int(pred_wholemap.shape[2]//plurial_param), figsize=(20, 6))
        #
        # for i in range(0,pred_wholemap.shape[2]//plurial_param):
        #     time_p=i*plurial_param+plurial_param-1
        #     im1 = ax[0][i].imshow(pred_wholemap[ :,:, time_p],vmin=0,vmax=vmax_p)#10)#5)#15 ,vmin=0,vmax=2
        #     im2 = ax[1][i].imshow(tag_wholemap[ :,:,  time_p],vmin=0,vmax=vmax_p)#10)#5)#,vmin=0,vmax=1 ,vmin=0,vmax=2
        #     error_p = (pred_wholemap[:, :, time_p] - tag_wholemap[:, :, time_p])
        #     # error_er[tag_wholemap[:, :, time_p]==0]=np.nan
        #     im3 = ax[2][i].imshow(
        #         (error_p), vmin=-1*v_error, vmax=v_error,cmap = 'coolwarm') #-0.5, vmax=0.5,cmap = 'coolwarm')#
        #     ax[2][i].set_xlabel('Time step '+str(time_p))
        #     # im3 = ax[np.mod(i, 5)][i // 5].imshow(
        #     #    pred_wholemap[:, :,  i], vmin=0, vmax=10)
        #     # im3 = ax[np.mod(i, 5)][i // 5].imshow(
        #     #      tag_wholemap[:, :, i] , vmin=0, vmax=10)#,cmap='PuBu'
        #     # im3 = ax[np.mod(i,5)][i//5].imshow(
        #     #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-2,vmax=2,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
        # # fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
        # # fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
        # # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        # ax[0][0].set_ylabel(r'$Pred_{c1}$')
        # ax[1][0].set_ylabel(r'$True_{c1}$')
        # ax[2][0].set_ylabel(r'$Error_{c1}$')
        # cbar1 = fig.colorbar(im1, ax=ax[0:2], shrink=1, pad=0.01)
        # cbar2 = fig.colorbar(im3, ax=ax[2], shrink=1, pad=0.01)
        # cbar1.ax.tick_params(labelsize=tick_font_size)
        # cbar2.ax.tick_params(labelsize=tick_font_size)
        # # plt.savefig('E:\Contaminant_causal\manuscript_writing\picture\\71n_pred_true_2contaim_'+str(ind_c)+'no_temporal_atte_.pdf')#no_emb_no_temporal_atte_
        # plt.ioff()
        # plt.show()
        # a = pred_wholemap-tag_wholemap
        # a[np.isnan(a)]=0
        # print(np.max(a))
        # print(np.min(a))
    #     #
    # # # # # # # # # # # # # # # # # # # # # # return  tag_wholemap,pred_wholemap
    # # # # # # # # # # # # # # # # # # # # # #
    #     encoder_inputs = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view][0], 1, 2), 0)
    #     decoder_input_view =torch.unsqueeze( torch.transpose(data_loader.dataset[time_view][1], 1, 2), 0)
    #     decoder_pump_inputs = decoder_input_view[:, :, :, 3:]
    #     encoder_output = net.encode(encoder_inputs)
    #     for step in range(predict_length):
    #         if step == 0:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -3:], encoder_output,
    #                                         encoder_inputs[:, :, -1:,
    #                                         :3])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
    #         else:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -3:], encoder_output,
    #                                         torch.cat((encoder_inputs[:, :, -1:, :3], predict_output[:, :, :, :]),
    #                                                   axis=-2)) #[0,1,3]
    #
    #     time_view1 = time_view+pred_wholemap.shape[-1]
    #     encoder_input_view = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][0], 1,2), 0)
    #     encoder_input_view_pump = encoder_input_view[:,:,:,-3:]
    #     # predict_output = torch.unsqueeze(predict_output,0)
    #     encoder_input_view_pump1 = torch.cat((predict_output[:,:,-encoder_input_view_pump.shape[-2]:,:],encoder_input_view_pump),axis=-1) #[0,1,3]
    #     encoder_output = net.encode(encoder_input_view_pump1)
    #     decoder_input_view = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][1], 1,2), 0)
    #     decoder_pump_inputs = decoder_input_view[:, :, :, 3:]
    #     for step in range(predict_length):
    #         if step == 0:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -3:], encoder_output,
    #                                         encoder_input_view_pump1[:, :, -1:,
    #                                         :3])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
    #         else:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -3:], encoder_output,
    #                                         torch.cat((encoder_input_view_pump1[:, :, -1:, :3], predict_output[:, :, :, :]),
    #                                                   axis=-2))#[0,1,3]
    #     labels = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][2], 1, 2),0)#[:,:,-encoder_input_view_pump.shape[-2]:,:]
    #
    #     # prediction =re_max_min_normalization(torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][0], 1,2), 0)[:,:,:,:2].cpu().numpy(), _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
    #     # _max1 = np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 0:3:2, 0]),axis=-1)
    #     # _min1 = np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 0:3:2, 0]), axis=-1)
    #     prediction1 = re_max_min_normalization(predict_output.cpu().numpy(), _max1, _min1) #_max[0, 0, 0:2, 0]
    #
    #     target_comp1 = re_max_min_normalization(labels.cpu().numpy(), _max2, _min2)
    #
    #     pred_wholemap = np.nan * np.ones((nrow, ncol, prediction1.shape[2]))
    #     tag_wholemap = np.nan * np.ones((nrow, ncol, prediction1.shape[2]))
    #     ind_c =1
    #     if ind_c==1:
    #         vmax_p = 10
    #         v_error = 0.5
    #     else:
    #         vmax_p = 5
    #         v_error = 0.3
    #     pred = prediction1[ 0,:, :, ind_c]
    #     target = target_comp1[0, :, :,ind_c]  # data_target_tensor
    #     # np.savetxt('71n_no_temporal_atte_ASTGNN_w_STemb_contaim2-d'+str(time_view)+'_ext_'+str(ind_c)+'pred.txt',
    #     #            pred,
    #     #            delimiter=',') # o_emb_no_temporal_atte_ no_temporal_atte_ pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
    #     # np.savetxt('71n_no_temporal_atte_ASTGNN_w_STemb_contaim2-d'+str(time_view)+'_ext_'+str(ind_c)+'true.txt',
    #     #            target,
    #     #            delimiter=',') #o_emb_no_temporal_atte_
    #     for key in index_adj.keys():
    #         pred_wholemap[index_adj[key][0], index_adj[key][1], :] = pred[key, :]
    #         tag_wholemap[index_adj[key][0], index_adj[key][1], :] = target[key, :]
    #     # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
    #     # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
    #     from matplotlib import pyplot as plt
    #     plt.ion()
    #     fig, ax = plt.subplots(3, int(pred_wholemap.shape[2] // plurial_param), figsize=(20, 6))
    #     for i in range(0, pred_wholemap.shape[2]//plurial_param):
    #         im1 = ax[0][i].imshow(pred_wholemap[ :,:, 9+i*plurial_param],vmin=0,vmax=vmax_p)#5)#15 ,vmin=0,vmax=2
    #         im2 = ax[1][i].imshow(tag_wholemap[ :,:, 9+i*plurial_param],vmin=0,vmax=vmax_p)#5)#,vmin=0,vmax=1 ,vmin=0,vmax=2
    #         im3 = ax[2][i].imshow(
    #             pred_wholemap[:, :, 9+i*plurial_param] -  tag_wholemap[:, :, 9+i*plurial_param], vmin=-1*v_error, vmax=v_error,cmap = 'coolwarm') #0.3, vmax=0.3,cmap = 'coolwarm')#
    #         ax[2][i].set_xlabel('Time step ' + str(10+i * 10))
    #         # im3 = ax[np.mod(i, 4)][i // 4].imshow(
    #         #     tag_wholemap[:, :, i], vmin=0, vmax=2)
    #         # im3 = ax[np.mod(i, 5)][i // 5].imshow(
    #         #      pred_wholemap[:, :, i] , vmin=0, vmax=5)#,cmap='PuBu'
    #         # im3 = ax[np.mod(i,5)][i//5].imshow(
    #         #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-3,vmax=3,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
    #     ax[0][0].set_ylabel(r'$Pred_{c2}$')
    #     ax[1][0].set_ylabel(r'$True_{c2}$')
    #     ax[2][0].set_ylabel(r'$Error_{c2}$')
    #     cbar1 = fig.colorbar(im1, ax=ax[0:2], shrink=1, pad=0.01)
    #     cbar2 = fig.colorbar(im3, ax=ax[2], shrink=1, pad=0.01)
    #     cbar1.ax.tick_params(labelsize=tick_font_size)
    #     cbar2.ax.tick_params(labelsize=tick_font_size)
    #     # plt.savefig('E:\Contaminant_causal\manuscript_writing\picture\\71n_pred_true_2contaim_ASTGNN_'+str(ind_c)+'71n_no_temporal_atte_.pdf') #no_emb_no_temporal_atte_.
    #     # fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
    #     # fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
    #     # fig.colorbar(im3, ax=ax[2], shrink=0.5, pad=0.01)
    #     plt.ioff()
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

def predict_and_save_results_pde_train(net, data_loader, data_target_tensor, epoch, _max, _min,decoder_dim, params_path, type, index_adj,nrow,ncol):
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
                # encoder_output = net.encode(encoder_inputs)
                # input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

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

                log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]
                dim_encode = encoder_inputs[:,:,:,4].shape[2]
                dim_decode = decoder_inputs[:,:,:,4].shape[2]
                encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
                decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
                # encode
                encoder_output = net.encode(encoder_inputs[:, :, :, [0,1,5,6]], log1, lat1,encoder_time) #[:,:,:,[0,1,5,6]]
                input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
                dinput_sh = int(decoder_inputs.shape[-1]/2)
                pump_dim = decoder_dim
                decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features
                decoder_input_list = [decoder_start_inputs[:,:,:,-2:]]

                #
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
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, encoder_inputs[:, :, -1:,
                                                                                                :pump_dim],decoder_pump_inputs[:, :, 0:step + 1, 0],decoder_pump_inputs[:, :, 0:step + 1, 1],decoder_time[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    else:  # added
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                                                                                            predict_output ),axis=-2), \
                             decoder_pump_inputs[:, :, 0:step + 1, 0], decoder_pump_inputs[:, :, 0:step + 1,
                                                                        1], decoder_time[:, :, 0:step + 1])
                    # if step == 0:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 encoder_inputs[:, :, -1:,
                    #                                 :dinput_sh])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    # else:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 torch.cat((encoder_inputs[:, :, -1:, :dinput_sh], predict_output[:, :, :, :]),
                    #                                           axis=-2)) #predict_output[:, :, :, [0,1,3]]

                prediction.append(predict_output.detach().cpu().numpy())
                target_comp.append(labels.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, :, 0], _min[0, 0, :, 0])

        # max_sh = int(_max.shape[2]/2)
        _max1 = _max[0, 0, 0:decoder_dim, 0]#np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 0:3:2, 0]),axis=0)
        _min1 = _min[0, 0, 0:decoder_dim, 0]#np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 0:3:2, 0]),axis=0)

        # _max2 = np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 2:3, 0]),axis=0) #0:3:2
        # _min2 = np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 2:3, 0]),axis=0) #0:3:2

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        # prediction = re_max_min_normalization(prediction,_max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        prediction = re_max_min_normalization(prediction, _max1, _min1)
        # data_target_tensor = np.transpose(data_target_tensor, (0, 1, 3, 2))
        # data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        target_comp = np.concatenate(target_comp, 0)#np.transpose(target_comp, (0, 1, 3, 2))
        # target_comp = re_max_min_normalization(target_comp, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        # target_comp = re_max_min_normalization(target_comp, _max2, _min2)
        target_comp = re_max_min_normalization(target_comp, _max1, _min1)


        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        tg_list = list(range(decoder_dim))
        for i in range(prediction_length):
            assert target_comp.shape[0] == prediction.shape[0] #data_target_tensor.shape
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten())
            rmse = mean_squared_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten()) ** 0.5
            mape = masked_mape_np(target_comp[:, :, i, tg_list], prediction[:, :, i, :], 0)
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
        mae = mean_absolute_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

def predict_and_save_results_pde_4induct(net, data_loader, data_target_tensor, epoch, _max, _min,decoder_dim, params_path, type,mask, index_adj,nrow,ncol):
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
                # encoder_output = net.encode(encoder_inputs)
                # input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

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

                log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]
                dim_encode = encoder_inputs[:,:,:,4].shape[2]
                dim_decode = decoder_inputs[:,:,:,4].shape[2]
                encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
                decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
                # encode
                encoder_output = net.encode(encoder_inputs[:, :, :, [0,1,5,6]], log1, lat1,encoder_time,mask) #[:,:,:,[0,1,5,6]]
                input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
                dinput_sh = int(decoder_inputs.shape[-1]/2)
                pump_dim = decoder_dim
                decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features
                decoder_input_list = [decoder_start_inputs[:,:,:,-2:]]

                #
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
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, encoder_inputs[:, :, -1:,
                                                                                                :pump_dim],decoder_pump_inputs[:, :, 0:step + 1, 0],decoder_pump_inputs[:, :, 0:step + 1, 1],decoder_time[:, :, 0:step + 1],mask)  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    else:  # added
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                                                                                            predict_output ),axis=-2), \
                             decoder_pump_inputs[:, :, 0:step + 1, 0], decoder_pump_inputs[:, :, 0:step + 1,
                                                                        1], decoder_time[:, :, 0:step + 1],mask)
                    # if step == 0:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 encoder_inputs[:, :, -1:,
                    #                                 :dinput_sh])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    # else:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 torch.cat((encoder_inputs[:, :, -1:, :dinput_sh], predict_output[:, :, :, :]),
                    #                                           axis=-2)) #predict_output[:, :, :, [0,1,3]]

                prediction.append(predict_output.detach().cpu().numpy())
                target_comp.append(labels.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, :, 0], _min[0, 0, :, 0])

        # max_sh = int(_max.shape[2]/2)
        _max1 = _max[0, 0, 0:decoder_dim, 0]#np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 0:3:2, 0]),axis=0)
        _min1 = _min[0, 0, 0:decoder_dim, 0]#np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 0:3:2, 0]),axis=0)

        # _max2 = np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 2:3, 0]),axis=0) #0:3:2
        # _min2 = np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 2:3, 0]),axis=0) #0:3:2

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        # prediction = re_max_min_normalization(prediction,_max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        prediction = re_max_min_normalization(prediction, _max1, _min1)
        # data_target_tensor = np.transpose(data_target_tensor, (0, 1, 3, 2))
        # data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        target_comp = np.concatenate(target_comp, 0)#np.transpose(target_comp, (0, 1, 3, 2))
        # target_comp = re_max_min_normalization(target_comp, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        # target_comp = re_max_min_normalization(target_comp, _max2, _min2)
        target_comp = re_max_min_normalization(target_comp, _max1, _min1)


        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        tg_list = list(range(decoder_dim))
        for i in range(prediction_length):
            assert target_comp.shape[0] == prediction.shape[0] #data_target_tensor.shape
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten())
            rmse = mean_squared_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten()) ** 0.5
            mape = masked_mape_np(target_comp[:, :, i, tg_list], prediction[:, :, i, :], 0)
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
        mae = mean_absolute_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

        # 计算误差

        workdir  ='results\\'
        time_view = 55#55
        # scenario = 'difficult_no_emb'
        scenario = 'easy1'#difficult _inductive'#'easy_no_emb'#t _inductive1
        # scenario = 'difficult'#easy_inductive_inductive'difficult
        # scenario = 'easy_inductive'
        monitor_sys = '51nd'#71n_t1 no_emb_71n'
        train_test = 'test'
        ind_c = 1
        if ind_c==1:
            vmax_p = 10
            v_error = 0.5#0.5
        else:
            vmax_p = 3
            v_error = 0.5#0.3
        pred_wholemap = np.nan * np.ones((nrow,ncol,prediction.shape[2]))
        tag_wholemap = np.nan * np.ones((nrow, ncol,prediction.shape[2]))
        pred = prediction[time_view,:,:,ind_c]
        target = target_comp[time_view,:,:,ind_c] #data_target_tensor
        np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_pred.npz',
                   prediction)
        np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_true.npz',
                   target_comp)
        for key in index_adj.keys():
            pred_wholemap[index_adj[key][0],index_adj[key][1],:] = pred[key,:]
            tag_wholemap[index_adj[key][0], index_adj[key][1],:] = target[key,:]
        # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
        # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
        # np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_v'+str(ind_c+1)+'_'+train_test+'_pred.txt',
        #            pred,
        #            delimiter=',') # no_emb_no_temporal_atte_ pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
        # np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_v'+str(ind_c+1)+'_'+train_test+'_true.txt',
        #            target,
        #            delimiter=',')#no_emb_no_temporal_atte_
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        plt.ion()
        label_size = 20
        tick_font_size = 14
        plt.rcParams['axes.labelsize'] = label_size
        plt.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['xtick.labelsize'] = tick_font_size
        mpl.rcParams['ytick.labelsize'] = tick_font_size
        plt.rcParams["font.family"] = "Calibri"
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plurial_param = 5
        fig, ax = plt.subplots(3, int(pred_wholemap.shape[2]//plurial_param), figsize=(20, 6))

        for i in range(0,pred_wholemap.shape[2]//plurial_param):
            time_p=i*plurial_param+plurial_param-1
            im1 = ax[0][i].imshow(pred_wholemap[ :,:, time_p],vmin=0,vmax=vmax_p)#10)#5)#15 ,vmin=0,vmax=2
            im2 = ax[1][i].imshow(tag_wholemap[ :,:,  time_p],vmin=0,vmax=vmax_p)#10)#5)#,vmin=0,vmax=1 ,vmin=0,vmax=2
            error_p = (pred_wholemap[:, :, time_p] - tag_wholemap[:, :, time_p])
            # error_er[tag_wholemap[:, :, time_p]==0]=np.nan
            im3 = ax[2][i].imshow(
                (error_p), vmin=-1*v_error, vmax=v_error,cmap = 'coolwarm') #-0.5, vmax=0.5,cmap = 'coolwarm')#
            ax[2][i].set_xlabel('Time step '+str(time_p))
            # im3 = ax[np.mod(i, 5)][i // 5].imshow(
            #    pred_wholemap[:, :,  i], vmin=0, vmax=10)
            # im3 = ax[np.mod(i, 5)][i // 5].imshow(
            #      tag_wholemap[:, :, i] , vmin=0, vmax=10)#,cmap='PuBu'
            # im3 = ax[np.mod(i,5)][i//5].imshow(
            #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-2,vmax=2,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
        # fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
        # fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
        # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        ax[0][0].set_ylabel(r'$Pred_{c1}$')
        ax[1][0].set_ylabel(r'$True_{c1}$')
        ax[2][0].set_ylabel(r'$Error_{c1}$')
        cbar1 = fig.colorbar(im1, ax=ax[0:2], shrink=1, pad=0.01)
        cbar2 = fig.colorbar(im3, ax=ax[2], shrink=1, pad=0.01)
        cbar1.ax.tick_params(labelsize=tick_font_size)
        cbar2.ax.tick_params(labelsize=tick_font_size)
        # plt.savefig('E:\Contaminant_causal\manuscript_writing\picture\\71n_pred_true_2contaim_'+str(ind_c)+'no_temporal_atte_.pdf')#no_emb_no_temporal_atte_
        plt.ioff()
        plt.show()
        a = pred_wholemap-tag_wholemap
        a[np.isnan(a)]=0
        print(np.max(a))
        print(np.min(a))

def predict_and_save_results_pde_ind(net, net_ind, data_loader, data_target_tensor,data_loader_o, data_target_tensor_o, epoch, _max, _min,decoder_dim, params_path, type,index_adj_ind, index_adj,nrow,ncol):
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
    net_ind.train(False)
    start_time = time()

    adj_ind_indx = { (index_adj_ind[key_in][0],index_adj_ind[key_in][1]):key_in for key_in in index_adj_ind.keys()}

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        target_comp = []

        input = []  # 存储所有batch的input
        input_o = []

        start_time = time()
        data_loader_iterator = iter(data_loader_o)
        for batch_index, batch_data in enumerate(data_loader):
            # if batch_index ==87 or batch_index ==88:

                encoder_inputs, decoder_inputs, labels = batch_data

                encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

                decoder_inputs = decoder_inputs.transpose(-1,
                                                          -2)  # decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1) ->(B, N, T, F)

                encoder_inputs_o, decoder_inputs_o, labels_o = data_loader_iterator.next()

                encoder_inputs_o = encoder_inputs_o.transpose(-1, -2)  # (B, N, T, F)

                decoder_inputs_o = decoder_inputs_o.transpose(-1,
                                                          -2)

                predict_length_o = labels_o.shape[-1]  # T
                labels_o = labels_o.transpose(-1, -2)
                # labels = labels.unsqueeze(-1)  # (B, N, T, 1)

                # predict_length = labels.shape[2]  # T
                predict_length = labels.shape[-1]  # T
                labels = labels.transpose(-1, -2)
                # encode
                # encoder_output = net.encode(encoder_inputs)
                # input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

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

                log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]
                dim_encode = encoder_inputs[:,:,:,4].shape[2]
                dim_decode = decoder_inputs[:,:,:,4].shape[2]
                encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
                decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
                # encode
                encoder_output = net_ind.encode(encoder_inputs[:, :, :, [0,1,5,6]], log1, lat1,encoder_time) #[:,:,:,[0,1,5,6]]
                input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)
                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
                dinput_sh = int(decoder_inputs.shape[-1]/2)
                pump_dim = decoder_dim
                decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features
                decoder_input_list = [decoder_start_inputs[:,:,:,-2:]]

                log1_o, lat1_o = decoder_inputs_o[:, :, :, 2], decoder_inputs_o[:, :, :, 3]
                dim_encode = encoder_inputs_o[:, :, :, 4].shape[2]
                dim_decode = decoder_inputs_o[:, :, :, 4].shape[2]
                encoder_time_o = torch.tensor(
                    [[list(range(dim_encode))] * encoder_inputs_o.shape[1]] * encoder_inputs_o.shape[0]).cuda()
                decoder_time_o = torch.tensor(
                    [[list(range(dim_decode))] * decoder_inputs_o.shape[1]] * decoder_inputs_o.shape[0]).cuda()
                # encode
                encoder_output_o = net.encode(encoder_inputs_o[:, :, :, [0, 1, 5, 6]], log1_o, lat1_o ,
                                            encoder_time_o)  # [:,:,:,[0,1,5,6]]
                input_o.append(encoder_inputs_o[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)
                decoder_start_inputs_o = decoder_inputs_o[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
                dinput_sh = int(decoder_inputs_o.shape[-1] / 2)
                pump_dim = decoder_dim
                decoder_pump_inputs_o = decoder_inputs_o[:, :, :, pump_dim:]  # 2 features
                decoder_input_list_o = [decoder_start_inputs_o[:, :, :, -2:]]

        #
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
                        predict_output = net.decode1(decoder_pump_inputs_o[:, :, 0:step + 1, -pump_dim:], encoder_output_o, encoder_inputs_o[:, :, -1:,\
                                                :pump_dim],decoder_pump_inputs_o[:, :, 0:step + 1, 0],decoder_pump_inputs_o[:, :, 0:step + 1, 1],decoder_time_o[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])

                        predict_output_ind = net_ind.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output,
                                                     encoder_inputs[:, :, -1:,
                                                     :pump_dim], decoder_pump_inputs[:, :, 0:step + 1, 0],
                                                     decoder_pump_inputs[:, :, 0:step + 1, 1],
                                                     decoder_time[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_o
                        for ind_i in range(len(index_adj.keys())):
                            ind_i_x, ind_i_y = index_adj[ind_i]
                            adj_indx = adj_ind_indx[(ind_i_x, ind_i_y )]
                            predict_output_ind[:,adj_indx,:,:] = predict_output[:,ind_i,:,:]
                        # predict_output_ind[predict_output_ind<0]=0
                    else:  # added
                        predict_output = net.decode1(decoder_pump_inputs_o[:, :, 0:step + 1, -pump_dim:], encoder_output_o, torch.cat((encoder_inputs_o[:, :, -1:, :pump_dim], \
                                                                                                                            predict_output ),axis=-2), \
                             decoder_pump_inputs_o[:, :, 0:step + 1, 0], decoder_pump_inputs_o[:, :, 0:step + 1,
                                                                        1], decoder_time_o[:, :, 0:step + 1])

                        predict_output_ind = net_ind.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output,
                                                     torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                                predict_output_ind), axis=-2), \
                                                     decoder_pump_inputs[:, :, 0:step + 1, 0], decoder_pump_inputs[:, :, 0:step + 1,
                                                                                               1], decoder_time[:, :, 0:step + 1])
                        for ind_i in range(len(index_adj.keys())):
                            ind_i_x, ind_i_y = index_adj[ind_i]
                            adj_indx = adj_ind_indx[(ind_i_x, ind_i_y)]
                            predict_output_ind[:, adj_indx, :, :] = predict_output[:, ind_i, :, :]
                        # predict_output_ind[predict_output_ind < 0] = 0
                    # if step == 0:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 encoder_inputs[:, :, -1:,
                    #                                 :dinput_sh])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    # else:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 torch.cat((encoder_inputs[:, :, -1:, :dinput_sh], predict_output[:, :, :, :]),
                    #                                           axis=-2)) #predict_output[:, :, :, [0,1,3]]

                prediction.append(predict_output_ind.detach().cpu().numpy())
                target_comp.append(labels.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, :, 0], _min[0, 0, :, 0])

        # max_sh = int(_max.shape[2]/2)
        _max1 = _max[0, 0, 0:decoder_dim, 0]#np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 0:3:2, 0]),axis=0)
        _min1 = _min[0, 0, 0:decoder_dim, 0]#np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 0:3:2, 0]),axis=0)

        # _max2 = np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 2:3, 0]),axis=0) #0:3:2
        # _min2 = np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 2:3, 0]),axis=0) #0:3:2

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        # prediction = re_max_min_normalization(prediction,_max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        prediction = re_max_min_normalization(prediction, _max1, _min1)
        # data_target_tensor = np.transpose(data_target_tensor, (0, 1, 3, 2))
        # data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        target_comp = np.concatenate(target_comp, 0)#np.transpose(target_comp, (0, 1, 3, 2))
        # target_comp = re_max_min_normalization(target_comp, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        # target_comp = re_max_min_normalization(target_comp, _max2, _min2)
        target_comp = re_max_min_normalization(target_comp, _max1, _min1)


        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        tg_list = list(range(decoder_dim))
        for i in range(prediction_length):
            assert target_comp.shape[0] == prediction.shape[0] #data_target_tensor.shape
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten())
            rmse = mean_squared_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten()) ** 0.5
            mape = masked_mape_np(target_comp[:, :, i, tg_list], prediction[:, :, i, :], 0)
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
        mae = mean_absolute_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

        workdir  ='results\\'
        time_view = 55#55#55
        # scenario = 'difficult_no_emb'
        scenario = 'difficult_inductive'#'easy_no_emb'#t _inductive1
        # scenario = 'difficult'#easy_inductive_inductive'difficult
        # scenario = 'easy_inductive'
        monitor_sys = '51nd1'#'71n1'#_no_emb _no_emb _no_emb _no_emb71n_t1 no_emb_71n'
        train_test = 'train'
        ind_c = 1
        if ind_c==1:
            vmax_p = 10
            v_error = 1#0.5
        else:
            vmax_p = 3
            v_error = 0.5#.5#0.3
        pred_wholemap = np.nan * np.ones((nrow,ncol,prediction.shape[2]))
        tag_wholemap = np.nan * np.ones((nrow, ncol,prediction.shape[2]))
        pred = prediction[time_view,:,:,ind_c]
        target = target_comp[time_view,:,:,ind_c] #data_target_tensor
        np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_pred1.npz',
                   prediction)
        np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_true1.npz',
                   target_comp)
        for key in index_adj_ind.keys():
            pred_wholemap[index_adj_ind[key][0],index_adj_ind[key][1],:] = pred[key,:]
            tag_wholemap[index_adj_ind[key][0], index_adj_ind[key][1],:] = target[key,:]
        # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
        # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
        # np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_v'+str(ind_c+1)+'_'+train_test+'_pred.txt',
        #            pred,
        #            delimiter=',') # no_emb_no_temporal_atte_ pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
        # np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_v'+str(ind_c+1)+'_'+train_test+'_true.txt',
        #            target,
        #            delimiter=',')#no_emb_no_temporal_atte_
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        plt.ion()
        label_size = 20
        tick_font_size = 14
        plt.rcParams['axes.labelsize'] = label_size
        plt.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['xtick.labelsize'] = tick_font_size
        mpl.rcParams['ytick.labelsize'] = tick_font_size
        plt.rcParams["font.family"] = "Calibri"
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plurial_param = 5
        fig, ax = plt.subplots(3, int(pred_wholemap.shape[2]//plurial_param), figsize=(20, 6))

        pred_wholemap[pred_wholemap<0]=0
        for i in range(0,pred_wholemap.shape[2]//plurial_param):
            time_p=i*plurial_param+plurial_param-1
            im1 = ax[0][i].imshow(pred_wholemap[ :,:, time_p],vmin=0,vmax=vmax_p)#10)#5)#15 ,vmin=0,vmax=2
            im2 = ax[1][i].imshow(tag_wholemap[ :,:,  time_p],vmin=0,vmax=vmax_p)#10)#5)#,vmin=0,vmax=1 ,vmin=0,vmax=2
            error_p = (pred_wholemap[:, :, time_p] - tag_wholemap[:, :, time_p])
            # error_er[tag_wholemap[:, :, time_p]==0]=np.nan
            im3 = ax[2][i].imshow(
                (error_p), vmin=-1*v_error, vmax=v_error ,cmap = 'coolwarm') # -0.5, vmax=0.5,cmap = 'coolwarm')#
            ax[2][i].set_xlabel('Time step '+str(time_p))
            # im3 = ax[np.mod(i, 5)][i // 5].imshow(
            #    pred_wholemap[:, :,  i], vmin=0, vmax=10)
            # im3 = ax[np.mod(i, 5)][i // 5].imshow(
            #      tag_wholemap[:, :, i] , vmin=0, vmax=10)#,cmap='PuBu'
            # im3 = ax[np.mod(i,5)][i//5].imshow(
            #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-2,vmax=2,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
        # fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
        # fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
        # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        ax[0][0].set_ylabel(r'$Pred_{c1}$')
        ax[1][0].set_ylabel(r'$True_{c1}$')
        ax[2][0].set_ylabel(r'$Error_{c1}$')
        cbar1 = fig.colorbar(im1, ax=ax[0:2], shrink=1, pad=0.01)
        cbar2 = fig.colorbar(im3, ax=ax[2], shrink=1, pad=0.01)
        cbar1.ax.tick_params(labelsize=tick_font_size)
        cbar2.ax.tick_params(labelsize=tick_font_size)
        # plt.savefig('E:\Contaminant_causal\manuscript_writing\picture\\71n_pred_true_2contaim_'+str(ind_c)+'no_temporal_atte_.pdf')#no_emb_no_temporal_atte_
        plt.ioff()
        plt.show()
        a = pred_wholemap-tag_wholemap
        a[np.isnan(a)]=0
        print(np.max(a))
        print(np.min(a))

def predict_and_save_results_pde(net, data_loader, data_target_tensor, epoch, _max, _min,decoder_dim, params_path, type, index_adj,nrow,ncol):
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
                # encoder_output = net.encode(encoder_inputs)
                # input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

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

                log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]
                dim_encode = encoder_inputs[:,:,:,4].shape[2]
                dim_decode = decoder_inputs[:,:,:,4].shape[2]
                encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
                decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
                # encode
                encoder_output = net.encode(encoder_inputs[:, :, :, [0,1,5,6]], log1, lat1,encoder_time) #[:,:,:,[0,1,5,6]]
                input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]
                dinput_sh = int(decoder_inputs.shape[-1]/2)
                pump_dim = decoder_dim
                decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features
                decoder_input_list = [decoder_start_inputs[:,:,:,-2:]]

                #
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
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, encoder_inputs[:, :, -1:,
                                                                                                :pump_dim],decoder_pump_inputs[:, :, 0:step + 1, 0],decoder_pump_inputs[:, :, 0:step + 1, 1],decoder_time[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    else:  # added
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                                                                                            predict_output ),axis=-2), \
                             decoder_pump_inputs[:, :, 0:step + 1, 0], decoder_pump_inputs[:, :, 0:step + 1,
                                                                        1], decoder_time[:, :, 0:step + 1])
                    # if step == 0:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 encoder_inputs[:, :, -1:,
                    #                                 :dinput_sh])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    # else:  # added
                    #     predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, :], encoder_output,
                    #                                 torch.cat((encoder_inputs[:, :, -1:, :dinput_sh], predict_output[:, :, :, :]),
                    #                                           axis=-2)) #predict_output[:, :, :, [0,1,3]]

                prediction.append(predict_output.detach().cpu().numpy())
                target_comp.append(labels.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, :, 0], _min[0, 0, :, 0])

        # max_sh = int(_max.shape[2]/2)
        _max1 = _max[0, 0, 0:decoder_dim, 0]#np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 0:3:2, 0]),axis=0)
        _min1 = _min[0, 0, 0:decoder_dim, 0]#np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 0:3:2, 0]),axis=0)

        # _max2 = np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 2:3, 0]),axis=0) #0:3:2
        # _min2 = np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 2:3, 0]),axis=0) #0:3:2

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        # prediction = re_max_min_normalization(prediction,_max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        prediction = re_max_min_normalization(prediction, _max1, _min1)
        # data_target_tensor = np.transpose(data_target_tensor, (0, 1, 3, 2))
        # data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        target_comp = np.concatenate(target_comp, 0)#np.transpose(target_comp, (0, 1, 3, 2))
        # target_comp = re_max_min_normalization(target_comp, _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
        # target_comp = re_max_min_normalization(target_comp, _max2, _min2)
        target_comp = re_max_min_normalization(target_comp, _max1, _min1)


        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        tg_list = list(range(decoder_dim))
        for i in range(prediction_length):
            assert target_comp.shape[0] == prediction.shape[0] #data_target_tensor.shape
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten())
            rmse = mean_squared_error(target_comp[:, :, i, tg_list].flatten(), prediction[:, :, i, :].flatten()) ** 0.5
            mape = masked_mape_np(target_comp[:, :, i, tg_list], prediction[:, :, i, :], 0)
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
        mae = mean_absolute_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(target_comp[:, :, :, tg_list].reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

        workdir  ='results\\'
        time_view = 55#55
        # scenario = 'difficult_no_emb'
        scenario = 'difficult1'#difficult _inductive'#'easy_no_emb'#t _inductive1
        # scenario = 'difficult'#easy_inductive_inductive'difficult
        # scenario = 'easy_inductive'
        monitor_sys = '51nd'#711nd1  71n_t1 no_emb_71n'
        train_test = 'train'
        ind_c = 1
        if ind_c==1:
            vmax_p = 10
            v_error = 0.5#0.5
        else:
            vmax_p = 3
            v_error = 0.5#0.3
        pred_wholemap = np.nan * np.ones((nrow,ncol,prediction.shape[2]))
        tag_wholemap = np.nan * np.ones((nrow, ncol,prediction.shape[2]))
        pred = prediction[time_view,:,:,ind_c]
        target = target_comp[time_view,:,:,ind_c] #data_target_tensor
        # np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_pred.npz',
        #            prediction)
        # np.save(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1_v'+str(ind_c+1)+'_'+train_test+'_true.npz',
        #            target_comp)
        for key in index_adj.keys():
            pred_wholemap[index_adj[key][0],index_adj[key][1],:] = pred[key,:]
            tag_wholemap[index_adj[key][0], index_adj[key][1],:] = target[key,:]
        # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
        # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
        # np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_v'+str(ind_c+1)+'_'+train_test+'_pred.txt',
        #            pred,
        #            delimiter=',') # no_emb_no_temporal_atte_ pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
        # np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_v'+str(ind_c+1)+'_'+train_test+'_true.txt',
        #            target,
        #            delimiter=',')#no_emb_no_temporal_atte_
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        plt.ion()
        label_size = 20
        tick_font_size = 14
        plt.rcParams['axes.labelsize'] = label_size
        plt.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['xtick.labelsize'] = tick_font_size
        mpl.rcParams['ytick.labelsize'] = tick_font_size
        plt.rcParams["font.family"] = "Calibri"
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plurial_param = 5
        fig, ax = plt.subplots(3, int(pred_wholemap.shape[2]//plurial_param), figsize=(20, 6))

        for i in range(0,pred_wholemap.shape[2]//plurial_param):
            time_p=i*plurial_param+plurial_param-1
            im1 = ax[0][i].imshow(pred_wholemap[ :,:, time_p],vmin=0,vmax=vmax_p)#10)#5)#15 ,vmin=0,vmax=2
            im2 = ax[1][i].imshow(tag_wholemap[ :,:,  time_p],vmin=0,vmax=vmax_p)#10)#5)#,vmin=0,vmax=1 ,vmin=0,vmax=2
            error_p = (pred_wholemap[:, :, time_p] - tag_wholemap[:, :, time_p])
            # error_er[tag_wholemap[:, :, time_p]==0]=np.nan
            im3 = ax[2][i].imshow(
                (error_p), vmin=-1*v_error, vmax=v_error,cmap = 'coolwarm') #-0.5, vmax=0.5,cmap = 'coolwarm')#
            ax[2][i].set_xlabel('Time step '+str(time_p))
            # im3 = ax[np.mod(i, 5)][i // 5].imshow(
            #    pred_wholemap[:, :,  i], vmin=0, vmax=10)
            # im3 = ax[np.mod(i, 5)][i // 5].imshow(
            #      tag_wholemap[:, :, i] , vmin=0, vmax=10)#,cmap='PuBu'
            # im3 = ax[np.mod(i,5)][i//5].imshow(
            #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-2,vmax=2,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
        # fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
        # fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
        # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        ax[0][0].set_ylabel(r'$Pred_{c1}$')
        ax[1][0].set_ylabel(r'$True_{c1}$')
        ax[2][0].set_ylabel(r'$Error_{c1}$')
        cbar1 = fig.colorbar(im1, ax=ax[0:2], shrink=1, pad=0.01)
        cbar2 = fig.colorbar(im3, ax=ax[2], shrink=1, pad=0.01)
        cbar1.ax.tick_params(labelsize=tick_font_size)
        cbar2.ax.tick_params(labelsize=tick_font_size)
        # plt.savefig('E:\Contaminant_causal\manuscript_writing\picture\\71n_pred_true_2contaim_'+str(ind_c)+'no_temporal_atte_.pdf')#no_emb_no_temporal_atte_
        plt.ioff()
        plt.show()
        a = pred_wholemap-tag_wholemap
        a[np.isnan(a)]=0
        print(np.max(a))
        print(np.min(a))
    # # # # # #     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # return  tag_wholemap,pred_wholemap
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #     encoder_inputs = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view][0], 1, 2), 0)
    #     decoder_input_view =torch.unsqueeze( torch.transpose(data_loader.dataset[time_view][1], 1, 2), 0)
    #     decoder_pump_inputs = decoder_input_view[:, :, :, decoder_dim:]
    #
    #     log1, lat1 = decoder_input_view[:, :, :, 2], decoder_input_view[:, :, :,3]
    #
    #     dim_encode = encoder_inputs[:, :, :, 4].shape[2]
    #     dim_decode = decoder_inputs[:, :, :, 4].shape[2]
    #     encoder_time = torch.tensor(
    #         [[list(range(dim_encode))] * encoder_inputs.shape[1]] * encoder_inputs.shape[0]).cuda()
    #     decoder_time = torch.tensor(
    #         [[list(range(dim_decode))] * encoder_inputs.shape[1]] * encoder_inputs.shape[0]).cuda()
    #
    #     encoder_output = net.encode(encoder_inputs[:, :, :, [0, 1, 5, 6]], encoder_inputs[:, :, :, 2], encoder_inputs[:, :, :, 3], encoder_time)
    #     # encoder_output = net.encode(encoder_inputs)
    #     for step in range(predict_length):
    #         if step == 0:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -decoder_dim:], encoder_output,
    #                                         encoder_inputs[:, :, -1:,:decoder_dim],log1[:, :, 0:step + 1], lat1[:, :, 0:step + 1],decoder_time[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
    #         else:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -decoder_dim:], encoder_output,
    #                                         torch.cat((encoder_inputs[:, :, -1:, :decoder_dim], predict_output[:, :, :, :]),
    #                                                   axis=-2),log1[:, :, 0:step + 1], lat1[:, :, 0:step + 1],decoder_time[:, :, 0:step + 1]) #[0,1,3]
    #
    #     time_view1 = time_view+pred_wholemap.shape[-1]
    #     encoder_input_view = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][0], 1,2), 0)
    #     encoder_input_view_pump = encoder_input_view[:,:,:,-decoder_dim:]
    #     # predict_output = torch.unsqueeze(predict_output,0)
    #     encoder_input_view_pump1 = torch.cat((predict_output[:,:,-encoder_input_view_pump.shape[-2]:,:],encoder_input_view_pump),axis=-1) #[0,1,3]
    #
    #     # encoder_time = encoder_input_view[:, :, :, 4]
    #
    #     encoder_output = net.encode(encoder_input_view_pump1,  encoder_input_view[:, :, :, 2],  encoder_input_view[:, :, :, 3], encoder_time)
    #
    #     # decoder_time=decoder_pump_inputs[:, :, :, 4]+decoder_pump_inputs[:, :, -1:, 4]-decoder_pump_inputs[:, :, :1, 4] #decoder_inputs[:, :, :, 4]
    #
    #     decoder_input_view = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][1], 1,2), 0)
    #     decoder_pump_inputs = decoder_input_view[:, :, :, decoder_dim:]
    #
    #     log1, lat1 = decoder_input_view[:, :, :, 2], decoder_input_view[:, :, :,
    #                                                                          3]
    #     # decoder_time = decoder_input_view[:, :, :, 4]
    #     for step in range(predict_length):
    #         if step == 0:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -decoder_dim:], encoder_output,
    #                                         encoder_input_view_pump1[:, :, -1:,
    #                                         :decoder_dim],log1[:, :, 0:step + 1], lat1[:, :, 0:step + 1],decoder_time[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
    #         else:  # added
    #             predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -decoder_dim:], encoder_output,
    #                                         torch.cat((encoder_input_view_pump1[:, :, -1:, :decoder_dim], predict_output[:, :, :, :]),
    #                                                   axis=-2),log1[:, :, 0:step + 1], lat1[:, :, 0:step + 1],decoder_time[:, :, 0:step + 1])#[0,1,3]
    #     labels = torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][2], 1, 2),0)#[:,:,-encoder_input_view_pump.shape[-2]:,:]
    #
    #     # prediction =re_max_min_normalization(torch.unsqueeze(torch.transpose(data_loader.dataset[time_view1][0], 1,2), 0)[:,:,:,:2].cpu().numpy(), _max[0, 0, 0:2, 0], _min[0, 0, 0:2, 0])
    #     # _max1 = np.concatenate((_max[0, 0, 0:2, 0], _max[0, 0, 0:3:2, 0]),axis=-1)
    #     # _min1 = np.concatenate((_min[0, 0, 0:2, 0], _min[0, 0, 0:3:2, 0]), axis=-1)
    #     prediction1 = re_max_min_normalization(predict_output.cpu().numpy(), _max1, _min1) #_max[0, 0, 0:2, 0]
    #
    #     target_comp1 = re_max_min_normalization(labels.cpu().numpy(), _max1, _min1)
    #
    #     pred_wholemap = np.nan * np.ones((nrow, ncol, prediction1.shape[2]))
    #     tag_wholemap = np.nan * np.ones((nrow, ncol, prediction1.shape[2]))
    #     ind_c =1
    #     if ind_c==1:
    #         vmax_p = 10
    #         v_error = 0.5
    #     else:
    #         vmax_p = 5
    #         v_error = 0.3
    #     pred = prediction1[ 0,:, :, ind_c]
    #     target = target_comp1[0, :, :,ind_c]  # data_target_tensor
    #     np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_ext_v'+str(ind_c+1)+'_'+train_test+'_pred.txt',
    #                pred,
    #                delimiter=',') # o_emb_no_temporal_atte_ no_temporal_atte_ pred_wholemap.reshape((pred_wholemap.shape[0] * pred_wholemap.shape[1], pred_wholemap.shape[2]))
    #     np.savetxt(workdir +scenario+'_'+monitor_sys+'_ASTGNN_w_STemb_contaim1-d'+str(time_view)+'_ext_v'+str(ind_c+1)+'_'+train_test+'_test_true.txt',
    #                target,
    #                delimiter=',') #o_emb_no_temporal_atte_
    #     for key in index_adj.keys():
    #         pred_wholemap[index_adj[key][0], index_adj[key][1], :] = pred[key, :]
    #         tag_wholemap[index_adj[key][0], index_adj[key][1], :] = target[key, :]
    #     # b = np.concatenate((np.zeros((1,prediction.shape[2])),prediction[0,:,:,1]),axis=0)
    #     # b_t = np.concatenate((np.zeros((1,prediction.shape[2])),data_target_tensor[0,:,:,1]),axis=0)
    #     from matplotlib import pyplot as plt
    #     plt.ion()
    #     fig, ax = plt.subplots(3, int(pred_wholemap.shape[2] // plurial_param), figsize=(20, 6))
    #     for i in range(0, pred_wholemap.shape[2]//plurial_param):
    #         im1 = ax[0][i].imshow(pred_wholemap[ :,:, 9+i*plurial_param],vmin=0,vmax=vmax_p)#5)#15 ,vmin=0,vmax=2
    #         im2 = ax[1][i].imshow(tag_wholemap[ :,:, 9+i*plurial_param],vmin=0,vmax=vmax_p)#5)#,vmin=0,vmax=1 ,vmin=0,vmax=2
    #         im3 = ax[2][i].imshow(
    #             pred_wholemap[:, :, 9+i*plurial_param] -  tag_wholemap[:, :, 9+i*plurial_param], vmin=-1*v_error, vmax=v_error,cmap = 'coolwarm') #0.3, vmax=0.3,cmap = 'coolwarm')#
    #         ax[2][i].set_xlabel('Time step ' + str(10+i * 10))
    #         # im3 = ax[np.mod(i, 4)][i // 4].imshow(
    #         #     tag_wholemap[:, :, i], vmin=0, vmax=2)
    #         # im3 = ax[np.mod(i, 5)][i // 5].imshow(
    #         #      pred_wholemap[:, :, i] , vmin=0, vmax=5)#,cmap='PuBu'
    #         # im3 = ax[np.mod(i,5)][i//5].imshow(
    #         #     pred_wholemap[ :,:, i]- tag_wholemap[ :,:, i],vmin=-3,vmax=3,cmap = 'coolwarm')# ,vmin=-0.25,vmax=0.25# ,vmin=-0.2,vmax=0.2
    #     ax[0][0].set_ylabel(r'$Pred_{c2}$')
    #     ax[1][0].set_ylabel(r'$True_{c2}$')
    #     ax[2][0].set_ylabel(r'$Error_{c2}$')
    #     cbar1 = fig.colorbar(im1, ax=ax[0:2], shrink=1, pad=0.01)
    #     cbar2 = fig.colorbar(im3, ax=ax[2], shrink=1, pad=0.01)
    #     cbar1.ax.tick_params(labelsize=tick_font_size)
    #     cbar2.ax.tick_params(labelsize=tick_font_size)
    #     # plt.savefig('E:\Contaminant_causal\manuscript_writing\picture\\71n_pred_true_2contaim_ASTGNN_'+str(ind_c)+'71n_no_temporal_atte_.pdf') #no_emb_no_temporal_atte_.
    #     # fig.colorbar(im1, ax=ax[0], shrink=0.5, pad=0.01)
    #     # fig.colorbar(im2, ax=ax[1], shrink=0.5, pad=0.01)
    #     # fig.colorbar(im3, ax=ax[2], shrink=0.5, pad=0.01)
    #     plt.ioff()
    #     plt.show()
    #     a = pred_wholemap-tag_wholemap
    #     a[np.isnan(a)]=0
    #     print(np.max(a))
    #     print(np.min(a))

def load_graphdata_normY_channel2(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, decoder_output_size,DEVICE, batch_size, shuffle=True, percent=1.0):
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
    _max1 = _max[:, :, 0:decoder_output_size, :]#np.concatenate((_max[:, :, 0:2, :],_max[:, :, 2:3, :]),axis=2) #0:3:2
    _min1 = _min[:, :, 0:decoder_output_size, :]#np.concatenate((_min[:, :, 0:2, :], _min[:, :, 2:3, :]), axis=2) #0:3:2
    train_target_norm = max_min_normalization(train_target, _max1,  _min1) #0:2 as two target to predict
    test_target_norm = max_min_normalization(test_target,_max1,  _min1)
    val_target_norm = max_min_normalization(val_target, _max1,  _min1)#_max[:, :, 0:2, :], _min[:, :, 0:2, :])

    #  ------- train_loader -------
    # train_decoder_input_start = train_x[:, :, 0:1, -1:]
    train_decoder_input_start = train_x[:, :, 0:decoder_output_size, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    get_list = list(range(int(decoder_output_size)))
    train_decoder_input_p = np.concatenate((train_decoder_input_start, train_target_norm[:, :,get_list, :-1]), axis=3) #[0,1,3]
    # train_decoder_input_p = np.concatenate((train_decoder_input_start, train_target_norm[:, :,:-1]), axis=2) # axis=2 # (B, N, T)
    # train_decoder_input = np.concatenate((np.expand_dims(train_decoder_input_p, axis=-2), train_decoder_x), axis=-2) #-> (B, N, F,T)
    train_decoder_input = np.concatenate((train_decoder_input_p, train_decoder_x), axis=-2)
    #  (B,N,T,F(2)) in pump and drawdown

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)-> (B, N, F,T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)#shuffle

    #  ------- val_loader -------
    # val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = val_x[:, :, 0:decoder_output_size, -1:]
    # val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    # val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T) -> (B, N, F,T)
    val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :,get_list, :-1]), axis=3)
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
    test_decoder_input_start = test_x[:, :, 0:decoder_output_size, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    # test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)
    test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :,get_list, :-1]), axis=3)  # (B, N, T)
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

def load_graphdata_normY_channel3(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, decoder_output_size,DEVICE, batch_size, shuffle=True, percent=1.0):
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
    _max1 = _max[:, :, 0:decoder_output_size, :]#np.concatenate((_max[:, :, 0:2, :],_max[:, :, 2:3, :]),axis=2) #0:3:2
    _min1 = _min[:, :, 0:decoder_output_size, :]#np.concatenate((_min[:, :, 0:2, :], _min[:, :, 2:3, :]), axis=2) #0:3:2
    train_target_norm = max_min_normalization(train_target, _max1,  _min1) #0:2 as two target to predict
    test_target_norm = max_min_normalization(test_target,_max1,  _min1)
    val_target_norm = max_min_normalization(val_target, _max1,  _min1)#_max[:, :, 0:2, :], _min[:, :, 0:2, :])

    #  ------- train_loader -------
    # train_decoder_input_start = train_x[:, :, 0:1, -1:]
    train_decoder_input_start = train_x[:, :, 0:decoder_output_size, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    get_list = list(range(int(decoder_output_size)))
    train_decoder_input_p = np.concatenate((train_decoder_input_start, train_target_norm[:, :,get_list, :-1]), axis=3) #[0,1,3]
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
    val_decoder_input_start = val_x[:, :, 0:decoder_output_size, -1:]
    # val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    # val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T) -> (B, N, F,T)
    val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :,get_list, :-1]), axis=3)
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
    test_decoder_input_start = test_x[:, :, 0:decoder_output_size, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    # test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)
    test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :,get_list, :-1]), axis=3)  # (B, N, T)
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

def load_graphdata_normY_channel_PINN(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, decoder_output_size,DEVICE, batch_size, shuffle=True, percent=1.0):
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
    # train_decoder_x= file_data['train_decoder_x']
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
    # val_decoder_x = file_data['val_decoder_x']
    #val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    # test_decoder_x = file_data['test_decoder_x']
    #test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    _max1 = _max[:, :, 0:decoder_output_size, :]#np.concatenate((_max[:, :, 0:2, :],_max[:, :, 2:3, :]),axis=2) #0:3:2
    _min1 = _min[:, :, 0:decoder_output_size, :]#np.concatenate((_min[:, :, 0:2, :], _min[:, :, 2:3, :]), axis=2) #0:3:2
    train_target_norm = max_min_normalization(train_target, _max1,  _min1) #0:2 as two target to predict
    test_target_norm = max_min_normalization(test_target,_max1,  _min1)
    val_target_norm = max_min_normalization(val_target, _max1,  _min1)#_max[:, :, 0:2, :], _min[:, :, 0:2, :])

    #  ------- train_loader -------
    # train_decoder_input_start = train_x[:, :, 0:1, -1:]
    train_decoder_input_start = train_x[:, :, 0:decoder_output_size, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    get_list = list(range(int(decoder_output_size)))
    train_decoder_input_p = np.concatenate((train_decoder_input_start, train_target_norm[:, :,get_list, :-1]), axis=3) #[0,1,3]
    # train_decoder_input = np.concatenate((train_decoder_input_p, train_decoder_x), axis=-2)#-> (B, N, F,T)
    #  (B,N,T,F(2)) in pump and drawdown

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    # train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)-> (B, N, F,T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor,  train_target_tensor)#train_decoder_input_tensor

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)#shuffle

    #  ------- val_loader -------
    # val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = val_x[:, :, 0:decoder_output_size, -1:]
    # val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    # val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T) -> (B, N, F,T)
    val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :,get_list, :-1]), axis=3)
    # val_decoder_input = np.concatenate((np.expand_dims(val_decoder_input_p,axis=-2),val_decoder_x),axis=-2)
    # val_decoder_input = np.concatenate((val_decoder_input_p,val_decoder_x), axis=-2)
    #  (B,N,T,F(2)) in pump and drawdown

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    # val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T) -> (B, N, F,T)
    #  (B,N,T,F(2)) in pump and drawdown
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)#val_decoder_input_tensor

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    # test_decoder_input_start = test_x[:, :, 0:1, -1:]
    test_decoder_input_start = test_x[:, :, 0:decoder_output_size, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    # test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    # test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)
    test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :,get_list, :-1]), axis=3)  # (B, N, T)
    # test_decoder_input = np.concatenate((np.expand_dims(test_decoder_input_p,axis=-2),test_decoder_x),axis=-2)
    # test_decoder_input = np.concatenate((test_decoder_input_p, test_decoder_x), axis=-2)
    #  (B,N,T,F(2)) in pump and drawdown

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    # test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  #(B, N, T) -> (B, N, F,T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor,test_target_tensor)#, test_decoder_input_tensor

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(),  train_target_tensor.size())#train_decoder_input_tensor.size(),
    print('val:', val_x_tensor.size(), val_target_tensor.size())#val_decoder_input_tensor.size(),
    print('test:', test_x_tensor.size(), test_target_tensor.size())# test_decoder_input_tensor.size(),

    return train_loader, train_target_tensor,train_timestamp, val_loader, val_target_tensor, val_timestamp,test_loader, test_target_tensor,test_timestamp,  _max, _min

