import os
import numpy as np
import torch
import torch.utils.data

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

def compute_val_loss(net, val_loader, criterion, sw,decoder_dim, epoch,DEVICE):
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

            # log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]

            dim_encode = encoder_inputs.shape[2]
            dim_decode = decoder_inputs.shape[2]
            encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
            decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
            # encode
            encoder_output = net.encode(encoder_inputs[:, :, :, [0,1,5,6]], encoder_inputs[:, :, :, 2], encoder_inputs[:, :, :, 3],encoder_time) #[:,:,:,[0,1,5,6]]

            pump_dim = decoder_dim#int(decoder_inputs.shape[-1]/2)
            decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features

            for step in range(predict_length):

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

            decoder_time_pumps = torch.count_nonzero(decoder_inputs[:, :, :, pump_dim] + 1, dim=1)
            weight_cf = torch.ones_like(predict_output[:, :, :, 0])
            if (decoder_time_pumps == 2).nonzero(as_tuple=False).shape[0] != 0:
                # print('here')
                for i in (decoder_time_pumps == 2).nonzero(as_tuple=False):
                    weight_cf[i[0], :, i[1]:i[1]+5]=5

            loss = criterion(predict_output[:, :, :, 0], labels[:, :, :, 0]) \
                   + 5*criterion( predict_output[:, :, :, 1],
                                   labels[:, :, :, 1]) \

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.4f, c_r: %.4f, l_hr: %.4f' % ( \
                    batch_index + 1, val_loader_length, loss.item(), c_r.item(), h_r.item()))#l_hr1: %.4f, h_r1.item()))


        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min,decoder_dim, params_path, type, index_adj,nrow,ncol):
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


                log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]
                dim_encode = encoder_inputs[:,:,:,:].shape[2]
                dim_decode = decoder_inputs[:,:,:,:].shape[2]
                encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
                decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
                # encode
                encoder_output = net.encode(encoder_inputs[:, :, :, [0,1,5,6]], log1, lat1,encoder_time) #[:,:,:,[0,1,5,6]]
                input.append(encoder_inputs[:, :, :, :].cpu().numpy())  # encoder_inputs[:, :, :, 0:1] (batch, T', 1)

                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                # decoder_pump_inputs = decoder_inputs[:, :, :, 1:]

                pump_dim = decoder_dim
                decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features


                for step in range(predict_length):

                    if step == 0:  # added
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, encoder_inputs[:, :, -1:,
                                                                                                :pump_dim],decoder_pump_inputs[:, :, 0:step + 1, 0],decoder_pump_inputs[:, :, 0:step + 1, 1],decoder_time[:, :, 0:step + 1])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                    else:  # added
                        predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + 1, -pump_dim:], encoder_output, torch.cat((encoder_inputs[:, :, -1:, :pump_dim], \
                                                                                                                            predict_output ),axis=-2), \
                             decoder_pump_inputs[:, :, 0:step + 1, 0], decoder_pump_inputs[:, :, 0:step + 1,
                                                                        1], decoder_time[:, :, 0:step + 1])


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



        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)

        prediction = re_max_min_normalization(prediction, _max1, _min1)

        target_comp = np.concatenate(target_comp, 0)#np.transpose(target_comp, (0, 1, 3, 2))

        target_comp = re_max_min_normalization(target_comp, _max1, _min1)



def load_graphdata_normY_channel(graph_signal_matrix_filename, num_of_lags,  decoder_output_size,DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    :param graph_signal_matrix_filename: str
    :param num_of_lags: int
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
                            file +  '_l' + str(num_of_lags) + '.npz')

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

    get_list = list(range(int(decoder_output_size)))
    train_decoder_input_p = np.concatenate((train_decoder_input_start, train_target_norm[:, :,get_list, :-1]), axis=3) #[0,1,3]

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
    # val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T) -> (B, N, F,T)
    val_decoder_input_p = np.concatenate((val_decoder_input_start, val_target_norm[:, :,get_list, :-1]), axis=3)
    val_decoder_input = np.concatenate((val_decoder_input_p,val_decoder_x), axis=-2)
    #  (B,N,T,F(2)) in pump and drawdown

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T) -> (B, N, F,T)
    #  (B,N,T,F(2)) in pump and drawdown
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:decoder_output_size, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入

    test_decoder_input_p = np.concatenate((test_decoder_input_start, test_target_norm[:, :,get_list, :-1]), axis=3)  # (B, N, T)

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
