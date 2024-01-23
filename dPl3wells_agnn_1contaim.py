#!/usr/bin/env python
# coding: utf-8
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.aGNN_dpl3_contam import make_model
from lib.utils_dpl3_contam import get_adjacency_matrix, get_adjacency_matrix_2direction, compute_val_loss, predict_and_save_results, load_graphdata_normY_channel
from tensorboardX import SummaryWriter
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/dPl3wells_1contaim.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')#
print("CUDA:", USE_CUDA, DEVICE, flush=True)

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
points_per_lags = int(data_config['points_per_lags'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
num_of_lags = int(training_config['num_of_lags'])
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch'])
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = int(training_config['batch_size'])
print('batch_size:', batch_size, flush=True)

direction = int(training_config['direction'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])
num_of_hours= 20
num_of_days = 0
num_of_weeks = 0
filename_npz = os.path.join(dataset_name + '_l' + str(points_per_lags)) + '.npz'
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])
ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
SE = bool(int(training_config['SE']))  # whether use spatial embedding
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

if direction == 2:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
if direction == 1:
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
folder_dir = 'MAE_%s_l%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (model_name, num_of_lags, num_layers, nb_head, d_model, encoder_input_size, direction, dropout, learning_rate)

if aware_temporal_context:
    folder_dir = folder_dir+'Tcontext'
if ScaledSAt:
    folder_dir = folder_dir + 'ScaledSAt'
if SE:
    folder_dir = folder_dir + 'SE' + str(smooth_layer_num)
if TE:
    folder_dir = folder_dir + 'TE'

dim_space = adj_mx.shape[1]

print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('../experiments', dataset_name, folder_dir)

decoder_output_size = 2
# all the input has been normalized into range [-1,1] by MaxMin normalization
train_loader, train_target_tensor,train_timestamp, val_loader, val_target_tensor, val_timestamp,test_loader, test_target_tensor,test_timestamp,  _max, _min\
    = load_graphdata_normY_channel(
    graph_signal_matrix_filename, num_of_lags,decoder_output_size,DEVICE, batch_size)

nrow, ncol = 15,30
obs_ind = np.loadtxt('obs_ind.txt')
nodes = len(obs_ind)
print('nodes:', nodes)
index = {i: obs_ind for i, obs_ind in enumerate(obs_ind)}
logitudelatitudes = obs_ind
net = make_model(DEVICE, logitudelatitudes, num_layers, encoder_input_size, decoder_input_size,decoder_output_size, d_model, distance_mx, nb_head,
                 num_of_lags, points_per_lags, num_for_predict, dropout=dropout, aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
# points_per_hour
print(net, flush=True)
# pp=0
# for p in list(net.parameters()):
#     nn=1
#     for s in list(p.size()):
#         nn = nn*s
#     pp += nn
def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):  # 从头开始训练，就要重新构建文件夹
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):  # 从中间开始训练，就要保证原来的目录存在
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')

    criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    # opt_para = list(net.parameters())+list(K_net.parameters())# + [dh_lt]  #
    optimizer = optim.Adam(net.parameters(), lr=learning_rate) #net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    total_param = 0
    print('Net\'s state_dict:', flush=True)
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param, flush=True)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    # train model
    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch, flush=True)

        print('load weight from: ', params_filename, flush=True)

    start_time = time()

    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, decoder_output_size,epoch,DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

        net.train()  # ensure dropout layers are in train mode

        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.transpose(-1, -2)#.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.transpose(-1, -2)

            optimizer.zero_grad()

            log1, lat1= decoder_inputs[:,:,:,2],decoder_inputs[:,:,:,3]
            dim_encode = encoder_inputs.shape[2]
            dim_decode = decoder_inputs.shape[2]
            encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
            decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()

            outputs = net(encoder_inputs[:,:,:,[0,1,5,6]], decoder_inputs[:,:,:,[0,1,5,6]],log1,lat1,encoder_time,decoder_time) #decoder_inputs[:, :, :, 2:]

            loss1 = criterion( outputs[:, :, :, 0],  labels[:, :, :, 0])
            loss2 = criterion( outputs[:, :, :, 1],  labels[:, :, :, 1])

            loss = criterion(outputs[:, :, :, 0], labels[:, :, :, 0])\
                    +5*criterion(outputs[:, :, :, 1], labels[:, :, :, 1])

            loss.backward()
            optimizer.step()

            #save loss_hpde
            training_loss = loss.item()
            #
            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)
            if batch_index == len(train_loader.dataset)-1:
                print('training_loss',training_loss, global_step)

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)
        print('epoch: %s, loss_h:%.4f, loss_c:%.4f' % (epoch, loss1, loss2), flush=True)  #


    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader, test_target_tensor, _max, _min,decoder_output_size, 'test')#test_target_tensor,

    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    print('fine tune the model ... ', flush=True)
    for epoch in range(epochs, epochs+fine_tune_epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        net.train()  # ensure dropout layers are in train mode

        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.transpose(-1, -2)#.unsqueeze(-1)  # (B, N, T, 1)


            predict_length = labels.shape[-1]  # T
            labels = labels.transpose(-1, -2)

            optimizer.zero_grad()

            pump_dim = decoder_output_size#int(decoder_inputs.shape[-1]/2)
            decoder_pump_inputs = decoder_inputs[:, :, :, pump_dim:]  # 2 features

            log1, lat1 = decoder_inputs[:, :, :, 2], decoder_inputs[:, :, :,3]
            dim_encode = encoder_inputs[:,:,:,4].shape[2]
            dim_decode = decoder_inputs[:,:,:,4].shape[2]
            encoder_time = torch.tensor([[list(range(dim_encode))]*encoder_inputs.shape[1]]*encoder_inputs.shape[0]).cuda()
            decoder_time = torch.tensor([[list(range(dim_decode))]*decoder_inputs.shape[1]]*decoder_inputs.shape[0]).cuda()
            # encode
            encoder_output = net.encode(encoder_inputs[:, :, :, [0,1,5,6]], encoder_inputs[:, :, :, 2], encoder_inputs[:, :, :, 3],encoder_time) #[:,:,:,[0,1,5,6]]

            interval = 1
            k = randint(0, interval-1)
            # 按着时间步进行预测

            for step in range(k,predict_length-1,interval):
                if step == k:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :,0:step + interval-k, -pump_dim:], encoder_output, encoder_inputs[:, :,  -interval:,
                                                                                            :pump_dim],decoder_pump_inputs[:, :, 0:step + interval-k, 0],
                                                 decoder_pump_inputs[:, :, 0:step + interval-k, 1],decoder_time[:, :, 0:step + interval-k])  # decoder_inputs[:,:,:,2:], encoder_output,encoder_inputs[:,:,-1:,:2])
                else:  # added
                    predict_output = net.decode1(decoder_pump_inputs[:, :, 0:step + interval-k, -pump_dim:], encoder_output, torch.cat((encoder_inputs[:, :,  -interval:, :pump_dim], \
                                                                                                                        predict_output ),axis=-2), \
                         decoder_pump_inputs[:, :, 0:step + interval-k, 0], decoder_pump_inputs[:, :, 0:step + interval-k,
                                                                    1], decoder_time[:, :, 0:step + interval-k]
                                                 ) #torch.cat((predict_output[:, :, :, :2],predict_output[:, :, :, 3:4]),axis=-1)

            loss = criterion( predict_output[:, :, :, 0],
                      labels[:, :, 0:step + interval-k, 0]) \
                    + 5 * criterion(predict_output[:, :, :, 1],
                             labels[:, :, 0:step + interval-k, 1])
            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

        # apply model on the validation data set
        val_loss =  compute_val_loss(net, val_loader, criterion, sw, decoder_output_size,epoch,DEVICE)#

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader,  test_target_tensor, _max, _min, decoder_output_size,'test')#test_target_tensor,


def predict_main(epoch, data_loader, data_target_tensor, _max, _min, decoder_dim,type): #data_target_tensor
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    ##############################################    concentrations and heads   ##############################################


    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

    print('load weight from:', params_filename, flush=True)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, decoder_dim,params_path, type, index,
                             nrow, ncol)


if __name__ == "__main__":

    train_main()

    # predict_main(385, train_loader, train_target_tensor, _max, _min,decoder_output_size, 'test')

    # predict_main(399, test_loader, test_target_tensor, _max, _min,decoder_output_size, 'train')












