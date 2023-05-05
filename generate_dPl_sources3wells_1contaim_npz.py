import numpy as np
import os
import pandas as pd
import read_binary_file
from matplotlib import pyplot as plt

def to_daily(data, dpstep):
    new_data = []
    for i in data:
        new_data.extend([i])
        new_data.extend([i] * dpstep)
    return new_data

def moving_average(a, n=4) :
    ret = np.cumsum(a, axis=0,dtype=float)
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    return ret[n - 1:,:] / n
import csv
workingdir = "E:\\aGNN\data\dPl3wells\\"

nrow = 15  # 132
ncol = 30  # 125
nlay = 1  # 5
str_period = 300
nsp = 7 * str_period  # *5###1
nlags_en = 20
nlags_pre = 20

num_data = 7


# for iq in range(1,3):
folder_w1 = 'E:\Contaminant_causal\GNN_modflow_nc_w2-train-1contaim-dK1_MODFLOW'  # +str(iq)+'_MODFLOW'
file_read_w1 = folder_w1 + '\GNN_modflow_nc_w2-train-1contaim-dK1.drw'  # +str(iq)+'.drw'#'\Pure_2Wells_w_recharge_dry_Model_prediction_w1.drw'
result_w1 = read_binary_file.read_binary_file(file_read_w1, nrow, ncol, nlay, nsp)

folder_w2 = 'E:\Contaminant_causal\GNN_modflow_nc_w2-train-1contaim-dK1_MT3DMS'#GNN_modflow_nc_w3homo-train1_MODFLOW_MT3DMS'#GNN_modflow_nc_w2-train_MT3DMS'  # +str(iq)+'_MODFLOW'
file_read_w2_1 = folder_w2 + '\MT3D001.UCN'  # +str(iq)+'.drw'#'\Pure_2Wells_w_recharge_dry_Model_prediction_w1.drw'
# file_read_w2_2 = folder_w2 + '\MT3D002.UCN'
result_w2_1 = read_binary_file.read_binary_file(file_read_w2_1, nrow, ncol, nlay, nsp)
# result_w2_2 = read_binary_file.read_binary_file(file_read_w2_2, nrow, ncol, nlay, nsp)
##############################################    concentrations and heads   ##############################################
pump_loc = [[11,20],[7,9],[5, 13], [4,
                      6]]
pump_loc1 = [ [7, 9], [5, 13], [4,6]]
#[[11,20],[7,9],[5, 13], [5,7]]  # [[18,54],[28,66],[40,70],[59,53],[49,65]]#[[51,60],[50,60],[49,60],[48,60],[47,60]]##[[52,59],[52,58],[52,57],[52,56],[52,55]] #trial_data_ddn_4.1 ##trial_data_ddn_4 [[12,30],[12,65],[11,29],[11,65],[25,31]],[12,38,[9,64],[9,61],[9,62],[9,63]
obs_ind = [[i, j] for i in range(1,nrow,2) for j in range(1,ncol,2) if result_w1[i, j, 0] > -999]
obs_ind = obs_ind+[[11,20],[4,6]]#,[4,8],[6,16],[7,11],[5,8],[8,10],[9,11]]

perifi_add = []
for pump_loc_i in pump_loc1:
    for peri_x in range(-1,2):
        for peri_y in range(-1,2):
            if [pump_loc_i[0]+peri_x,pump_loc_i[1]+peri_y] not in perifi_add:
                perifi_add.extend([[pump_loc_i[0]+peri_x,pump_loc_i[1]+peri_y]])
for perifi_add_i in perifi_add:
    if perifi_add_i not in obs_ind:
        obs_ind.extend([perifi_add_i])
# # # obs_ind = [[i, j] for i in range(1,nrow,2) for j in range(1,ncol,2) if result_w1[i, j, 0] > -999]
# # obs_ind = obs_ind+[[11,20]]
# # bc = [[i, j] for i in range(nrow) for j in range(ncol) if result_w1[i, j, 0] > -999]
nodes = len(obs_ind)
print ('nodes:',nodes)
index = {i: obs_ind for i, obs_ind in enumerate(obs_ind)}
obs_data = np.transpose(np.array([result_w1[ind[0], ind[1],:] for ind in obs_ind]))
# obs_data = np.expand_dims(obs_data,axis=-1)[:-52*7*2]
id_obs_data = { id:i for (id,i) in enumerate(obs_ind)}
obs_id_data = { tuple(i):id for (id,i) in enumerate(obs_ind)}
selectedh_data = np.transpose(np.array([result_w1[i[0],i[1],:] for (id,i) in enumerate(obs_ind)]))

every_num = 3


selectedh_data = np.mean(selectedh_data[30:2070,:].reshape(-1, every_num,nodes), axis=1)

selectedh_data[abs(selectedh_data)<0.001] = 0

selectedc_data1 = np.transpose(np.array([result_w2_1[i[0],i[1],:] for (id,i) in enumerate(obs_ind)]))

selectedc_data1 = np.mean(selectedc_data1[30:2070,:].reshape(-1, every_num,nodes), axis=1)

selectedc_data1[(selectedc_data1)<0.001] = 0

##############################################  end  concentrations and heads   ##############################################

##############################################    pumping   ##############################################
x = pd.read_csv('E:\small_physical_model\code\GNN_pumping_rates2.txt', sep=',') #GNN_pumping_rates1.txt
pump1 = x[x['name'] == 'well1']['Q']
# p1 = to_daily(pump1.values.tolist(), num_data - 1)
p1 = pump1
# p1 = len(p1)*[0]

# pumping 2
pump2 = x[x['name'] == 'well2']['Q']
# p2 = to_daily(pump2.values.tolist(), num_data - 1)
p2 = pump2

# pumping 2
pump3 = x[x['name'] == 'well3']['Q']
# p3 = to_daily(pump3.values.tolist(), num_data - 1)
p3 = pump3

# pumping 2
pump4 = x[x['name'] == 'well4']['Q']
# p4 = to_daily(pump4.values.tolist(), num_data - 1)
p4 = pump4

p1_data = np.array(p1)
p2_data = np.array(p2)
p3_data = np.array(p3)
p4_data = np.array(p4)


pumping_location = np.zeros_like(result_w1)
for id,i in enumerate(pump_loc):
    if id==0:
        pumping_location[i[0],i[1],:] = p1_data
    elif id ==1:
        pumping_location[i[0], i[1], :] = p2_data
    elif id ==2:
        pumping_location[i[0], i[1], :] = p3_data
    elif id ==3:
        pumping_location[i[0], i[1], :] = p4_data
selected_pumping = np.transpose(np.array([pumping_location[i[0],i[1],:] for i in obs_ind]))
# selected_pumping = np.mean(selected_pumping[int(52*0.8*7):int(52*0.8*7)+1800,:].reshape(-1, every_num,199), axis=1)
selected_pumping = np.mean(selected_pumping[30:2070,:].reshape(-1, every_num,nodes), axis=1)
# selected_pumping = selected_pumping[30:2070,:]
# selected_pumping = moving_average(selected_pumping,every_num)
# selected_pumping = selected_pumping[int(52*0.8*7)::1]
#################################   end pumping   #################################


#################################    contaiminant discharge  #################################
contaiminant_location1 = np.zeros_like(result_w1)
contaiminant_location2 = np.zeros_like(result_w1)
import copy
p2_data1 = copy.deepcopy(p2_data)
p3_data1 = copy.deepcopy(p3_data)
p4_data1 = copy.deepcopy(p4_data)
layer_no = 1
well_no = 4
free_month = 52
pump_week = 4
period_no =300
Q = {'Pump1':[],'Pump2':[],'Pump3':[],'Pump4':[],'Pump5':[]}
well_dict = {'Q': []}
for j in range(layer_no):
    for i in range(well_no):
        # if i % well_no <= 0:#26:
        #     pass
        if i % well_no <= 0:
            continue
        elif i % well_no <= 1: #"elif i % well_no <=1:#35:
            aa = [0] * free_month
            a2 = [100]*pump_week+[0]*(period_no-pump_week-free_month)
            aa.extend(a2)
            p1c =to_daily( aa[:period_no], num_data - 1)
            p2_data[p2_data>0]=70
            p2_data1[p2_data1 > 200] = 30
            p2_data1[(p2_data1>0) & (p2_data1 <30)] = 50
            p1c = list(p2_data)
            p1c1 = list(p2_data1)
            contaiminant_location1[pump_loc[i][0], pump_loc[i][1], :] = p1c
            contaiminant_location2[pump_loc[i][0], pump_loc[i][1], :] = p1c1
        elif i % well_no <=2:#2:#44:
            aa = [0] * free_month
            a3 = [100]*pump_week+[0]*(period_no-pump_week-free_month)
             # *i2/period_no
            aa.extend(a3)
            p2c = to_daily(aa[:period_no], num_data - 1)
            p3_data[(p3_data!=0)&(p3_data < 300)] = 60
            p3_data[p3_data >= 300] = 100
            p3_data1[(p3_data1!=0)&(p3_data1 < 500)] = 50
            p3_data1[p3_data1 >=500] = 40
            p2c = list(p3_data)
            p2c1 = list(p3_data1)
            contaiminant_location1[pump_loc[i][0], pump_loc[i][1], :] = p2c
            contaiminant_location2[pump_loc[i][0], pump_loc[i][1], :] = p2c1
        elif i % well_no <= 3:#3:#53:
            aa = [0] * free_month*2
            a4 = [50]*pump_week+[0]*26+[25]*pump_week+[0]*(period_no-pump_week-26-pump_week-free_month*2)
            aa.extend(a4)
            p3c = to_daily(aa[:period_no], num_data - 1)
            p4_data[p4_data > 0] = 50
            p4_data1[(p4_data1!=0)&(p4_data1 < 500)] = 40
            p4_data1[p4_data1 >= 500] = 20
            p3c = list(p4_data)
            p3c1 = list(p4_data1)
            contaiminant_location1[pump_loc[i][0], pump_loc[i][1], :] = p3c
            contaiminant_location2[pump_loc[i][0], pump_loc[i][1], :] = p3c1
selected_contam1 = np.transpose(np.array([contaiminant_location1[i[0],i[1],:] for i in obs_ind]))
selected_contam1 = np.mean(selected_contam1[30:2070,:].reshape(-1, every_num,nodes), axis=1)

#################################   end contaiminant discharge  #################################



nodes = selected_pumping.shape[1]
min_len = min(selectedc_data1.shape[0],selected_contam1.shape[0])
min_len = min(min_len,2500)
extra_len = 70
data_pd = np.zeros((min_len+extra_len,nodes,4))#6))#6))
data_pd[:,:,0]=np.concatenate((np.repeat(selectedh_data[:1,:],extra_len,axis=0),selectedh_data[:min_len])) # head
data_pd[:,:,1]=np.concatenate((np.repeat(selectedc_data1[:1,:],extra_len,axis=0),selectedc_data1[:min_len])) # concentration1
data_pd[:,:,2]=np.concatenate((np.repeat(selected_pumping[:1,:],extra_len,axis=0),selected_pumping[:min_len])) # pumping
data_pd[:,:,3]=np.concatenate((np.repeat(selected_contam1[:1,:],extra_len,axis=0),selected_contam1[:min_len])) # contaiminant


np.savez_compressed(workingdir+'One-contaim_dPl3wells.npz', data=data_pd)

pump_index = [45,51,90]#  #,179
dist_array = []
dist_matrix = np.zeros((nodes,nodes))
from collections import defaultdict
has_link_dict = defaultdict(list)


def Sort(sub_li):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key=lambda x: x[1], reverse=True)
    return sub_li
# dist_array.append([0,0,0])
for i_node in range(nodes):
    for j_node in range(nodes):
        if i_node == 45 and j_node ==51:
            print(i_node, j_node)
        # if i_node < self.n_obs and j_node< self.n_obs:
        # dist = np.sqrt((id_obs_data[i_node][0] - id_obs_data[j_node][0]) ** 2 +
        #                (id_obs_data[i_node][1] - id_obs_data[j_node][1]) ** 2)
        dist = np.sqrt((id_obs_data[i_node][0] - id_obs_data[j_node][0]) ** 2 +
                       (id_obs_data[i_node][1] - id_obs_data[j_node][1]) ** 2)
        if dist <= 3:#3:  # 1.5
            # if True:
            #     dist_array.append([i_node,j_node,4-dist]) #3-dist #27
            dist_matrix[i_node, j_node] = 4 - dist
            has_link_dict[i_node].append([j_node, 4 - dist])
        # if (i_node in pump_index) and dist>=2:
        #     dist_array.append([i_node, j_node, dist])

import copy

has_link_dict_t = copy.deepcopy(has_link_dict)
for i in has_link_dict.keys():
    Sort(has_link_dict[i])
    quadrant_check = {i_x: 0 for i_x in range(8)}
    for ij in has_link_dict[i]:
        n1 = index[i]
        n2 = index[ij[0]]
        if n1[0] < n2[0] and n1[1] < n2[1]:
            quadrant_check[0] += 1
        elif n1[0] < n2[0] and n1[1] > n2[1]:
            quadrant_check[1] += 1
        elif n1[0] > n2[0] and n1[1] < n2[1]:
            quadrant_check[2] += 1
        elif n1[0] > n2[0] and n1[1] > n2[1]:
            quadrant_check[3] += 1
        elif n1[0] == n2[0] and n1[1] > n2[1]:
            quadrant_check[4] += 1
        elif n1[0] == n2[0] and n1[1] < n2[1]:
            quadrant_check[5] += 1
        elif n1[0] > n2[0] and n1[1] == n2[1]:
            quadrant_check[6] += 1
        elif n1[0] < n2[0] and n1[1] == n2[1]:
            quadrant_check[7] += 1

        if any(ii > 1 for (id, ii) in quadrant_check.items()):
            index_to_delete = [n for (n, iii) in enumerate(quadrant_check.items()) if iii[1] > 1][0]  # [0]
            if ij in has_link_dict_t[i]: has_link_dict_t[i].remove(ij)
            if [i, ij[1]] in has_link_dict_t[ij[0]]: has_link_dict_t[ij[0]].remove([i, ij[1]])
            quadrant_check[index_to_delete] -= 1

has_link_dict = copy.deepcopy(has_link_dict_t)

for ij in has_link_dict.keys():
    has_link_dict[ij] = Sort(has_link_dict[ij])[:9]


for i_node in has_link_dict.keys():
    for ele in has_link_dict[i_node]:
        dist_array.append([i_node, ele[0], ele[1]])
    # if (i_node in pump_index) and dist>=2:
    #     dist_array.append([i_node, j_node, dist])

csvfile = workingdir+"One-contaim_dPl3wells.csv"
np.savetxt(csvfile, np.array(dist_array), delimiter=",")

