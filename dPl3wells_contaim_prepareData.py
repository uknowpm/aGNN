import os
import numpy as np
import argparse
import configparser


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def search_data( num_of_depend, label_start_idx,
                ):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    return [(label_start_idx,label_start_idx+num_of_depend)]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None


    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  1, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        num_of_depend = num_of_hours
        hour_indices = search_data( num_of_depend,
                                   label_start_idx)
        # hour_indices = search_data(data_sequence.shape[0], num_of_hours,
        #                            label_start_idx, num_for_predict,
        #                            1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    # target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    target = data_sequence[label_start_idx+num_of_depend: label_start_idx+num_of_depend + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def MinMaxnormalization(train, val, test,flag):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    all = np.concatenate((train, val, test),axis=0)
    _max = all.max(axis=(0, 1, 3), keepdims=True) #train
    _min = all.min(axis=(0, 1, 3), keepdims=True)

    if flag =='encoder':
        # concen_a = np.array(all[:, :, 1, :])#np.concatenate((all[:, :, 1, :], all[:, :, 2, :]), axis=0)
        # discharge_a = np.array(all[:, :, 3, :])#np.concatenate((all[:, :, 4, :], all[:, :, 5, :]), axis=0)
        # concen_a_max = concen_a.max(axis=(0, 1, 2))
        # discharge_a_max = discharge_a.max(axis=(0, 1, 2))
        # _max[0, 0, 1:2, 0] = concen_a_max
        # _max[0, 0, 2:3, 0] = discharge_a_max
        # concen_a_min = concen_a.min(axis=(0, 1, 2))
        # discharge_a_min = discharge_a.min(axis=(0, 1, 2))
        # _min[0, 0, 1:2, 0] = concen_a_min
        # _min[0, 0, 2:3, 0] = discharge_a_min
        with open('GNN_max_en_c1.npy', 'wb') as f:
            np.save(f, _max)
        with open('GNN_min_en_c1.npy', 'wb') as f:
            np.save(f, _min)
    elif flag == 'decoder':
        with open('GNN_max_en_c1.npy', 'rb') as f:
            _max = np.load(f)[:,:,2:,:]# np.save(f, _max)
        with open('GNN_min_en_c1.npy', 'rb') as f:
            _min = np.load(f)[:,:,2:,:] #_min = np.save(f, _min)
    # with open('GNN_max.npy', 'rb') as f:
    #     a = np.load(f)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename,
                                              num_of_weeks, num_of_days,
                                              num_of_hours, num_for_predict,
                                              points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

    all_samples = []
    for idx in range(data_seq.shape[0]-num_of_hours-num_for_predict):
        # if idx == 353 or idx ==350:
        #     print('d')
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        decoder = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 2:, :] # (1,N,T) as in pumping stategy at current time
        t1 = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0:2, :]
        # t2 = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0:3:2, :]# (1,N,T)
        target =  t1#np.concatenate((t1,t2),axis =2)

        sample.append(target)
        sample.append(decoder)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        decoder_nonzero = np.nonzero(decoder[0, :, 1, :])
        length_decoder_nonzero = len(decoder_nonzero[0])

        encoder_nonzero = np.nonzero(hour_sample[0, :, 1, :])
        length_encoder_nonzero = len(encoder_nonzero[0])

        from collections import defaultdict
        lib_nonzero = {}
        lib_nonzero = defaultdict(lambda: 0,lib_nonzero)

        all_samples.append(
                    sample)


    split_line1 = int(len(all_samples) * 0.8)
    split_line2 = int(len(all_samples) * 0.9)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line2])]#zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-3], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-3], axis=-1)
    test_x = np.concatenate(testing_set[:-3], axis=-1)

    train_target = training_set[-3]  # (B,N,T)
    val_target = validation_set[-3]
    test_target = testing_set[-3]

    train_dx = training_set[-2]#np.expand_dims(training_set[-2],axis=-2)  # (B,N,T)
    val_dx = validation_set[-2]#np.expand_dims(validation_set[-2],axis=-2)
    test_dx = testing_set[-2]#np.expand_dims(testing_set[-2],axis=-2)

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # max-min normalization on x
    (stats, train_x_norm, val_x_norm, test_x_norm) = MinMaxnormalization(train_x, val_x, test_x,'encoder')
    (dstats, train_dx_norm, val_dx_norm, test_dx_norm) = MinMaxnormalization(train_dx, val_dx, test_dx,'decoder')

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'decoder_x':train_dx_norm,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'decoder_x': val_dx_norm,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'decoder_x': test_dx_norm,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_max': stats['_max'],
            '_min': stats['_min'],
            '_dmax': dstats['_max'],
            '_dmin': dstats['_min'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train decoder_x:', all_data['train']['decoder_x'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val decoder_x:', all_data['val']['decoder_x'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test decoder_x:', all_data['test']['decoder_x'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data max :', stats['_max'].shape, stats['_max'])
    print('train data min :', stats['_min'].shape, stats['_min'])
    print('train data decoder max :', dstats['_max'].shape, dstats['_max'])
    print('train data decoder min :', dstats['_min'].shape, dstats['_min'])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath,
                                file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_decoder_x=all_data['train']['decoder_x'],train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_decoder_x=all_data['val']['decoder_x'],val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_decoder_x=all_data['test']['decoder_x'],test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_max'], std=all_data['stats']['_min'],
                            dmean=all_data['stats']['_dmax'], dstd=all_data['stats']['_dmin'],
                            )
    return all_data


# prepare dataset
workingdir = "E:\ASTGNN-main\\"
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=workingdir+'configurations\dPl3wells_1contaim.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
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
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
data = np.load(graph_signal_matrix_filename)
# data['data'].shape

all_data = read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)
