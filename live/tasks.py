from celery.result import allow_join_result
from celery.task import task
import pika
import re
import json
from collections import defaultdict, OrderedDict, deque
import numpy as np
from scalogram import *
from scipy.signal import argrelextrema
from scipy.stats import stats
from sklearn.externals import joblib
import pandas as pd
import requests

private_ip_pattern = re.compile("(^127\.)|(^10\.)|(^172\.1[6-9]\.)|(^172\.2[0-9]\.)|(^172\.3[0-1]\.)|(^192\.168\.)")

download = defaultdict(list)
upload = defaultdict(list)
batch_time = 0
last_timestamp_up = 0
last_timestamp_down = 0

label_map = ['netflix', 'youtube', 'acestream', 'twitch']
@task(name="extract_sample")
def extract_sample(sample_size=60, window=1):
    window = 1 / window

    # Scalogram
    N = sample_size
    dj = 1 / 128
    s0 = 2
    J = 1 / dj * np.log2(0.5 * N / s0)
    scales = s0 * 2 ** (np.arange(J) * dj)

    # Establish connection to queue
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    consumed_messages = 0
    channel.queue_declare(queue='pack', durable=True)
    channel.basic_qos(prefetch_count=1)

    channel.queue_declare(queue='predictions', durable=True)
    channel.basic_qos(prefetch_count=1)

    res = channel.queue_declare(
        queue='pack',
        durable=True,
        passive=True
    )

    messages_to_consume = res.method.message_count

    if messages_to_consume > 0:
        method_frame, header_frame, body = channel.basic_get('pack')
        consumed_messages += 1
        pkt = json.loads(body)
  #      print(" [x] Received packet")
  #      print(" [x] Done")
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        global batch_time, last_timestamp_up, last_timestamp_down, upload, download
        batch_time = int(pkt['timestamp'] * window)
        last_timestamp_up = int(pkt['timestamp'] * window)
        last_timestamp_down = int(pkt['timestamp'] * window)
        print("\n-------TIMESTAMP START------- ")
        print(pkt['timestamp'])

        download = defaultdict(list)
        upload = defaultdict(list)
        download[last_timestamp_up] = []
        upload[last_timestamp_up] = []

        redirect_packets(pkt, window, sample_size, False)
    else:
        print("No packets to process")

    print(messages_to_consume)
    for i in range(consumed_messages, messages_to_consume):
        method_frame, header_frame, body = channel.basic_get('pack')
        consumed_messages += 1
        pkt = json.loads(body)
#        print(" [x] Received packet")
#        print(" [x] Done")
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
#        print(str(consumed_messages) + "/" + str(messages_to_consume))
        if consumed_messages == messages_to_consume:
            redirect_packets(pkt, window, sample_size, True, scales=scales)
        else:
            redirect_packets(pkt, window, sample_size, False)


@task(ignore_result=True)
def predict(sample):
    '''
    sample = sample.as_matrix().astype(np.float)

    if np.any(np.isnan(sample)):
        print("There are NaNs.")
    if not np.all(np.isfinite(sample)):
        print("There are infinites.")
    '''
    sample = pd.read_json(sample)

    model = joblib.load('../models/brVsvid_best.sav')
    prediction =  model.predict(sample)
    print("------PREDICTION------")
    print(prediction)
    print("Video ->")
    model = joblib.load('../models/vidVsvid_best.sav')
    prediction = model.predict(sample)
    print(prediction)
    print("\t" + str(label_map[int(prediction[0])]))



def redirect_packets(pkt, window, sample_size, last_packet, scales=None):
   # print("Timestamp: " + str(pkt['timestamp']) + "\tLen: " + str(pkt['len']))

    global upload, download, batch_time, last_timestamp_down, last_timestamp_up
    if last_packet:
        if private_ip_pattern.match(pkt['src']):
            time_diff = int(pkt['timestamp'] * window) - last_timestamp_up
            if time_diff > 1:
                for i in range(1, time_diff):
                    upload[last_timestamp_up + i] = []
            last_timestamp_up = int(pkt['timestamp'] * window)
            upload[int(pkt['timestamp'] * window)].append(pkt)
        elif private_ip_pattern.match(pkt['dst']):
            time_diff = int(pkt['timestamp'] * window) - last_timestamp_down
            if time_diff > 1:
                for i in range(1, time_diff):
                    download[last_timestamp_down + i] = []
            last_timestamp_down = int(pkt['timestamp'] * window)
            download[int(pkt['timestamp'] * window)].append(pkt)
        else:
            print("Curious!\n", pkt)


        info = get_info(upload, download)
        nr_windows_up = len(info['up']['packet_count'])
        nr_windows_down = len(info['up']['packet_count'])
        if nr_windows_up < sample_size or nr_windows_down < sample_size:
            info['up']['packet_count'] = info['up']['packet_count'] + [0] * (sample_size - nr_windows_up)
            info['up']['byte_count'] = info['up']['byte_count'] + [0] * (sample_size - nr_windows_up)
            info['down']['packet_count'] = info['down']['packet_count'] + [0] * (sample_size - nr_windows_down)
            info['down']['byte_count'] = info['down']['byte_count'] + [0] * (sample_size - nr_windows_down)
      #  print("----------------------------")
      #  print("LenUp: " + str(len(info['up']['byte_count'])))
      #  print("LenDown: " + str(len(info['down']['byte_count'])))

      #  print('\n---INFO---\n')
        info = normalize_info(info, len(info['up']['byte_count']))

        sample = get_stats_from_info(info, scales)

        batch_time = int(pkt['timestamp'] * window)
        download = defaultdict(list)
        upload = defaultdict(list)
        download[last_timestamp_up] = []
        upload[last_timestamp_up] = []
        last_timestamp_up = int(pkt['timestamp'] * window)
        last_timestamp_down = int(pkt['timestamp'] * window)
        print("----TIMESTAMP END-----")
        print(pkt['timestamp'])

        sample = pd.read_json(sample)

        model = joblib.load('../models/brVsvid_best.sav')
        prediction = model.predict(sample)
        print("------PREDICTION------")
        print(prediction)
        print("Video ->")
        model = joblib.load('../models/vidVsvid_best.sav')
        prediction = model.predict(sample)
        print(prediction)
        app_name = str(label_map[int(prediction[0])])
        print("\t" + app_name)

        requests.put('http://localhost:5200/api/predictions/1', json={'prediction': app_name})


    else:
        if private_ip_pattern.match(pkt['src']):
            time_diff = int(pkt['timestamp'] * window) - last_timestamp_up
            if time_diff > 1:
                for i in range(1, time_diff):
                    upload[last_timestamp_up + i] = []
            last_timestamp_up = int(pkt['timestamp'] * window)
            upload[int(pkt['timestamp'] * window)].append(pkt)
        elif private_ip_pattern.match(pkt['dst']):
            time_diff = int(pkt['timestamp'] * window) - last_timestamp_down
            if time_diff > 1:
                for i in range(1, time_diff):
                    download[last_timestamp_down + i] = []
            last_timestamp_down = int(pkt['timestamp'] * window)
            download[int(pkt['timestamp'] * window)].append(pkt)
        else:
            print("Curious!\n", pkt)

def normalize_info(info, length):
    asnumpy = np.column_stack((
        info['up']['packet_count'],
        info['up']['byte_count'],
        info['down']['packet_count'],
        info['down']['byte_count']
    ))

    print(asnumpy)

    scaler = joblib.load('../models/std_scaler_1s_60.sav')
    scaled = scaler.fit_transform(asnumpy)
    up = defaultdict(list)
    down = defaultdict(list)

    result = {}
    up['packet_count'] = scaled[:, 0]
    up['byte_count'] = scaled[:, 1]
    down['packet_count'] = scaled[:, 2]
    down['byte_count'] = scaled[:, 3]
    result['up'] = up
    result['down'] = down
    return result

def get_info(up_pkts, down_pkts):
    up_ordered = OrderedDict(sorted(up_pkts.items(), key=lambda t: t[0]))
    down_ordered = OrderedDict(sorted(down_pkts.items(), key=lambda t: t[0]))

    result = {}
    up = defaultdict(list)
    down = defaultdict(list)

    for pkts in up_ordered.values():
        up['byte_count'].append(sum(int(pkt['len']) for pkt in pkts))
        up['packet_count'].append(len(pkts))
    for pkts in down_ordered.values():
        down['byte_count'].append(sum(int(pkt['len']) for pkt in pkts))
        down['packet_count'].append(len(pkts))
    result['up'] = up
    result['down'] = down
    return result


def calc_scalogram(data, scales):
    S, scales = scalogramCWT(data, scales)
    return S


# Get top X spikes from scalogram, sorted by value
def get_spikes(scalo, comparator, scales):
    len(scalo)
    spikes = deque([(-1, -1)] * 5, maxlen=5)
    # aux = argrelextrema(scalo, comparator, order=int(len(scalo)/10))
    aux = argrelextrema(scalo, comparator)
    if aux[0].size:
        for x in np.nditer(aux) or []:
            spikes.append((scalo[x], scales[x]))
    ordered = sorted(spikes, key=lambda x: x[1], reverse=True)
    values = np.hstack(zip(*ordered))
    return values


from itertools import groupby


def get_stats_numpy(data, zero):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    var = np.var(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    pc = [25, 50, 75, 90]
    percentiles = np.array(np.percentile(data, pc))
    silences = np.count_nonzero(np.asarray(data) == zero)
    silence_mean = np.mean(list(sum(1 for _ in g) for k, g in groupby(data) if k == zero))
    longest_silence = max(sum(1 for _ in g) for k, g in groupby(data) if k == 0) if silences > 0 else 0
    shortest_silence = min(sum(1 for _ in g) for k, g in groupby(data) if k == 0) if silences > 0 else 0

    # print("Mean: " + str(mean))
    # print("Media: " + str(median))
    # print("StdDev: " + str(std))
    # print("Variance: " + str(var))
    # print("Skewness: " + str(skew))
    # print("Kurtosis: " + str(kurt))
    # print("Pc25: " + str(percentiles[0]))
    # print("Pc50: " + str(percentiles[1]))
    # print("Pc75: " + str(percentiles[2]))

    features = np.hstack(
        (mean, median, std, var, skew, kurt, percentiles, silences, silence_mean, longest_silence, shortest_silence))

    
    return features


# Put it all on a numpy array
def get_features_numpy(info, local_max_up_bytes, local_min_up_bytes, local_max_up_packet, local_min_up_packet,
                       local_max_down_bytes, local_min_down_bytes, local_max_down_packet, local_min_down_packet, zeros):
    np.set_printoptions(suppress=True)

    result = np.hstack(
        (
            get_stats_numpy(info['up']['byte_count'], zeros[1]),
            get_stats_numpy(info['up']['packet_count'], zeros[0]),
            local_max_up_bytes, local_min_up_bytes,
            local_max_up_packet, local_min_up_packet,
            get_stats_numpy(info['down']['byte_count'], zeros[3]),
            get_stats_numpy(info['down']['packet_count'], zeros[2]),
            local_max_down_bytes, local_min_down_bytes,
            local_max_down_packet, local_min_down_packet,
        )
    )
    return result


def get_stats_from_info(info, scales):

    scalo_up = (calc_scalogram(np.asarray(info['up']['byte_count']), scales),
                calc_scalogram(np.asarray(info['up']['packet_count']), scales))

    scalo_down = (calc_scalogram(np.asarray(info['down']['byte_count']), scales),
                  calc_scalogram(np.asarray(info['down']['packet_count']), scales))

    #    show_scalo([scalos_down[idx], scalos_up[idx]], scales, ['r', 'b'])
    # smooth_down = np.convolve(scalo_down, np.ones(len(scalo_down)), mode='same')
    # smooth_up = np.convolve(scalo_up, np.ones(len(scalo_up)), mode='same')
    # show_scalo([smooth_down, smooth_up], scales, ['r', 'b'])

    # scalo, scales = calc_and_show(np.asarray(stats['down']['packet_count']), 'r')
    # scalo, scales = calc_and_show(np.asarray(stats['up']['packet_count']), 'b')

    local_max_up_bytes = get_spikes(scalo_up[0], np.greater, scales)
    local_min_up_bytes = get_spikes(scalo_up[0], np.less, scales)
    local_max_up_packet = get_spikes(scalo_up[1], np.greater, scales)
    local_min_up_packet = get_spikes(scalo_up[1], np.less, scales)

    local_max_down_bytes = get_spikes(scalo_down[0], np.greater, scales)
    local_min_down_bytes = get_spikes(scalo_down[0], np.less, scales)
    local_max_down_packet = get_spikes(scalo_down[1], np.greater, scales)
    local_min_down_packet = get_spikes(scalo_down[1], np.less, scales)

    zeros = []
    file_with_zeros = open('../models/zeros_60_1.txt')
    for line in file_with_zeros.readlines():
        zeros.append(float(line))


    sample = get_features_numpy(info, local_max_up_bytes, local_min_up_bytes, local_max_up_packet, local_min_up_packet,
                                local_max_down_bytes, local_min_down_bytes, local_max_down_packet,
                                local_min_down_packet, zeros)

   # print(sample)


    names = [
        'up_bytes_mean', 'up_bytes_median', 'up_bytes_std', 'up_bytes_var', 'up_bytes_skew', 'up_bytes_kurt',
        'up_bytes_perc25', 'up_bytes_perc50', 'up_bytes_perc75', 'up_bytes_perc90',
        'up_bytes_silences', 'up_bytes_silence_mean', 'up_bytes_longest_silence', 'up_bytes_shortest_silence',
        'up_packet_mean', 'up_packet_median', 'up_packet_std', 'up_packet_var', 'up_packet_skew', 'up_packet_kurt',
        'up_packet_perc25', 'up_packet_perc50', 'up_packet_perc75', 'up_packet_perc90',
        'up_packet_silences', 'up_packet_silence_mean', 'up_packet_longest_silence', 'up_packet_shortest_silence',
        'up_bytes_1max_y', 'up_bytes_2max_y', 'up_bytes_3max_y', 'up_bytes_4max_y', 'up_bytes_5max_y',
        'up_bytes_1max_x', 'up_bytes_2max_x', 'up_bytes_3max_x', 'up_bytes_4max_x', 'up_bytes_5max_x',
        'up_bytes_1min_y', 'up_bytes_2min_y', 'up_bytes_3min_y', 'up_bytes_4min_y', 'up_bytes_5min_y',
        'up_bytes_1min_x', 'up_bytes_2min_x', 'up_bytes_3min_x', 'up_bytes_4min_x', 'up_bytes_5min_x',
        'up_packet_1max_y', 'up_packet_2max_y', 'up_packet_3max_y', 'up_packet_4max_y', 'up_packet_5max_y',
        'up_packet_1max_x', 'up_packet_2max_x', 'up_packet_3max_x', 'up_packet_4max_x', 'up_packet_5max_x',
        'up_packet_1min_y', 'up_packet_2min_y', 'up_packet_3min_y', 'up_packet_4min_y', 'up_packet_5min_y',
        'up_packet_1min_x', 'up_packet_2min_x', 'up_packet_3min_x', 'up_packet_4min_x', 'up_packet_5min_x',

        'down_bytes_mean', 'down_bytes_median', 'down_bytes_std', 'down_bytes_var', 'down_bytes_skew',
        'down_bytes_kurt',
        'down_bytes_perc25', 'down_bytes_perc50', 'down_bytes_perc75', 'down_bytes_perc90',
        'down_bytes_silences', 'down_bytes_silence_mean', 'down_bytes_longest_silence', 'down_bytes_shortest_silence',
        'down_packet_mean', 'down_packet_median', 'down_packet_std', 'down_packet_var', 'down_packet_skew',
        'down_packet_kurt',
        'down_packet_perc25', 'down_packet_perc50', 'down_packet_perc75', 'down_packet_perc90',
        'down_packet_silences', 'down_packet_silence_mean', 'down_packet_longest_silence',
        'down_packet_shortest_silence',
        'down_bytes_1max_y', 'down_bytes_2max_y', 'down_bytes_3max_y', 'down_bytes_4max_y', 'down_bytes_5max_y',
        'down_bytes_1max_x', 'down_bytes_2max_x', 'down_bytes_3max_x', 'down_bytes_4max_x', 'down_bytes_5max_x',
        'down_bytes_1min_y', 'down_bytes_2min_y', 'down_bytes_3min_y', 'down_bytes_4min_y', 'down_bytes_5min_y',
        'down_bytes_1min_x', 'down_bytes_2min_x', 'down_bytes_3min_x', 'down_bytes_4min_x', 'down_bytes_5min_x',
        'down_packet_1max_y', 'down_packet_2max_y', 'down_packet_3max_y', 'down_packet_4max_y', 'down_packet_5max_y',
        'down_packet_1max_x', 'down_packet_2max_x', 'down_packet_3max_x', 'down_packet_4max_x', 'down_packet_5max_x',
        'down_packet_1min_y', 'down_packet_2min_y', 'down_packet_3min_y', 'down_packet_4min_y', 'down_packet_5min_y',
        'down_packet_1min_x', 'down_packet_2min_x', 'down_packet_3min_x', 'down_packet_4min_x', 'down_packet_5min_x'
    ]

    scalogram_1 = ['up_bytes_1max_y', 'up_bytes_1max_x', 'up_bytes_1min_y', 'up_bytes_1min_x',
                   'up_packet_1max_y', 'up_packet_1max_x', 'up_packet_1min_y', 'up_packet_1min_x', 'down_bytes_1max_y',
                   'down_bytes_1max_x', 'down_bytes_1min_y', 'down_bytes_1min_x', 'down_packet_1max_y',
                   'down_packet_1max_x', 'down_packet_1min_y', 'down_packet_1min_x']

    scalogram_2 = ['up_bytes_2max_y', 'up_bytes_2max_x', 'up_bytes_2min_y', 'up_bytes_2min_x',
                   'up_packet_2max_y', 'up_packet_2max_x', 'up_packet_2min_y', 'up_packet_2min_x', 'down_bytes_2max_y',
                   'down_bytes_2max_x', 'down_bytes_2min_y', 'down_bytes_2min_x', 'down_packet_2max_y',
                   'down_packet_2max_x', 'down_packet_2min_y', 'down_packet_2min_x']

    scalogram_3 = ['up_bytes_3max_y', 'up_bytes_3max_x', 'up_bytes_3min_y', 'up_bytes_3min_x',
                   'up_packet_3max_y', 'up_packet_3max_x', 'up_packet_3min_y', 'up_packet_3min_x', 'up_packet_3min_y',
                   'down_bytes_3max_y', 'down_packet_3min_y', 'down_bytes_3max_x', 'down_bytes_3min_y',
                   'down_bytes_3min_x', 'down_packet_3max_y', 'down_packet_3max_x', 'down_packet_3min_y',
                   'down_packet_3min_x']
    scalogram_4 = ['up_bytes_4max_y', 'up_bytes_4max_x', 'up_bytes_4min_y', 'up_bytes_4min_x',
                   'up_packet_4max_y', 'up_packet_4max_x', 'up_packet_4min_y', 'up_packet_4min_x', 'down_bytes_4max_y',
                   'down_bytes_4max_x', 'down_bytes_4min_y', 'down_bytes_4min_x', 'down_packet_4max_y',
                   'down_packet_4max_x', 'down_packet_4min_y', 'down_packet_4min_x']
    scalogram_5 = ['up_bytes_5max_y', 'up_bytes_5max_x', 'up_bytes_5min_y', 'up_bytes_5min_x',
                   'up_packet_5max_y', 'up_packet_5max_x', 'up_packet_5min_y', 'up_packet_5min_x', 'down_bytes_5max_y',
                   'down_bytes_5max_x', 'down_bytes_5min_y', 'down_bytes_5min_x', 'down_packet_5max_y',
                   'down_packet_5max_x', 'down_packet_5min_y', 'down_packet_5min_x']

    scalogram = scalogram_1 + scalogram_2 + scalogram_3 + scalogram_4 + scalogram_5

    df = pd.DataFrame(np.asmatrix(sample), columns=names)

    # Not necessary to have silences in both 'bytes' and 'packet'
    df.drop(columns=['up_packet_silence_mean', 'down_packet_silence_mean',
                     'down_packet_longest_silence', 'down_packet_shortest_silence']+ scalogram, inplace=True)
   # print(df.columns)

    imputer = joblib.load('../models/imputer.sav')
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)

    return df.to_json()
