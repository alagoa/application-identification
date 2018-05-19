import pyshark as ps
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks_cwt
from scipy.stats import stats
import numpy as np
import re
import pprint as pp
from datetime import datetime
from collections import OrderedDict, defaultdict, deque
from scalogram import *
from itertools import groupby
import os

def get_info(up_pkts, down_pkts):
    up_ordered = OrderedDict(sorted(up_pkts.items(), key=lambda t: t[0]))
    down_ordered= OrderedDict(sorted(down_pkts.items(), key=lambda t: t[0]))    

    result = {}
    up = defaultdict(list)
    down = defaultdict(list)
    byte_count = defaultdict(list)

    for pkts in up_ordered.values():
        up['byte_count'].append(sum(int(pkt.captured_length) for pkt in pkts))
        up['packet_count'].append(len(pkts))
    for pkts in down_ordered.values():
        down['byte_count'].append(sum(int(pkt.captured_length) for pkt in pkts))
        down['packet_count'].append(len(pkts))
    result['up'] = up
    result['down'] = down
    return result

def redirect_packets(pkt):
    if hasattr(pkt, 'ip'):
        global batch_time
        global last_timestamp_up
        global last_timestamp_down
        global download, upload
        global window
        window = 1
        window = 1/window
        if (int(pkt.sniff_time.timestamp() * window) - batch_time) >= step:
            info.append(get_info(upload, download))
            batch_time = int(pkt.sniff_time.timestamp() * window)
            download = defaultdict(list)
            upload = defaultdict(list)
            download[last_timestamp_up] = []
            upload[last_timestamp_up] = []
            last_timestamp_up = int(pkt.sniff_time.timestamp() * window)
            last_timestamp_down = int(pkt.sniff_time.timestamp() * window)
        if private_ip_pattern.match(pkt.ip.src.get_default_value()):
            time_diff = int(pkt.sniff_time.timestamp() * window) - last_timestamp_up
            if time_diff > 1:
                for i in range(1, time_diff):
                    upload[last_timestamp_up + i] = []
            last_timestamp_up = int(pkt.sniff_time.timestamp() * window)
            upload[int(pkt.sniff_time.timestamp() * window)].append(pkt)
        elif private_ip_pattern.match(pkt.ip.dst.get_default_value()):
            time_diff = int(pkt.sniff_time.timestamp() * window) - last_timestamp_down
            if time_diff > 1:
                for i in range(1, time_diff):
                    download[last_timestamp_down + i] = []
            last_timestamp_down = int(pkt.sniff_time.timestamp() * window)
            download[int(pkt.sniff_time.timestamp() * window)].append(pkt)
        else:
            print("Curious!\n", pkt)
    elif hasattr(pkt, 'ipv6'):
        print("not yet implemented")
        # TODO
    global count
    print(count, end="\r")
    count += 1

#N=len(data)

def calc_scalogram(data, scales):    
    S,scales= scalogramCWT(data,scales)
    return S

def show_scalo(data, scales, colors):
    for i in range (0, len(data)):
        plt.plot(scales, data[i], colors[i], lw=3)
    plt.show()

# Get top X spikes from scalogram, sorted by value
def get_spikes(scalo, comparator):
    len(scalo)
    spikes = deque([(-1,-1)] * 5, maxlen=5)
    #aux = argrelextrema(scalo, comparator, order=int(len(scalo)/10))
    aux = argrelextrema(scalo, comparator)
    if aux[0].size:
        for x in np.nditer(aux) or []:
            spikes.append((scalo[x], scales[x]))
    ordered = sorted(spikes, key=lambda x: x[1], reverse=True)
    values = np.hstack(zip(*ordered))
    return values

def get_stats_numpy(data):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    var = np.var(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    pc = [25,50,75]
    percentiles = np.array(np.percentile(data, pc))
    silences = np.count_nonzero(np.asarray(data)==0.0)
    longest_silence = max(sum(1 for _ in g) for k, g in groupby(data) if k==0) if silences > 0 else 0
    shortest_silence = min(sum(1 for _ in g) for k, g in groupby(data) if k==0) if silences > 0 else 0
    
    #print("Mean: " + str(mean))
    #print("Media: " + str(median))
    #print("StdDev: " + str(std))
    #print("Variance: " + str(var))
    #print("Skewness: " + str(skew))
    #print("Kurtosis: " + str(kurt))
    #print("Pc25: " + str(percentiles[0]))
    #print("Pc50: " + str(percentiles[1]))
    #print("Pc75: " + str(percentiles[2]))
    
    features = np.hstack((mean, median, std, var, skew, kurt, percentiles, silences, longest_silence, shortest_silence))
    return features

def get_stats_json(data):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    var = np.var(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    pc = [25,50,75]
    percentiles = np.array(np.percentile(data, pc))
    silences = np.count_nonzero(np.asarray(data)==0.0)
    longest_silence = max(sum(1 for _ in g) for k, g in groupby(data) if k==0) if silences > 0 else 0
    shortest_silence = min(sum(1 for _ in g) for k, g in groupby(data) if k==0) if silences > 0 else 0
    #print("Mean: " + str(mean))
    #print("Media: " + str(median))
    #print("StdDev: " + str(std))
    #print("Variance: " + str(var))
    #print("Skewness: " + str(skew))
    #print("Kurtosis: " + str(kurt))
    #print("Pc25: " + str(percentiles[0]))
    #print("Pc50: " + str(percentiles[1]))
    #print("Pc75: " + str(percentiles[2]))
    
    statistiscs = {
        'mean': mean,
        'median': median,
        'std': std,
        'var': var,
        'skew': skew,
        'kurt': kurt,
        'pc25': percentiles[0],
        'pc50': percentiles[1],
        'pc75': percentiles[2],
    }
    
    return statistiscs


    # Put it all on a numpy array
def get_features_numpy(info):
    np.set_printoptions(suppress=True)

    result = []

    for idx in range(0, len(info)):
        result.append(
            np.hstack(
                (
                 get_stats_numpy(info[idx]['up']['byte_count']),
                 get_stats_numpy(info[idx]['up']['packet_count']),
                 local_max_up_bytes[idx], local_min_up_bytes[idx],
                 local_max_up_packet[idx], local_min_up_packet[idx],
                 get_stats_numpy(info[idx]['down']['byte_count']),
                 get_stats_numpy(info[idx]['down']['packet_count']),
                 local_max_down_bytes[idx], local_min_down_bytes[idx],
                 local_max_down_packet[idx], local_min_down_packet[idx],

            ))
        )
    return result


# Put it all on a json
def get_features_json(info):
    stat = {
        'down': defaultdict(list),
        'up': defaultdict(list)
    }

    result = []
    for idx in range(0, len(info)):
        stat['down']['byte_count'] = get_stats_json(info[idx]['down']['byte_count'])
        stat['down']['packet_count'] = get_stats_json(info[idx]['down']['packet_count'])
        stat['down']['byte_count']['scalo_spikes_max'] =  local_max_down_bytes[idx]
        stat['down']['byte_count']['scalo_spikes_min'] =  local_min_down_bytes[idx]
        stat['down']['packet_count']['scalo_spikes_max'] =  local_max_down_packet[idx]
        stat['down']['packet_count']['scalo_spikes_min'] =  local_min_down_packet[idx]

        stat['up']['byte_count'] = get_stats_json(info[idx]['up']['byte_count'])
        stat['up']['packet_count'] = get_stats_json(info[idx]['up']['packet_count'])
        stat['up']['byte_count']['scalo_spikes_max'] =  local_max_up_bytes[idx]
        stat['up']['byte_count']['scalo_spikes_min'] =  local_min_up_packet[idx]
        stat['up']['packet_count']['scalo_spikes_max'] =  local_max_up_packet[idx]
        stat['up']['packet_count']['scalo_spikes_min'] =  local_min_up_packet[idx]


        result.append(stat)
    return result


def pcap_to_csv(path, filename):

    cap = ps.FileCapture(os.path.join(path, filename))
    #cap.load_packets()
    
    private_ip_pattern = re.compile("(^127\.)|(^10\.)|(^172\.1[6-9]\.)|(^172\.2[0-9]\.)|(^172\.3[0-1]\.)|(^192\.168\.)")
    step = 30 # batches of 30 seconds
    info = []
    window = 1 # 1 second
    window = 1/window

    batch_time = int(cap[0].sniff_time.timestamp() * window)
    last_timestamp_up = int(cap[0].sniff_time.timestamp() * window)
    last_timestamp_down = int(cap[0].sniff_time.timestamp() * window)

    download = defaultdict(list)
    upload = defaultdict(list)
    download[last_timestamp_up] = []
    upload[last_timestamp_up] = []
    count = 0
    
    cap.apply_on_packets(redirect_packets)
    pp.pprint(info)
    
    scalos_up = []
    scalos_down = []

    N = step
    dj=1/128
    s0=2
    J=1/dj * np.log2(0.5*N/s0)
    scales=s0*2**(np.arange(J)*dj)

    for idx, sample in enumerate(info):
        scalos_up.append(
            (calc_scalogram(np.asarray(sample['up']['byte_count']), scales),
             calc_scalogram(np.asarray(sample['up']['packet_count']), scales))
        )
        scalos_down.append(
            (calc_scalogram(np.asarray(sample['down']['byte_count']), scales),
             calc_scalogram(np.asarray(sample['down']['packet_count']), scales))
        )
    #    show_scalo([scalos_down[idx], scalos_up[idx]], scales, ['r', 'b'])
    #smooth_down = np.convolve(scalo_down, np.ones(len(scalo_down)), mode='same')
    #smooth_up = np.convolve(scalo_up, np.ones(len(scalo_up)), mode='same')
    #show_scalo([smooth_down, smooth_up], scales, ['r', 'b'])

    #scalo, scales = calc_and_show(np.asarray(stats['down']['packet_count']), 'r')
    #scalo, scales = calc_and_show(np.asarray(stats['up']['packet_count']), 'b')
    
    local_max_up_bytes = []
    local_min_up_bytes = []
    local_max_up_packet = []
    local_min_up_packet = []
    local_max_down_bytes = []
    local_min_down_bytes = []
    local_max_down_packet = []
    local_min_down_packet = []


    for scalo in scalos_up:
        local_max_up_bytes.append(get_spikes(scalo[0], np.greater))
        local_min_up_bytes.append(get_spikes(scalo[0], np.less))
        local_max_up_packet.append(get_spikes(scalo[1], np.greater))
        local_min_up_packet.append(get_spikes(scalo[1], np.less))

    for scalo in scalos_down:
        local_max_down_bytes.append(get_spikes(scalo[0], np.greater))
        local_min_down_bytes.append(get_spikes(scalo[0], np.less))
        local_max_down_packet.append(get_spikes(scalo[1], np.greater))
        local_min_down_packet.append(get_spikes(scalo[1], np.less))
        
    import pandas as pd

    samples = get_features_numpy(info)

    names = [
        'up_bytes_mean', 'up_bytes_median', 'up_bytes_std', 'up_bytes_var', 'up_bytes_skew', 'up_bytes_kurt',
        'up_bytes_perc25', 'up_bytes_perc50', 'up_bytes_perc75',
        'up_bytes_silences', 'up_bytes_longest_silence', 'up_bytes_shortest_silence',
        'up_packet_mean', 'up_packet_median', 'up_packet_std', 'up_packet_var', 'up_packet_skew', 'up_packet_kurt',
        'up_packet_perc25', 'up_packet_perc50', 'up_packet_perc75',
        'up_packet_silences', 'up_packet_longest_silence', 'up_packet_shortest_silence',
        'up_bytes_1max_y', 'up_bytes_2max_y', 'up_bytes_3max_y', 'up_bytes_4max_y', 'up_bytes_5max_y',
        'up_bytes_1max_x', 'up_bytes_2max_x', 'up_bytes_3max_x', 'up_bytes_4max_x', 'up_bytes_5max_x',
        'up_bytes_1min_y', 'up_bytes_2min_y', 'up_bytes_3min_y', 'up_bytes_4min_y', 'up_bytes_5min_y',
        'up_bytes_1min_x', 'up_bytes_2min_x', 'up_bytes_3min_x', 'up_bytes_4min_x', 'up_bytes_5min_x',
        'up_packet_1max_y', 'up_packet_2max_y', 'up_packet_3max_y', 'up_packet_4max_y', 'up_packet_5max_y',
        'up_packet_1max_x', 'up_packet_2max_x', 'up_packet_3max_x', 'up_packet_4max_x', 'up_packet_5max_x',
        'up_packet_1min_y', 'up_packet_2min_y', 'up_packet_2min_y', 'up_packet_4min_y', 'up_packet_5min_y',
        'up_packet_1min_x', 'up_packet_2min_x', 'up_packet_3min_x', 'up_packet_4min_x', 'up_packet_5min_x',

        'down_bytes_mean', 'down_bytes_median', 'down_bytes_std', 'down_bytes_var', 'down_bytes_skew', 'down_bytes_kurt',
        'down_bytes_perc25', 'down_bytes_perc50', 'down_bytes_perc75',
        'down_bytes_silences', 'down_bytes_longest_silence', 'down_bytes_shortest_silence',
        'down_packet_mean', 'down_packet_median', 'down_packet_std', 'down_packet_var', 'down_packet_skew', 'down_packet_kurt',
        'down_packet_perc25', 'down_packet_perc50', 'down_packet_perc75',  
        'down_packet_silences', 'down_packet_longest_silence', 'down_packet_shortest_silence',
        'down_bytes_1max_y', 'down_bytes_2max_y', 'down_bytes_3max_y', 'down_bytes_4max_y', 'down_bytes_5max_y',
        'down_bytes_1max_x', 'down_bytes_2max_x', 'down_bytes_3max_x', 'down_bytes_4max_x', 'down_bytes_5max_x',
        'down_bytes_1min_y', 'down_bytes_2min_y', 'down_bytes_3min_y', 'down_bytes_4min_y', 'down_bytes_5min_y',
        'down_bytes_1min_x', 'down_bytes_2min_x', 'down_bytes_3min_x', 'down_bytes_4min_x', 'down_bytes_5min_x',
        'down_packet_1max_y', 'down_packet_2max_y', 'down_packet_3max_y', 'down_packet_4max_y', 'down_packet_5max_y',
        'down_packet_1max_x', 'down_packet_2max_x', 'down_packet_3max_x', 'down_packet_4max_x', 'down_packet_5max_x',
        'down_packet_1min_y', 'down_packet_2min_y', 'down_packet_2min_y', 'down_packet_4min_y', 'down_packet_5min_y',
        'down_packet_1min_x', 'down_packet_2min_x', 'down_packet_3min_x', 'down_packet_4min_x', 'down_packet_5min_x'
    ]


    df = pd.DataFrame(samples, columns=names)
    
    # Not necessary to have silences in both 'bytes' and 'packet'
    df.drop(columns=['down_packet_silences', 'up_packet_silences', 'up_packet_longest_silence', 'up_packet_shortest_silence'], inplace=True)
    #df.describe()
    
    df['label'] = os.path.basename(path)
    out = 'csv/30s1s/' + filename.split('.')[0] + '.csv'
    df.to_csv(out, sep=',', encoding='utf-8')

base_pcap = "../../../shared/"

for path, subdirs, files in os.walk(base_pcap):
    for name in files:
        print(os.path.join(str(path), str(name)))
        pcap_to_csv(path, name)