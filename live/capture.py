import pyshark as ps
import pika
import json
from libcap import *

class LiveCap():
    def __init__(self):

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()

    def start(self):
        print("Starting live capture...")
        devices = pcap_findalldevs( )
        device = devices[0]
        snaplen = 65535
        promisc = 1
        to_ms = 10000
        print("Capturing on %s" % device)
        handle = pcap_open_live(device, snaplen, promisc, to_ms)
        if handle:
            bpf = bpf_program()
            result = pcap_setfilter(handle, bpf)
            print("pcap_setfilter -> %r" % result)
            pcap_loop(handle, -1, self.my_callback, None)
        print("Closing live capture")
        pcap_close(handle)

    def my_callback(self, pkthdr, data, user_data):
        ip_header = data[ETHERNET_HDR_SIZE:20 + ETHERNET_HDR_SIZE]

        iph = unpack(IP_HDR_binary_string_format, ip_header)

        s_addr = socket.inet_ntoa(iph[8])
        d_addr = socket.inet_ntoa(iph[9])


        timestamp = pkthdr.ts.tv_sec + pkthdr.ts.tv_usec / 1000000

        pkt_info = {
            'timestamp': timestamp,
            'src': s_addr,
            'dst': d_addr,
            'len': pkthdr.len
        }
        self.channel.queue_declare(queue='testinho', durable=True)
        self.channel.basic_publish(exchange='',
                              routing_key='testinho',
                              body=json.dumps(pkt_info),
                              properties=pika.BasicProperties(
                                  delivery_mode=2,  # make message persistent
                              ))

LiveCap().start()