import pika
from pika.exceptions import IncompatibleProtocolError
import json
from libpcapy import capture, types
from socket import inet_ntoa
from struct import unpack
from time import sleep

class LiveCap():

    def __init__(self):

        self.connection = None
        self.channel = None

    def start(self):

        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
            self.channel = self.connection.channel()
        except IncompatibleProtocolError:
            print("No response, trying again in 5 seconds...")
            sleep(5)
            self.start()

        print("Starting live capture...")
        devices = capture.pcap_findalldevs()
        device = devices[0]
        snaplen = 65535
        promisc = 1
        to_ms = 10000
        print("Capturing on %s" % device)
        handle = capture.pcap_open_live(device, snaplen, promisc, to_ms)
        if handle:
            bpf = capture.bpf_program()
            result = capture.pcap_setfilter(handle, bpf)
            print("pcap_setfilter -> %r" % result)
            capture.pcap_loop(handle, -1, self.my_callback, None)
        print("Closing live capture")
        capture.pcap_close(handle)

    def my_callback(self, pkthdr, data, user_data):
        ip_header = data[types.ETHERNET_HDR_SIZE:20 + types.ETHERNET_HDR_SIZE]

        iph = unpack(types.IP_HDR_binary_string_format, ip_header)

        s_addr = inet_ntoa(iph[8])
        d_addr = inet_ntoa(iph[9])

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
