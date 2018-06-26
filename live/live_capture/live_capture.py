from libcap import *
import pika


class LiveCap():
    def __init__(self):

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()

    def start(self):
        print("Starting live capture...")
        devices = pcap_findalldevs()
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
            pcap_loop(handle, -1, self.redirect_packets(), None)
        print("Closing live capture")
        pcap_close(handle)

    def redirect_packets(self, pkt):
        self.channel.queue_declare(queue='testinho', durable=True)
        self.channel.basic_publish(exchange='',
                                   routing_key='testinho',
                                   body=pickle.dumps(pkt),
                                   properties=pika.BasicProperties(
                                       delivery_mode=2,  # make message persistent
                                   ))

def callback(pkthdr, data, user_data):
    ip_header = data[ETHERNET_HDR_SIZE:20 + ETHERNET_HDR_SIZE]

    iph = unpack(IP_HDR_binary_string_format, ip_header)

    s_addr = socket.inet_ntoa(iph[8])
    d_addr = socket.inet_ntoa(iph[9])

    timestamp = pkthdr.ts.tv_sec + pkthdr.ts.tv_usec / 1000000;

    print(pkthdr.len)
    print(timestamp)
    print(s_addr + " -> " + d_addr)
    print("-----------")


if __name__ == "__main__":
    print("Testcases for libpcap module")

    # Test pcap_lookupdev function
    device = pcap_lookupdev()
    print("pcap_lookupdev() -> %r" % device)

    # Test pcap_findalldevs function
    devices = pcap_findalldevs()
    print("pcap_findalldevs() -> %r" % devices)

    # Test pcap_open_live function
    device = devices[0]
    snaplen = 65535
    promisc = 1
    to_ms = 10000
    print("pcap_open_live(%s, %s, %s, %s)" % (device, snaplen, promisc, to_ms))
    handle = pcap_open_live(device, snaplen, promisc, to_ms)
    print("pcap_open_live(...) -> %r" % handle)
    if handle:
        bpf = bpf_program()

        result = pcap_setfilter(handle, bpf)
        print("pcap_setfilter -> %r" % result)

        pcap_loop(handle, -1, callback, None)

    # Test pcap_close function
    pcap_close(handle)
