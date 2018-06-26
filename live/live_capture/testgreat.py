import ctypes, sys

class timeval(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long),
                ('tv_usec', ctypes.c_long)]

class pcap_pkthdr(ctypes.Structure):
    _fields_ = [('ts', timeval),
                ('caplen', ctypes.c_uint),
                ('len', ctypes.c_uint)]

class sockaddr(ctypes.Structure):
    _fields_ = [("sa_family", ctypes.c_ushort),
                ("sa_data", ctypes.c_char * 14)]

class pcap_addr(ctypes.Structure):
    pass

pcap_addr._fields_ = [('next', ctypes.POINTER(pcap_addr)),
                      ('addr', ctypes.POINTER(sockaddr)),
                      ('netmask', ctypes.POINTER(sockaddr)),
                      ('broadaddr', ctypes.POINTER(sockaddr)),
                      ('dstaddr', ctypes.POINTER(sockaddr))]

class pcap_if(ctypes.Structure):
    pass

pcap_if._fields_ = [('next', ctypes.POINTER(pcap_if)),
                    ('name', ctypes.c_char_p),
                    ('description', ctypes.c_char_p),
                    ('addresses', ctypes.POINTER(pcap_addr)),
                    ('flags', ctypes.c_uint)]

def pcap_next(handle):
    '''Return the next available packet in a dictionary.'''
    # u_char* pcap_next(pcap_t* p, struct pcap_pkthdr* h)
    pcap_next = _libpcap_lib.pcap_next
    pcap_next.restype = ctypes.POINTER(ctypes.c_char)
    pcap_next.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                          ctypes.POINTER(pcap_pkthdr)]
    pkthdr = pcap_pkthdr()
    pktdata = pcap_next(handle, ctypes.byref(pkthdr))
    return pkthdr, pktdata[:pkthdr.len]

_pcap = ctypes.cdll.LoadLibrary("libpcap.so")

errbuf = ctypes.create_string_buffer(256)

pcap_lookupdev = _pcap.pcap_lookupdev
pcap_lookupdev.restype = ctypes.c_char_p
dev = pcap_lookupdev(errbuf)
print(dev)

# create handler
pcap_create = _pcap.pcap_create
handle = pcap_create(dev, errbuf)
print(handle)
if not handle:
    print("failed creating handler:",errbuf)
    exit()

_pcap.pcap_next(handle, )