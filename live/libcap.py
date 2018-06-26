import ctypes
from struct import *
import sys
import socket


PCAP_NETMASK_UNKNOWN = 0xffffffff

ETHERNET_HDR_SIZE = 14
IP_HDR_binary_string_format = '!BBHHHBBH4s4s'


libpcap_filename = "libpcap.so"
_libpcap_lib = ctypes.cdll.LoadLibrary(libpcap_filename)

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


class timeval(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long),
                ('tv_usec', ctypes.c_long)]


class pcap_pkthdr(ctypes.Structure):
    _fields_ = [('ts', timeval),
                ('caplen', ctypes.c_uint),
                ('len', ctypes.c_uint)]


class bpf_insn(ctypes.Structure):
    _fields_ = [('code', ctypes.c_ushort),
                ('jt', ctypes.c_ubyte),
                ('jf', ctypes.c_ubyte),
                ('k', ctypes.c_ulong)]


class bpf_program(ctypes.Structure):
    _fields_ = [('bf_len', ctypes.c_int),
                ('bpf_insn', ctypes.POINTER(bpf_insn))]


class ipv4_header(ctypes.Structure):
    _fields_ = [('ip_vhl', ctypes.c_uint8),
                ('ip_tos', ctypes.c_uint8),
                ('ip_len', ctypes.c_uint16),
                ('ip_id', ctypes.c_uint16),
                ('ip_off', ctypes.c_uint16),
                ('ip_ttl', ctypes.c_uint8),
                ('ip_p', ctypes.c_uint8),
                ('ip_sum', ctypes.c_uint16),
                ('ip_src', ctypes.c_uint32),
                ('ip_dst', ctypes.c_uint32)]


def pcap_lookupdev():

    errbuf = ctypes.create_string_buffer(256)
    pcap_lookupdev = _libpcap_lib.pcap_lookupdev
    pcap_lookupdev.restype = ctypes.c_char_p
    return pcap_lookupdev(errbuf)


def pcap_findalldevs():

    pcap_findalldevs = _libpcap_lib.pcap_findalldevs
    pcap_findalldevs.restype = ctypes.c_int
    pcap_findalldevs.argtypes = [ctypes.POINTER(ctypes.POINTER(pcap_if)),
                                 ctypes.c_char_p]
    errbuf = ctypes.create_string_buffer(256)
    alldevs = ctypes.POINTER(pcap_if)()
    result = pcap_findalldevs(ctypes.byref(alldevs), errbuf)
    if result == 0:
        devices = []
        device = alldevs.contents
        while (device):
            devices.append(device.name)
            if device.next:
                device = device.next.contents
            else:
                device = False
        # free to avoid leaking every time we call findalldevs
        pcap_freealldevs(alldevs)
    else:
        raise Exception(errbuf)
    return devices


def pcap_freealldevs(alldevs):
    pcap_freealldevs = _libpcap_lib.pcap_freealldevs
    pcap_freealldevs.restype = None
    pcap_freealldevs.argtypes = [ctypes.POINTER(pcap_if)]
    pcap_freealldevs(alldevs)


def pcap_open_live(device, snaplen, promisc, to_ms):

    pcap_open_live = _libpcap_lib.pcap_open_live
    pcap_open_live.restype = ctypes.POINTER(ctypes.c_void_p)
    pcap_open_live.argtypes = [ctypes.c_char_p,
                               ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_char_p]
    errbuf = ctypes.create_string_buffer(256)
    handle = pcap_open_live(device, snaplen, promisc, to_ms, errbuf)
    if not handle:
        print("Error opening device %s." % device)
        return None
    return handle


def pcap_next(handle):
    pcap_next = _libpcap_lib.pcap_next
    pcap_next.restype = ctypes.POINTER(ctypes.c_char)
    pcap_next.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                          ctypes.POINTER(pcap_pkthdr)]
    pkthdr = pcap_pkthdr()
    pktdata = pcap_next(handle, ctypes.byref(pkthdr))
    return pkthdr, pktdata[:pkthdr.len]


class pcap_pkthdr(ctypes.Structure):
    _fields_ = [
        ('ts', timeval),
        ('caplen', ctypes.c_uint32),
        ('len', ctypes.c_uint32)
    ]


def __callback_wrapper(user_data, pkthdr_p, data):
    (callback, obj) = user_data.contents.value
    if callback:
        callback(pkthdr_p.contents, ctypes.string_at(data, pkthdr_p.contents.caplen), obj)


def pcap_loop(hpcap, cnt, callback, user_data):
    pcap_pkthdr_p = ctypes.POINTER(pcap_pkthdr)
    pcap_handler = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.py_object), pcap_pkthdr_p,
                                    ctypes.POINTER(ctypes.c_ubyte))

    retcode = _libpcap_lib.pcap_loop(hpcap, cnt, pcap_handler(__callback_wrapper),
                                     ctypes.pointer(ctypes.py_object((callback, user_data))))
    if retcode == -1:
        exit(1)
    return retcode


def pcap_setfilter(handle, bpf):
    pcap_setfilter = _libpcap_lib.pcap_setfilter
    pcap_setfilter.restype = ctypes.c_int
    pcap_setfilter.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                               ctypes.POINTER(bpf_program)]
    result = pcap_setfilter(handle, bpf)
    return result


def pcap_close(handle):
    pcap_close = _libpcap_lib.pcap_close
    pcap_close.restype = None
    pcap_close.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    pcap_close(handle)


class Packet(object):
    def __init__(self, header, data):
        self._length = header.len
        self._data = data

    def getData(self):
        return self._data

    def getLength(self):
        return self._length