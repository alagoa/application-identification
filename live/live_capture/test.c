#include <stdio.h>
#include <time.h>
#include <pcap.h>
#include <netinet/in.h>
#include <netinet/if_ether.h>
#include "faster.h"

struct ipv4_filter_list filters_ipv4 = {.valid_filters=0, .filter=NULL};

void print_packet_info(const u_char *packet, struct pcap_pkthdr packet_header) {
    printf("Packet capture length: %d\n", packet_header.caplen);
    printf("Packet total length %d\n", packet_header.len);
}

int main(int argc, char *argv[]) {

	pcap_t *handle;			/* Session handle */
 	char *dev;
    char errbuf[PCAP_ERRBUF_SIZE];
    struct pcap_pkthdr packet_header;
    pcap_t* descr;
    bpf_u_int32 maskp;            /* subnet mask */
    bpf_u_int32 netp;             /* ip */
	const u_char *packet;		/* The actual packet */
 
    /* Now get a device */
    dev = pcap_lookupdev(errbuf);
     
    if(dev == NULL) {
        fprintf(stderr, "%s\n", errbuf);
        exit(1);
    }
        /* Get the network address and mask */
    pcap_lookupnet(dev, &netp, &maskp, errbuf);
 
    /* open device for reading in promiscuous mode */
    descr = pcap_open_live(dev, BUFSIZ, 1,-1, errbuf);
    if(descr == NULL) {
        printf("pcap_open_live(): %s\n", errbuf);
        exit(1);
    }

    /* add filters */
    read_ipv4_filter("ipv4_filter.conf",&filters_ipv4);

    uint valid_filters;
    char ipv4_prefix[INET_ADDRSTRLEN],ipv4_mask[INET_ADDRSTRLEN];
    valid_filters=filters_ipv4.valid_filters;

    if(valid_filters>0){
        printf("Local Network %d\n",valid_filters);
        for(uint j=0;j<valid_filters;j++)
        {
            ipv4_str_addr(filters_ipv4.filter[j].net_prefix,ipv4_prefix);
            ipv4_str_addr(filters_ipv4.filter[j].net_mask,ipv4_mask);
            printf("\t-> %s %s\n",ipv4_prefix,ipv4_mask);       
        }
    }
    else
        printf("No IPv4 filters\n");
    printf("####\n");

     /* Attempt to capture one packet. If there is no network traffic
      and the timeout is reached, it will return NULL */
     packet = pcap_next(handle, &packet_header);
     if (packet == NULL) {
        printf("No packet found.\n");
        return 2;
    }

    /* Our function to output some info */
    print_packet_info(packet, packet_header);

    return 0;
}
