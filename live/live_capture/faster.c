#include <pcap.h>
#include <stdio.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/if_ether.h>
#include "faster.h"

struct ipv4_filter_list filters_ipv4 = {.valid_filters=0, .filter=NULL};

typedef struct pkt_info {
    double timestamp;
    uint dir; // 0 - upload, 1 - download, 2 - both, 3 - none
    uint len;
    uint16_t port;
} pkt_info;

void my_callback(u_char *args, const struct pcap_pkthdr* pkthdr, const u_char*
    packet)
{

    pkt_info *packet_info = malloc(sizeof(pkt_info));

    double timestamp;
    timestamp=pkthdr->ts.tv_sec+(double)pkthdr->ts.tv_usec/1000000;


    struct ipv4_header *ipv4ptr; 
    ipv4ptr = (struct ipv4_header *) (packet + ETHER_HDR_LEN); 

    char ipv4_srcaddr[INET_ADDRSTRLEN],ipv4_dstaddr[INET_ADDRSTRLEN];        
    ipv4_str_addr(ipv4ptr->ip_src,ipv4_srcaddr);
    ipv4_str_addr(ipv4ptr->ip_dst,ipv4_dstaddr);

    uint filt_res=addr_ipv4_filter(ipv4ptr->ip_src, ipv4ptr->ip_dst, &filters_ipv4);
    
    packet_info->timestamp = timestamp;

    if(filt_res==1)
        packet_info->dir = 0;
    else if(filt_res==2)
        packet_info->dir = 1;
    else if(filt_res==3)
        packet_info->dir = 2;
    else
        packet_info->dir = 3;

    packet_info->len=ntohs(ipv4ptr->ip_len);

    printf("\n");
    printf("%f\n", packet_info->timestamp);
    printf("%d\n", packet_info->dir);
    printf("%d\n", packet_info->len);
    printf("----\n---");
}

void start()
{
    char *dev;
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* descr;
    bpf_u_int32 maskp;            /* subnet mask */
    bpf_u_int32 netp;             /* ip */
 
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

    /* loop for callback function */
    pcap_loop(descr, -1, my_callback, NULL);
}

int main(int argc,char **argv) {
    start();
    return 0;
}