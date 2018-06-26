#include <pcap.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/if_ether.h>
#include "filtering.h"

struct ipv4_header
{
	u_int8_t	ip_vhl;		/* header length, version */
	u_int8_t	ip_tos;		/* type of service */
	u_int16_t	ip_len;		/* total length */
	u_int16_t	ip_id;		/* identification */
	u_int16_t	ip_off;		/* fragment offset field */
	u_int8_t	ip_ttl;		/* time to live */
	u_int8_t	ip_p;		/* protocol */
	u_int16_t	ip_sum;		/* checksum */
	u_int32_t	ip_src,ip_dst;	/* source and dest address */
};

struct tcp_header {
	uint16_t src_port;  
	uint16_t dst_port;  
	uint32_t sent_seq;  
	uint32_t recv_ack;  
	uint8_t  data_off;  
	uint8_t  tcp_flags; 
	uint16_t rx_win;    
	uint16_t cksum;     
	uint16_t tcp_urp;   
};

static inline void ipv4_str_addr(const u_int32_t in_addr,char * addr_str)
{
	u_int32_t addr=ntohl(in_addr);
	sprintf(addr_str,"%d.%d.%d.%d",addr>>24,(addr>>16)&0xFF,(addr>>8)&0xFF,addr&0xFF);
	return;
}

static inline void ipv6_str_addr(const uint8_t *addr,char * addr_str)
{
	if (inet_ntop(AF_INET6, addr, addr_str, INET6_ADDRSTRLEN) == NULL) {
		perror("inet_ntop");
		exit(EXIT_FAILURE);
	}
	return;
}
