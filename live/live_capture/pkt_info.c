typedef struct pkt_info {
	double timestamp;
	uint direction; // 0 - upload, 1 - download, 2 - both
	uint len;
	uint16_t port;
}