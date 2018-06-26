struct ipv4_filter {
	uint32_t net_prefix;
	uint32_t net_mask;              
} __attribute__((__packed__));

struct ipv4_filter_list {
	uint16_t valid_filters;
	struct ipv4_filter *filter;
} __attribute__((__packed__));

static inline void gen_ipv4_mask(uint8_t msb, uint32_t *mask) {
	if(msb>32) msb=32;
	*mask=0;
	*mask=(-1) << (32 - msb);
	
	*mask=ntohl(*mask);
	return;
}

static inline void addr_ipv4_clean(uint32_t *addr, const uint32_t mask){
	*addr&=mask;
}

static inline void read_ipv4_filter(const char *filename, struct ipv4_filter_list *filters)
{
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char addr[INET_ADDRSTRLEN];
    int msb, pf;
    uint nline=0, valid_filters=0;
    uint32_t ipv4mask, ipv4prefix;


	filters->valid_filters=0;
    fp = fopen(filename, "r");
    if (fp == NULL){
		printf("File [for IPv4 filtering] not found: %s\n",filename);
		exit(-1);
    }
	else{
		printf("Reading file [for IPv4 filtering]: %s\n",filename);
		while ((read = getline(&line, &len, fp)) != -1) {
			msb=-1;
			sscanf(line,"%[^/]/%d",addr,&msb);
			if(msb>=0 || msb<=32){
				pf = ntohl(inet_pton(AF_INET, addr, &ipv4prefix));
				if (pf > 0) valid_filters++;
			}
		}
		
		filters->valid_filters=valid_filters;
		if(valid_filters>0){
			filters->filter=malloc(valid_filters*sizeof(struct ipv4_filter));
			
			valid_filters=0;
			fseek(fp,0,SEEK_SET);		
			while ((read = getline(&line, &len, fp)) != -1) {
				nline++;
				msb=-1;
				sscanf(line,"%[^/]/%d",addr,&msb);
				if(msb<0 || msb>32)
					printf("\tError in line %d of file %s\n",nline,filename);
				else{
					pf = inet_pton(AF_INET, addr, &ipv4prefix);
					
					if (pf <= 0) {
						if (pf == 0)
							fprintf(stderr, "\tError in line %d of file %s\n",nline,filename);
					}
					else{			
						gen_ipv4_mask(msb,&ipv4mask);
						printf("\tAdding network %s/%d to IPv4 filter\n",addr,msb);
						addr_ipv4_clean(&ipv4prefix,ipv4mask);
						filters->filter[valid_filters].net_prefix=ipv4prefix;
						filters->filter[valid_filters].net_mask=ipv4mask;
						valid_filters++;
					}
				}
			}
		}
		
		printf("Done\n");

		fclose(fp);
		if (line)
			free(line);
			
   }
   return;
}

static inline uint8_t addr_ipv4_filter(const uint32_t saddr, const uint32_t daddr, const struct ipv4_filter_list *filters){
	//0: do not accept, 1: upload, 2: download, 3: both (saddr and daddr in filter)
	uint8_t sfound=0, dfound=0;
	uint16_t i;
	
	i=0;
	while(sfound==0 && i<filters->valid_filters){
		if((saddr&filters->filter[i].net_mask)==filters->filter[i].net_prefix)
			sfound=1;
		i++;
	}
	
	i=0;
	while(dfound==0 && i<filters->valid_filters){
		if((daddr&filters->filter[i].net_mask)==filters->filter[i].net_prefix)
			dfound=2;
		i++;
	}
			
	return sfound+dfound;
}


