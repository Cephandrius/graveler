#include "graveler_c.h"

#include <time.h>

int main(int argc, char ** argv){
	srand(time(NULL));
	int simulations = 1000000000;
	int max_successes = 0;
	int num_successes = 0;
#ifdef __linux__
	struct timespec start;
	struct timespec end;
  	clock_gettime(CLOCK_MONOTONIC, &start);
#endif
	for(int i = 0; i < simulations; i++){
		num_successes = simulate();
		if(num_successes > max_successes){
			max_successes = num_successes;
		}
	}
	printf("Max Successes:%i\n", max_successes);
#ifdef __linux__
	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Time Taken:%f seconds\n", (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0);
#endif
}

int simulate(){
	int num_successes = 0;
	for(int i = 0; i < 231; i++){
		if(roll()){
			num_successes++;
		}
	}
	return num_successes;
}

bool roll(){
	static int num_sims_left = 0;
	static int sims = 0;
	bool success = false;
	if(num_sims_left == 0){
		sims = make_more_sims();
		num_sims_left = SIMS_PER_INT;
	}
	num_sims_left -= 1;
	success = sims & FIRST_SIM == 1;
	sims >>= 2;
	return success;
}

int make_more_sims(){
	int sims = rand();
	int even_bits = sims & EVEN_BITS;
	int odd_bits = sims & ~ EVEN_BITS;
	even_bits >>= 1;
	sims = even_bits & odd_bits;
	return sims;
}
