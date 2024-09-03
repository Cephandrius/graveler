#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>

#define EVEN_BITS 0xaaaaaaaa
#define FIRST_SIM 0x1
#define SIMS_PER_INT 15

int simulate();
bool roll();
int make_more_sims();
