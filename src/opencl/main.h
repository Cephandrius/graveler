#define _XOPEN_SOURCE 700
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

cl_int initialize_kernel(cl_device_id device, cl_context *context, cl_program *program, cl_kernel *kernel);
cl_int make_buffers(cl_context context, size_t num_seeds, cl_mem *seeds, cl_mem *res);
cl_int do_one_iteration(cl_command_queue queue, cl_kernel kernel, cl_mem seeds, cl_mem res, uint64_t *host_seeds, uint16_t *host_res, size_t num_seeds, size_t *sims_per_seed, size_t work_group_size, cl_event *read_res, cl_event *execute_kernel, int *most_successes, bool first_execute, bool last_execute, size_t param_max_size, void* param);
cl_int set_kernel_num_sims(cl_kernel kernel, uint64_t num_sims);
cl_int set_kernel_args(cl_kernel kernel, uint64_t num_sims, cl_mem seeds, cl_mem res);
cl_int calc_res_size(cl_device_id device, size_t *work_group_size, size_t *num_seeds);
cl_int get_device_info(cl_device_id device, size_t *max_work_group_size, cl_uint *num_compute_units);
cl_int get_platform(cl_platform_id* platform, bool is_my_computer);
void print_cl_error(char* function_name, cl_int err, int line_num);
cl_int get_device(cl_platform_id platform, cl_device_id* device, bool is_my_computer);
void create_seeds(uint64_t* host_seeds, size_t num_seeds);
void print_help(char* program_name);

#ifdef _WIN32
#define COMPILER_OPTS "-I ..\\generators\\"
#else
#define COMPILER_OPTS "-I ../generators/"
#endif

#define KERNEL_NAME "simulate"
#define CL_PLATFORM_NOT_FOUND_KHR -1001
#define FIRST_PLATFORM 0
#define FIRST_DEVICE 0
#define MAX_MEMORY_USAGE (long) 4 * 1024 * 1024 * 1024

#define SIMS_TOO_BIG -1
#define TOO_MUCH_MEMORY -2
#define OK 1
#define TOO_WEIRD -12345
