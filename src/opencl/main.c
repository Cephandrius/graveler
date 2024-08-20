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

const char *program_text[] = {
"#include <tyche_i.cl>\n\
kernel void simulate(ulong num_sims, global ulong* seed, global ushort* res){\n\
	uint gid = get_global_id(0);\n\
	uint current_successes;\n\
	uint most_successes = 0;\n\
	tyche_i_state state;\n\
	tyche_i_seed(&state, seed[gid]);\n\
	for(ulong i = 0; i < num_sims; i++){\n\
		current_successes = 0;\n\
		for(int j = 0; j < 231; j++){\n\
			current_successes += tyche_i_uint(state) % 4 == 0;\n\
		}\n\
		if(most_successes < current_successes){\n\
			most_successes = current_successes;\n\
		}\n\
	}\n\
	res[gid] = most_successes;\n\
}\n"
};

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

int main(int argc, char ** argv){
	size_t param_max_size;
	void* param;
	size_t param_size;
	cl_int CL_err;
	cl_platform_id platform;
	cl_device_id device; 
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_mem seeds;
	cl_mem res;
	cl_command_queue queue;
	cl_event read_res;
	cl_event execute_kernel;
	long num_simulations;
	size_t num_seeds;
	size_t sims_per_seed;
	size_t work_group_size;
	uint64_t *host_seeds;
	uint16_t *host_res;
	int most_successes = 0;
	long queued_simulations = 0;
	bool my_computer;
	int num_simulations_arg;
	int my_computer_arg;
#ifdef __linux__
	struct timespec start;
	struct timespec end;
  	clock_gettime(CLOCK_MONOTONIC, &start);
#endif
	if(argc == 3){
		num_simulations_arg = 2;
		my_computer_arg = 1;
	}else if(argc == 2){
		num_simulations_arg = 1;
		my_computer_arg = -1;
	}else{
		print_help(argv[0]);
		return -1;
	}
	if(my_computer_arg == 1){
		if(strcmp(argv[my_computer_arg], "--my-computer")== 0){
			my_computer = true;
		}else{
			print_help(argv[0]);
			return -1;
		}
	}else{
		my_computer = false;
	}

	num_simulations = strtol(argv[num_simulations_arg], NULL, 10);	
	if(num_simulations == 0){
		print_help(argv[0]);
		return -1;
	}

	CL_err = get_platform(&platform, my_computer);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}

	CL_err = get_device(platform, &device, my_computer);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}

	CL_err = initialize_kernel(device, &context, &program, &kernel);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}
	printf("Kernel initalized.\n");

	CL_err = calc_res_size(device, &work_group_size, &num_seeds);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}	

	CL_err = make_buffers(context, num_seeds, &seeds, &res);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}
	printf("Made buffers.\n");
	host_seeds = malloc(sizeof(uint64_t) * num_seeds);
	host_res = malloc(sizeof(uint16_t) * num_seeds);

	cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	queue = clCreateCommandQueueWithProperties(context, device,properties , &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateCommandQueue", CL_err, __LINE__);
		return CL_err;
	}
	printf("Made command queue.\n");
	
	sims_per_seed = 1000;
	CL_err = set_kernel_args(kernel, (uint64_t) sims_per_seed, seeds, res);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}

	srand(time(NULL));

	printf("Starting Computation.\n");	
	param_max_size = 1024;
	param = malloc(param_max_size);
	CL_err = do_one_iteration(queue, kernel, seeds, res, host_seeds, host_res, num_seeds, &sims_per_seed, work_group_size, &read_res, &execute_kernel, &most_successes, true, false, param_max_size, param); 
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}
	for(queued_simulations = num_seeds * sims_per_seed; queued_simulations < num_simulations; queued_simulations += num_seeds * sims_per_seed){
		if(queued_simulations >= num_simulations){
			CL_err = do_one_iteration(queue, kernel, seeds, res, host_seeds, host_res, num_seeds, &sims_per_seed, work_group_size, &read_res, &execute_kernel, &most_successes, false, true, param_max_size, param); 
			if(CL_err != CL_SUCCESS){
				return CL_err;
			}
		}else{
			CL_err = do_one_iteration(queue, kernel, seeds, res, host_seeds, host_res, num_seeds, &sims_per_seed, work_group_size, &read_res, &execute_kernel,  &most_successes, false, false, param_max_size, param); 
			if(CL_err != CL_SUCCESS){
				return CL_err;
			}
			printf("Progress: %li simulations\n", queued_simulations);
		}
	}

	printf("Num Simulations:%li\nMost Successes:%i\n", queued_simulations, most_successes);
	free(host_seeds);
	free(host_res);
	free(param);
#ifdef __linux__
	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Time Taken:%f seconds\n", (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0);
#endif
	return 0;
}

cl_int do_one_iteration(cl_command_queue queue, cl_kernel kernel, cl_mem seeds, cl_mem res, uint64_t *host_seeds, uint16_t *host_res, size_t num_seeds, size_t *sims_per_seed, size_t work_group_size, cl_event *read_res, cl_event *execute_kernel, int *most_successes, bool first_execute, bool last_execute, size_t param_max_size, void* param){
	cl_int CL_err;
	cl_event write_seeds;
	cl_ulong start;
	cl_ulong end;
	if(!first_execute){
		CL_err = clWaitForEvents(1, read_res);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clWaitForEvents", CL_err, __LINE__);
			return CL_err;
		}
		CL_err = clGetEventProfilingInfo(*execute_kernel, CL_PROFILING_COMMAND_START, param_max_size, param, NULL);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clGetEventProfilingInfo", CL_err, __LINE__);
			return CL_err;
		}
		start = *((cl_ulong *) param);
		CL_err = clGetEventProfilingInfo(*execute_kernel, CL_PROFILING_COMMAND_END, param_max_size, param, NULL);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clGetEventProfilingInfo", CL_err, __LINE__);
			return CL_err;
		}
		end = *((cl_ulong *) param);
		*sims_per_seed = *sims_per_seed * 1000000000.0 / (end - start);
		CL_err = set_kernel_num_sims(kernel, *sims_per_seed);
		if(CL_err != CL_SUCCESS){
			return CL_err;
		}
	}
	if(!last_execute){
		create_seeds(host_seeds, num_seeds);
		CL_err = clEnqueueWriteBuffer(queue, seeds, CL_FALSE, 0, sizeof(uint64_t) * num_seeds, host_seeds, 0, NULL, &write_seeds);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clEnqueueWriteBuffer", CL_err, __LINE__);
			return CL_err;
		}
		CL_err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &num_seeds, &work_group_size, 1, &write_seeds, execute_kernel);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clEnqueueNDRangeKernel", CL_err, __LINE__);
			return CL_err;
		}
	}

	if(!first_execute){
		for(size_t i = 0; i < num_seeds; i++){
			if(*most_successes < host_res[i]){
				*most_successes = host_res[i];
			}
		}
	}

	if(!last_execute){
		CL_err = clEnqueueReadBuffer(queue, res, CL_FALSE, 0, sizeof(uint16_t) * num_seeds, host_res, 1, execute_kernel, read_res);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clEnqueueReadBuffer", CL_err, __LINE__);
			return CL_err;
		}
	}
	return CL_SUCCESS;
}

cl_int set_kernel_num_sims(cl_kernel kernel, uint64_t num_sims){
	cl_int CL_err = clSetKernelArg(kernel, 0, sizeof(uint64_t), &num_sims);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clSetKernelArg", CL_err, __LINE__);
		return CL_err;
	}
	return CL_SUCCESS;
}

cl_int set_kernel_args(cl_kernel kernel, uint64_t num_sims, cl_mem seeds, cl_mem res){
	cl_int CL_err = set_kernel_num_sims(kernel, num_sims);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}
	CL_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &seeds);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clSetKernelArg", CL_err, __LINE__);
		return CL_err;
	}
	CL_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &res);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clSetKernelArg", CL_err, __LINE__);
		return CL_err;
	}
	return CL_SUCCESS;
}

cl_int make_buffers(cl_context context, size_t num_seeds, cl_mem *seeds, cl_mem *res){
	cl_int CL_err;
	*seeds = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint64_t) * num_seeds, NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateBuffer", CL_err, __LINE__);
		return CL_err;
	}
	*res = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint16_t) * num_seeds , NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateBuffer", CL_err, __LINE__);
		return CL_err;
	}
	return CL_SUCCESS;
}

cl_int calc_res_size(cl_device_id device, size_t *work_group_size, size_t *num_seeds){
	cl_int CL_err;
	size_t max_work_group_size;
	cl_uint num_compute_units;

	CL_err = get_device_info(device, &max_work_group_size, &num_compute_units);
	if(CL_err != CL_SUCCESS){
		return CL_err;
	}

	*num_seeds = num_compute_units * max_work_group_size;
	*work_group_size = max_work_group_size;
	return CL_SUCCESS;
}

cl_int get_device_info(cl_device_id device, size_t *max_work_group_size, cl_uint *num_compute_units){
	cl_int CL_err;
	size_t param_max_size;
	void* param;
	size_t param_size;
	char major_version;

	param_max_size = 128;
	param = malloc(param_max_size);

	CL_err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, param_max_size, param, &param_size);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clGetDeviceInfo", CL_err, __LINE__);
		free(param);
		return CL_err;
	}
	*max_work_group_size = *(size_t*) param;

	CL_err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, param_max_size, param, &param_size);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clGetDeviceInfo", CL_err, __LINE__);
		free(param);
		return CL_err;
	}
	*num_compute_units = *(cl_uint*) param;
	free(param);
	return CL_SUCCESS;
}

cl_int initialize_kernel(cl_device_id device, cl_context *context, cl_program *program, cl_kernel *kernel){
	cl_int CL_err;
	size_t param_max_size;
	void* param;
	size_t param_size;
	*context = clCreateContext(NULL, 1, &device, NULL, NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateContext", CL_err, __LINE__);
		return CL_err;
	}
	*program = clCreateProgramWithSource(*context, 1, program_text, NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateProgramWithSource", CL_err, __LINE__);
		return CL_err;
	}	

	CL_err = clBuildProgram(*program, 1, &device, COMPILER_OPTS, NULL, NULL);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clBuildProgram", CL_err, __LINE__);
		param_max_size = 2048;
		param = malloc(param_max_size);
		clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, param_max_size, param, &param_size);
		printf("%s\n", (char *) param);
		free(param);
		return CL_err;	
	}

	*kernel = clCreateKernel(*program, KERNEL_NAME, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateKernel", CL_err, __LINE__);
		return CL_err;
	}
	return CL_SUCCESS;
}

void create_seeds(uint64_t* host_seeds, size_t num_seeds){
	for(int i = 0; i < num_seeds; i++){
		host_seeds[i] = rand();
	}
}

cl_int get_device(cl_platform_id platform, cl_device_id* device, bool is_my_computer){
	cl_uint CL_err;
	size_t param_max_size;
	void* param;
	size_t param_size;
	bool found_my_device = false;
	size_t devices_max_size = 5;
	cl_device_id* devices = malloc(sizeof(cl_device_id) * devices_max_size);
	cl_int num_devices;

	CL_err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices_max_size, devices, &num_devices);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clGetDeviceIDs", CL_err, __LINE__);
		return CL_err;
	}


	if(is_my_computer){
		param_max_size = 1024;
		param = malloc(param_max_size);
	
		for(int i = 0; i < num_devices; i++){
			CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, param_max_size, param, &param_size);
			if(CL_err != CL_SUCCESS){
				print_cl_error("clGetDeviceInfo", CL_err, __LINE__);
				return CL_err;
			}
			if(strcmp("gfx1031", (char *) param) == 0){
				found_my_device = true;
				*device = devices[i];
			}
		}
		free(param);
		if(!found_my_device){
			printf("Naughty boy! Don't use --my-computer unless you are using the creators computer.\n");
			free(devices);
			return -1;
		}
	}else{
		*device = devices[FIRST_DEVICE];
	}
	param_max_size = 1024;
	param = malloc(param_max_size);
	CL_err = clGetDeviceInfo(*device, CL_DEVICE_NAME, param_max_size, param, &param_size);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clGetDeviceInfo", CL_err, __LINE__);
		return CL_err;
	}
	printf("Using %s\n", (char *) param);
	free(param);
	free(devices);
	return CL_SUCCESS;
}

cl_int get_platform(cl_platform_id* platform, bool is_my_computer){
	cl_int CL_err;
	size_t param_max_size;
	void* param;
	size_t param_size;
	bool found_my_platform = false;
	cl_uint num_platforms = 0;
	size_t platforms_max_size = 5;
	cl_platform_id* platforms = malloc(sizeof(cl_platform_id) * platforms_max_size);

	CL_err = clGetPlatformIDs(platforms_max_size, platforms, &num_platforms);
	if(CL_err != CL_SUCCESS){
		if(CL_err == CL_PLATFORM_NOT_FOUND_KHR){
			printf("No platforms are installed.\n");
			return CL_err;
		}else{
			print_cl_error("clGetPlatformIDs", CL_err, __LINE__);
			return CL_err;
		}
	}

	if(is_my_computer){
		param_max_size = 1024;
		param = malloc(param_max_size);
		for(int i = 0; i < num_platforms; i++){
			CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_max_size, param, &param_size);
			if(CL_err != CL_SUCCESS){
				print_cl_error("clGetPlatformInfo", CL_err, __LINE__);
				return CL_err;
			}
			if(strcmp("AMD Accelerated Parallel Processing", (char *) param) == 0){
				found_my_platform = true;
				*platform = platforms[i];
			}
		}
		free(param);
		if(!found_my_platform){
			printf("Naughty boy! Don't use --my-computer unless you are using the creators computer.\n");
			free(platforms);
			return -1;
		}
	}else{
		*platform = platforms[FIRST_PLATFORM];	
	}
	param_max_size = 1024;
	param = malloc(param_max_size);
	CL_err = clGetPlatformInfo(*platform, CL_PLATFORM_NAME, param_max_size, param, &param_size);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clGetPlatformInfo", CL_err, __LINE__);
		return CL_err;
	}
	printf("Using %s\n", (char *) param);
	free(param);
	free(platforms);
	return CL_SUCCESS;
}

void print_cl_error(char* function_name,  cl_int err, int line_num){
	printf("%s:%i %s(%i)\n", __FILE__, line_num, function_name, err);
}

void print_help(char* program_name){
	printf("%s [--my-computer] num_simulations\n", program_name);
}
