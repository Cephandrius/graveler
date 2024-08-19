#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

void print_cl_error(char* function_name, cl_int err, int line_num);
void get_platforms(cl_platform_id* platforms, cl_uint num_platforms, cl_platform_id* amd_platform, cl_platform_id* pocl_platform);
void get_devices(cl_platform_id amd_platform, cl_platform_id pocl_platform, cl_device_id* gpu, cl_device_id* cpu);
void create_seeds(uint64_t* host_seeds);

const char *program_text[] = {
"#include <tyche_i.cl>\n",
"kernel void simulate(uint num_sims, global ulong* seed, global ushort* res){\n",
"	uint gid = get_global_id(0);\n",
"	tyche_i_state state;\n",
"	tyche_i_seed(&state, seed[gid]);\n",
"	for(int i = 0; i < num_sims; i++){\n",
"		res[gid * num_sims + i] = tyche_i_uint(state) % 4 == 0;\n",
"	}\n",
"}\n"
};

#define COMPILER_OPTS "-I ../generators/"
#define NUM_SEEDS 16
#define NUM_SIMS 128


int main(){
	size_t param_max_size = 2048;
	void* param = malloc(param_max_size);
	size_t param_size;
	cl_int CL_err = CL_SUCCESS;
	cl_uint num_platforms = 0;
	size_t platforms_max_size = 5;
	cl_platform_id* platforms = malloc(sizeof(cl_platform_id) * platforms_max_size);

	CL_err = clGetPlatformIDs(platforms_max_size, platforms, &num_platforms);

	if(CL_err == CL_SUCCESS)
		printf("%u platforms(s) found.\n", num_platforms);
	else
		printf("clGetPlatformIDs(%i)\n", CL_err);

	cl_platform_id amd_platform;
	cl_platform_id pocl_platform;
	get_platforms(platforms, num_platforms, &amd_platform, &pocl_platform);


	cl_device_id gpu; 
	cl_device_id cpu;
	get_devices(amd_platform, pocl_platform, &gpu, &cpu);

	cl_context context = clCreateContext(NULL, 1, &gpu, NULL, NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateContext", CL_err, __LINE__);
	}

	cl_program program = clCreateProgramWithSource(context, 9, program_text, NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateProgramWithSource", CL_err, __LINE__);
	}	

	CL_err = clBuildProgram(program, 1, &gpu, COMPILER_OPTS, NULL, NULL);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clBuildProgram", CL_err, __LINE__);
		param_max_size = 2048;
		param = malloc(param_max_size);
		param_size;
		CL_err = clGetProgramBuildInfo(program, gpu, CL_PROGRAM_BUILD_LOG, param_max_size, param, &param_size);
		printf("%s\n", (char *) param);
		free(param);
		
	}

	cl_kernel kernel = clCreateKernel(program, "simulate", &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateKernel", CL_err, __LINE__);
	}

	cl_mem seeds = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint64_t) * NUM_SEEDS, NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateBuffer", CL_err, __LINE__);
	}
	srand(time(NULL));
	uint64_t *host_seeds = malloc(sizeof(uint64_t) * NUM_SEEDS);
	create_seeds(host_seeds);

	cl_mem res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint16_t) * NUM_SEEDS * NUM_SIMS, NULL, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateBuffer", CL_err, __LINE__);
	}

	cl_command_queue queue = clCreateCommandQueue(context, gpu, CL_QUEUE_PROFILING_ENABLE, &CL_err);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clCreateCommandQueue", CL_err, __LINE__);
	}
	
	cl_event seeds_write;
	CL_err = clEnqueueWriteBuffer(queue, seeds, CL_FALSE, 0, sizeof(uint64_t) * NUM_SEEDS, host_seeds, 0, NULL, &seeds_write);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clEnqueueWriteBuffer", CL_err, __LINE__);	
	}

	int arg0 = NUM_SIMS;
	CL_err = clSetKernelArg(kernel, 0, sizeof(arg0), &arg0);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clSetKernelArg", CL_err, __LINE__);
	}
	CL_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &seeds);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clSetKernelArg", CL_err, __LINE__);
	}
	CL_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &res);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clSetKernelArg", CL_err, __LINE__);
	}

	cl_event execute_kernel;
	size_t work_size = NUM_SIMS;
	CL_err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size, &work_size, 1, &seeds_write, &execute_kernel);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clEnqueueNDRangeKernel", CL_err, __LINE__);
	}
	
	uint16_t *host_res = malloc(sizeof(uint16_t) * NUM_SEEDS * NUM_SIMS);
	CL_err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, sizeof(uint16_t) * NUM_SEEDS * NUM_SIMS, host_res, 1, &execute_kernel, NULL);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clEnqueueReadBuffer", CL_err, __LINE__);
	}

	for(int i = 0; i < 300;i ++){
		printf("%i\n", host_res[i]);
	}

	free(host_seeds);
	free(host_res);
	return 0;
}

void create_seeds(uint64_t* host_seeds){
	for(int i = 0; i < NUM_SEEDS; i++){
		host_seeds[i] = rand();
	}
}

void get_devices(cl_platform_id amd_platform, cl_platform_id pocl_platform, cl_device_id* gpu, cl_device_id* cpu){
	cl_uint CL_err;
	size_t devices_max_size = 5;
	cl_device_id* devices = malloc(sizeof(cl_device_id) * devices_max_size);
	cl_uint num_amd_devices = 0;
	cl_uint num_pocl_devices = 0;

	CL_err = clGetDeviceIDs(amd_platform, CL_DEVICE_TYPE_ALL, devices_max_size, devices, &num_amd_devices);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clGetDeviceIDs", CL_err, __LINE__);
	}

	CL_err = clGetDeviceIDs(pocl_platform, CL_DEVICE_TYPE_ALL, devices_max_size - num_amd_devices, devices + num_amd_devices, &num_pocl_devices);
	if(CL_err != CL_SUCCESS){
		print_cl_error("clGetDeviceIDs", CL_err, __LINE__);
	}

	size_t param_max_size = 100;
	void* param = malloc(param_max_size);
	size_t param_size;

	for(int i = 0; i < num_amd_devices + num_pocl_devices; i++){
		CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, param_max_size, param, &param_size);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clGetDeviceInfo", CL_err, __LINE__);
		}
		if(strcmp("pthread-AMD Ryzen 7 1700X Eight-Core Processor", (char *) param) == 0){
			printf("Found CPU.\n");
			*cpu = devices[i];
		}
		if(strcmp("gfx1031", (char *) param) == 0){
			printf("Found GPU.\n");
			*gpu = devices[i];
		}
	}
	free(devices);
}

void get_platforms(cl_platform_id* platforms, cl_uint num_platforms, cl_platform_id* amd_platform, cl_platform_id* pocl_platform){
	cl_int CL_err;
	size_t param_max_size = 100;
	void* param = malloc(param_max_size);
	size_t param_size;
	for(int i = 0; i < num_platforms; i++){
		CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_max_size, param, &param_size);
		if(CL_err != CL_SUCCESS){
			print_cl_error("clGetPlatformInfo", CL_err, __LINE__);
		}
		if(strcmp("AMD Accelerated Parallel Processing", (char *) param) == 0){
			printf("Found AMD APP Platform.\n");
			*amd_platform = platforms[i];
		}
		if(strcmp("Portable Computing Language", (char *) param) == 0){
			printf("Found PoCL.\n");
			*pocl_platform = platforms[i];
		}
	}
	free(param);
}

void print_cl_error(char* function_name,  cl_int err, int line_num){
	printf("%s:%i %s(%i)\n", __FILE__, line_num, function_name, err);
}
