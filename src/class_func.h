// Updated at Feb. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iomanip>

#include "utility.h"
#include "version.h"
#include "hash.h"
#include "table_format.h"
#include "db.h"
#include "otu_parser.h"
#include "timer.h"

#ifndef class_func_h
#define class_func_h

using namespace std;

#define BUFF 10000000
#define STRLEN 1000

class _KO_Index_Copy {

	public:
		_KO_Index_Copy() {
			Index = 0;
			Copy = 0;
		}
		_KO_Index_Copy(int index, float copy) {
			Index = index;
			Copy = copy;
		}

		int Index;
		float Copy;
};



class _KO {

	public:
		_KO() {
			Name = "K_NA";
			Des = "None";
			Pathway = "None";
			Index = 0;
		}

		_KO(string name, string des, string pathway, int index) {
			Name = name;
			Des = des;
			Pathway = pathway;
			Index = index;
		}

		string Get_Name() {
			return Name;
		}

		string Get_Des() {
			return Des;
		}

		string Get_Pathway() {
			return Pathway;
		}

		int Get_Index() {
			return Index;
		}

	private:
		string Name;
		string Des;
		string Pathway;
		int Index;
};

class _KO_OTU_Table_All {

	public:
		_KO_OTU_Table_All() {
			Sample_count = 0;
		}

		_KO_OTU_Table_All(_PMDB db, int count, int mode, bool skipNormalization);
		_KO_OTU_Table_All(_PMDB db, int count, int mode, int coren, bool skipNormalization);
		int Load_KO_Id(const char * idfilename, const char * desfilename, const char * pathwayfilename);
		int Load_OTU_KO_Index(const char * abdfilename);
		int Load_OTU_KO_Index_openMP(const char * abdfilename, int coren);
		int Load_Sample(const char * infilename, string sample_name, int sample, bool skipNormalization, double max_nsti);
		int Load_Sample_By_OTU_Table(_Table_Format * table, int coren, bool skipNormalization, double max_nsti);
		int Load_Sample_By_OTU_Table_G(_Table_Format * table, int coren, bool skipNormalization, double max_nsti, string gpuMode, int gpu_number);
		int Load_Sample_By_OTU_Table_newG(_Table_Format * table, int coren, bool skipNormalization, string gpuMode, int num, bool over, int gpu_number);
		int Output_By_Category(const char * outfilename, int level, float max, float min);
		int Output_By_Category_openMP(const char * outfilename, int level, float max, float min, int coren);
		int Output_By_Category_app_openMP(const char * outfilename, int level, float max, float min, int coren, int num);
		int Output_By_KO_openMP(const char * outfilename, int level, float max, float min, int coren);
		int Output_By_KO_app_openMP(const char * outfilename, int level, float max, float min, int coren, int num);

		void Output_Nsti(bool useGPU, const char * outfilename, _Table_Format * table, int coren, bool skipNormalization, double max_nsti, int num, bool over);

		void matrixMultiplyM(float *B, float *A, size_t N, size_t K, size_t M, int coren, int gpu_number);
		void matrixMultiplyS(float *B, float *A, size_t N, size_t K, size_t M, int coren);
		void matrixMultiplyS_more(float *B, float *A, size_t N, size_t K, size_t M, int coren);
		void matrixMultiplyM_more(float *B, float *A, size_t N, size_t K, size_t M, int coren);
		void make_B(_Table_Format * table, double max_nsti, int coren);
		void make_batch_S(_Table_Format * table, size_t N, size_t K, size_t M);
		void make_batch_M(_Table_Format * table, size_t N, size_t K, size_t M, int coren, int gpu_number);
		void releaseMemory();
	protected:
		int Sample_count;
		string * Sample_names;
		vector <_KO> KOs; //KO information
		hash_map <string, int, std_string_hash> KO_Index; //KO number to index
		hash_map <string, vector<_KO_Index_Copy>, std_string_hash> OTU_KO_Index; //OTU to KO
		vector<vector<float>> KO_Abd;
		_OTU_Parser Otu_parser;
		float total_time = 0.0f;

		float *B_host = nullptr;

		vector<float*> device_B;
		vector<float*> device_A;
		vector<float*> device_C;
		vector<float*> result_C;
		vector<cublasHandle_t> handles;

		// === events and time ===
		vector<cudaEvent_t> start_events;
		vector<cudaEvent_t> stop_events;
		vector<float> gpu_times;  // all GPUs time

		// Partition and Batch size
		size_t batchSize_B;
		size_t batchSize_A;

		int max_gpus;
		int actual_coren;

		bool need_batching = false;

		virtual int Load_Sample_By_OTU(hash_map <string, int, std_string_hash> * otu_seq_count, int sample, bool skipNormalization, double max_nsti);
};

double Get_Nsti(_OTU_Parser* Otu_parser, vector<string>& OTU, float* A, size_t sample_number, size_t feature_size, bool skipNormalization, double max_nsti) {
	double count = 0;
	double nsti_value = 0;

	for (size_t i = 0; i < feature_size; i++) {
		size_t idx = sample_number * feature_size + i;
		float abd = A[idx];
		if (abd <= 0.0) continue;

		string a_otu = Check_OTU(OTU[i]);
		double nsti = Otu_parser->Get_nsti_by_OTU(a_otu);
		if (nsti == 0) continue;

		float cp_number = Otu_parser->Get_cp_by_OTU(a_otu);
		if (skipNormalization) cp_number = 1.0;

		if (nsti < 0 || nsti > max_nsti) {
			abd = 0.0;
		}

		count += abd / cp_number;
		nsti_value += nsti * abd / cp_number;
	}

	if (count == 0) return 0;
	return nsti_value / count;
}

double Get_Nsti(_OTU_Parser* Otu_parser, vector<string> OTU, vector<float> seq_count, bool skipNormalization, double max_nsti) {
	double count = 0;
	double nsti_value = 0;

	size_t feature_size = OTU.size();
	for (size_t i = 0; i < feature_size; i++) {
		if (seq_count[i] <= 0.0) continue;

		float abd = seq_count[i];
		string a_otu = Check_OTU(OTU[i]);

		double nsti = Otu_parser->Get_nsti_by_OTU(a_otu);
		if (nsti == 0) continue;

		float cp_number = Otu_parser->Get_cp_by_OTU(a_otu);
		if (skipNormalization) cp_number = 1.0;

		if (nsti < 0 || nsti > max_nsti) {
			abd = 0.0;
		}

		count += abd / cp_number;
		nsti_value += nsti * abd / cp_number;
	}

	if (count == 0) return 0;
	return nsti_value / count;
}

void _KO_OTU_Table_All::Output_Nsti(bool useGPU, const char * outfilename, _Table_Format * table, int coren, bool skipNormalization, double max_nsti, int num, bool over) {
	// If this is the first write, delete the old file and create a new one
	if (num == 0) {
		ofstream outfile(outfilename, ios::out | ios::trunc);
		if (!outfile) {
			cerr << "Error: Cannot open output file: " << outfilename << endl;
			return;
		}
		outfile << "sample\tweighted_NSTI" << endl;
		outfile.close();
	}

	// Afterwards, always write in append mode
	ofstream outfile(outfilename, ios::out | ios::app);
	if (!outfile) {
		cerr << "Error: Cannot open output file: " << outfilename << endl;
		return;
	}

	size_t sample_size = table->Get_Sample_Size();
	double * nsti_values = new double[sample_size];
	vector<string> OTUs = table->Get_Feature_Names();

	if (useGPU) {
		float *A = table->GetA();
		size_t M = sample_size;
		size_t K = table->Get_Feature_Size();
		omp_set_num_threads(coren);
		#pragma omp parallel for schedule(dynamic, 1)
		for (size_t i = 0; i < M; i++) {
			nsti_values[i] = Get_Nsti(&Otu_parser, OTUs, A, i, K, skipNormalization, max_nsti);
		}
	} else {
		omp_set_num_threads(coren);
		#pragma omp parallel for schedule(dynamic, 1)
		for (size_t i = 0; i < sample_size; i++) {
			vector<float> abd = table->Get_Abd(i);
			nsti_values[i] = Get_Nsti(&Otu_parser, OTUs, abd, skipNormalization, max_nsti);
		}
	}

	outfile << std::fixed << std::setprecision(15);
	vector<string> sample_names = table->Get_Sample_Names();
	for (size_t i = 0; i < sample_size; i++) {
		outfile << sample_names[i] << "\t" << nsti_values[i] << endl;
	}

	delete[] nsti_values;

	if(over) {
		table->ReleaseMemory();
		outfile.close();
		outfile.clear();
	}
}

// CUDA error check
#define CHECK_CUDA_ERROR(err) if (err != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE); \
	}

// CUBLAS error check
#define CHECK_CUBLAS_ERROR(err) if (err != CUBLAS_STATUS_SUCCESS) { \
		std::cerr << "CUBLAS Error" << std::endl; \
		exit(EXIT_FAILURE); \
	}

// calculate need memory
size_t calculate_memory_size(size_t N, size_t K, size_t M) {
	size_t element_size = sizeof(float);
	size_t size_B = N * K * element_size;  // database
	size_t size_A = K * M * element_size;  // sample
	size_t size_C = N * M * element_size;  // result
	return size_B + size_A + size_C;       // all
}

void _KO_OTU_Table_All::make_batch_S(_Table_Format * table, size_t N, size_t K, size_t M) {
	// 1. get total GPU memory
	size_t total_mem = 0;
	cudaMemGetInfo(&total_mem, nullptr);

	// 2. calculate needed memory
	size_t required_mem = calculate_memory_size(N, K, M);
	cout << "Required memory: " << required_mem / (1024 * 1024) << " MB" << endl;
	cout << "Available memory: " << total_mem / (1024 * 1024) << " MB" << endl;

	// 3. whether batch splitting is needed
	need_batching = false;

	size_t element_size = sizeof(float);

	// Check whether the size of database exceeds 70% of GPU memory
	if (N * K * element_size > total_mem * 0.7) {
		need_batching = true;
		batchSize_B = static_cast<size_t>(total_mem * 0.7 / K / element_size); // Split database into partitions occupying 70% of GPU memory

		size_t A_C_size = K * M * element_size + N * M * element_size;
		if (A_C_size > total_mem * 0.2) {
			batchSize_A = static_cast<size_t>(total_mem * 0.2 / element_size / (K + N)); // Split sample
		} else {
			batchSize_A = M;  // do not need split sample, calculate directly
		}
	} else {
		size_t B_A_C_size = N * K * element_size + K * M * element_size + N * M * element_size;
		if (B_A_C_size <= total_mem * 0.9) {
			batchSize_B = N;
			batchSize_A = M;  // calculate directly
		} else {
			need_batching = true;
			size_t available_mem_for_A_C = total_mem * 0.9 - (N * K * element_size);
			batchSize_A = static_cast<size_t>(available_mem_for_A_C / (K * element_size + N * element_size));
			batchSize_B = N;
		}
	}

	cout << "Partition-database Size: " << batchSize_B << ", Batch-sample Size: " << batchSize_A << endl;
	cout << "Need batching: " << need_batching << endl;

	max_gpus = 1;
	cout << "use " << max_gpus << " GPUs." << endl;

	device_A.resize(1);
	device_B.resize(1);
	device_C.resize(1);
	result_C.resize(1);
	handles.resize(1);

	start_events.resize(1);
	stop_events.resize(1);
	gpu_times.resize(1);

	// allocate GPU memory
	CHECK_CUDA_ERROR(cudaMalloc((void**)&device_B[0], batchSize_B * K * sizeof(float))); // database
	CHECK_CUDA_ERROR(cudaMalloc((void**)&device_A[0], K * batchSize_A * sizeof(float))); // sample
	CHECK_CUDA_ERROR(cudaMalloc((void**)&device_C[0], batchSize_B * batchSize_A * sizeof(float))); // result
	CHECK_CUBLAS_ERROR(cublasCreate(&handles[0]));

	// create cuda event
	cudaEventCreate(&start_events[0]);
	cudaEventCreate(&stop_events[0]);

	if (!need_batching) {
		// allocate result in RAM
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&result_C[0], N * M * sizeof(float), cudaHostAllocDefault));
		// check
		if (result_C[0] != nullptr) {
			printf("result Memory allocation successful!\n");
		} else {
			fprintf(stderr, "result Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		// transfer database from RAM to GPU
		CHECK_CUDA_ERROR(cudaMemcpy(device_B[0], B_host, batchSize_B * K * sizeof(float), cudaMemcpyHostToDevice));
	} else {
		// allocate result in RAM
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&result_C[0], batchSize_B * batchSize_A * sizeof(float), cudaHostAllocDefault));
		if (result_C[0] != nullptr) {
			printf("result Memory allocation successful!\n");
		} else {
			fprintf(stderr, "result Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}
	}
}

void _KO_OTU_Table_All::releaseMemory() {
	if (max_gpus == 1) {
		// release resource by single GPU
		cudaFree(device_A[0]);
		cudaFree(device_B[0]);
		cudaFree(device_C[0]);
		cudaFreeHost(result_C[0]);
		cublasDestroy(handles[0]);

		cudaEventDestroy(start_events[0]);
		cudaEventDestroy(stop_events[0]);
	} else {
		// release resource by multiple GPUs
		#pragma omp parallel for num_threads(max_gpus)
		for (int i = 0; i < max_gpus; ++i) {
			cudaSetDevice(i);
			cudaFree(device_A[i]);
			cudaFree(device_B[i]);
			cudaFree(device_C[i]);
			cudaFreeHost(result_C[i]);
			cublasDestroy(handles[i]);

			cudaEventDestroy(start_events[i]);
			cudaEventDestroy(stop_events[i]);
		}
	}
}

void _KO_OTU_Table_All::make_batch_M(_Table_Format * table, size_t N, size_t K, size_t M, int coren, int gpu_number) {
//	// Automatically get the number of available GPUs
//	int gpu_count = 0;
//	CHECK_CUDA_ERROR(cudaGetDeviceCount(&gpu_count));
//	if (gpu_count == 0) {
//		cerr << "No GPU devices found!" << endl;
//		exit(EXIT_FAILURE);
//	}
//	cout << "Found " << gpu_count << " GPUs available." << endl;

	// Automatically get the number of available GPUs
	int gpu_count = 0;
	CHECK_CUDA_ERROR(cudaGetDeviceCount(&gpu_count));
	cout << "Found " << gpu_count << " GPUs available." << endl;
	if (gpu_number > 0 && gpu_number <= gpu_count) {
		gpu_count = gpu_number;
	}
	if (gpu_count == 0) {
		cerr << "No GPU devices found!" << endl;
		exit(EXIT_FAILURE);
	}

	// 1. get total GPU memory
	size_t total_mem = 0;
	cudaMemGetInfo(&total_mem, nullptr);

	// 2. calculate needed memory
	size_t required_mem = calculate_memory_size(N, K, M);
	cout << "Required memory: " << required_mem / (1024 * 1024) << " MB" << endl;
	cout << "Available memory: " << total_mem / (1024 * 1024) << " MB" << endl;

	// 3. whether batch splitting is needed
	size_t element_size = sizeof(float);

	// Check whether the size of database exceeds 70% of GPU memory
	if (N * K * element_size > total_mem * 0.7) {
		need_batching = true;
		batchSize_B = static_cast<size_t>(total_mem * 0.7 / K / element_size); // Split database into partitions occupying 70% of GPU memory

		size_t A_C_size = K * M * element_size + N * M * element_size;
		if (A_C_size > total_mem * 0.2) {
			batchSize_A = static_cast<size_t>(total_mem * 0.2 / element_size / (K + N)); // Split sample
		} else {
			batchSize_A = M;  // do not need split sample, calculate directly
		}
	} else {
		size_t B_A_C_size = N * K * element_size + K * M * element_size + N * M * element_size;
		if (B_A_C_size <= total_mem * 0.9) {
			batchSize_B = N;
			batchSize_A = M;  // calculate directly
		} else {
			need_batching = true;
			size_t available_mem_for_A_C = total_mem * 0.9 - (N * K * element_size);
			batchSize_A = static_cast<size_t>(available_mem_for_A_C / (K * element_size + N * element_size));
			batchSize_B = N;
		}
	}

//	size_t batch = (M + gpu_count - 1) / gpu_count;
	size_t batch = M / gpu_count;
	if (batch < 1) batch = 1;
	while (batch > batchSize_A) {
		batch = batch / gpu_count;
		if (batch < 1) batch = 1;
	}
	batchSize_A = batch;
	need_batching = true;

	cout << "Partition-database Size: " << batchSize_B << ", Batch-sample Size:: " << batchSize_A << endl;
	cout << "Need batching: " << need_batching << endl;

	// calculate the number of batch (sample)
	size_t num_batches_A = (M + batchSize_A - 1) / batchSize_A; // round up
	cout << "number of Batch-sample: " << num_batches_A << endl;

	// If the number of sample batches is less than the number of GPUs, allocate memory only for the GPUs actually in use
	max_gpus = min(gpu_count, static_cast<int>(num_batches_A));
	cout << "use " << max_gpus << " GPUs." << endl;

	// adjust the size of vector by the number of GPU
	device_A.resize(max_gpus);
	device_B.resize(max_gpus);
	device_C.resize(max_gpus);
	result_C.resize(max_gpus);
	handles.resize(max_gpus);

	start_events.resize(max_gpus);
	stop_events.resize(max_gpus);
	gpu_times.resize(max_gpus);

	// allocate memory and cublasHandle_t for all GPUs
	#pragma omp parallel for num_threads(max_gpus)
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_CUDA_ERROR(cudaSetDevice(i));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&device_B[i], batchSize_B * K * sizeof(float)));  // database
		CHECK_CUDA_ERROR(cudaMalloc((void**)&device_A[i], K * batchSize_A * sizeof(float)));  // sample
		CHECK_CUDA_ERROR(cudaMalloc((void**)&device_C[i], batchSize_B * batchSize_A * sizeof(float)));  // result
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&result_C[i], batchSize_B * batchSize_A * sizeof(float), cudaHostAllocDefault));

		// create cublas handle
		CHECK_CUBLAS_ERROR(cublasCreate(&handles[i]));

		// create cuda event
		CHECK_CUDA_ERROR(cudaEventCreate(&start_events[i]));
		CHECK_CUDA_ERROR(cudaEventCreate(&stop_events[i]));
	}

	actual_coren = coren/max_gpus - max_gpus;
	cout << "actual_coren = " << actual_coren << endl;
}

_KO_OTU_Table_All::_KO_OTU_Table_All(_PMDB db, int count, int mode, bool skipNormalization) { //mode 0: norm; mode 1: sim

	string ko_id_file = db.Get_Func_Id();
	string ko_abd_file = db.Get_Func();
	string ko_des_file = db.Get_Func_Des();
	string ko_pw_file = db.Get_Func_Pw();

	Load_KO_Id(ko_id_file.c_str(), ko_des_file.c_str(), ko_pw_file.c_str());

	Sample_count = count;
	Sample_names = new string[count];

	if (mode == 0)
		Load_OTU_KO_Index(ko_abd_file.c_str());

	Otu_parser = _OTU_Parser(db, skipNormalization);

	//Init KO_Abd
//	KO_Abd = new float * [count];
//	for (int i = 0; i < count; i ++) {
//		KO_Abd[i] = new float [KOs.size()];
//		for (int j = 0; j < KOs.size(); j ++)
//			KO_Abd[i][j] = 0;
//	}

	KO_Abd.resize(count, vector<float>(KOs.size(), 0));

}
_KO_OTU_Table_All::_KO_OTU_Table_All(_PMDB db, int count, int mode, int coren, bool skipNormalization) { //mode 0: norm; mode 1: sim

	string ko_id_file = db.Get_Func_Id();
	string ko_abd_file = db.Get_Func();
	string ko_des_file = db.Get_Func_Des();
	string ko_pw_file = db.Get_Func_Pw();

	Load_KO_Id(ko_id_file.c_str(), ko_des_file.c_str(), ko_pw_file.c_str());

	Sample_count = count;
	Sample_names = new string[count];

	if (mode == 0)
		Load_OTU_KO_Index_openMP(ko_abd_file.c_str(), coren);

	Otu_parser = _OTU_Parser(db, coren, skipNormalization);

	//Init KO_Abd
//	KO_Abd = new float * [count];
//	for (int i = 0; i < count; i ++) {
//		KO_Abd[i] = new float [KOs.size()];
//		for (int j = 0; j < KOs.size(); j ++)
//			KO_Abd[i][j] = 0;
//	}
	KO_Abd.resize(count, vector<float>(KOs.size(), 0));

}

void _KO_OTU_Table_All::make_B(_Table_Format * table, double max_nsti, int coren) {
	size_t N = KO_Index.size();    // Number of rows in B
	size_t K = table->Get_Feature_Size();  // Number of columns in B, number of rows in A
	size_t M = table->Get_Sample_Size();    // Number of columns in A

	Sample_count = M;

	vector<string> otus = table->Get_Feature_Names();

//	if (cudaHostAlloc((void**)&B_host, N * K * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
//		fprintf(stderr, "Error allocating memory for B\n");
//		exit(EXIT_FAILURE);
//	}

	B_host = (float*)malloc(N * K * sizeof(float));

	// check and init database to 0
	if (B_host != nullptr) {
		printf("database Memory allocation successful!\n");
		memset(B_host, 0, N * K * sizeof(float));
	} else {
		fprintf(stderr, "database Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	omp_set_num_threads(coren);
	// fill database
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < K; ++i) {
		string a_otu_j = Check_OTU(otus[i]);

		// Use find() to avoid implicit write by operator[], ensuring thread safety
		auto it = OTU_KO_Index.find(a_otu_j);
		if (it != OTU_KO_Index.end()) {
			vector<_KO_Index_Copy>& kos = it->second;
			double nsti = Otu_parser.Get_nsti_by_OTU(a_otu_j);
			if (nsti >= 0 && nsti <= max_nsti) {
				for (int j = 0; j < kos.size(); ++j) {
					B_host[i * N + kos[j].Index] = kos[j].Copy;
				}
			}
		} else {
			// Skip if OTU index does not exist
			continue;
		}
	}

}

int _KO_OTU_Table_All::Load_KO_Id(const char * idfilename, const char * desfilename, const char * pathwayfilename) {
	ifstream in_idfile(idfilename, ifstream::in);
	if (!in_idfile) {
		cerr << "Error: Open KO ID file error : " << idfilename << endl;
		exit(0);
	}

	ifstream in_desfile(desfilename, ifstream::in);
	if (!in_desfile) {
		cerr << "Error: Open KO description file error : " << desfilename << endl;
		exit(0);
	}

	ifstream in_pwfile(pathwayfilename, ifstream::in);
	if (!in_pwfile) {
		cerr << "Error: Open KO pathway file error : " << pathwayfilename << endl;
		exit(0);
	}

	string buffer_id;
	string buffer_des;
	string buffer_pw;
	int i = 0;
	while(getline(in_idfile, buffer_id)) {
		getline(in_desfile, buffer_des);
		getline(in_pwfile, buffer_pw);
		KOs.push_back(_KO(buffer_id, buffer_des, buffer_pw, i));
		KO_Index[buffer_id] = i;
		i ++;
	}

	in_idfile.close();
	in_idfile.clear();

	in_desfile.close();
	in_desfile.clear();

	in_pwfile.close();
	in_pwfile.clear();

	return KOs.size();

}

int _KO_OTU_Table_All::Load_OTU_KO_Index(const char * abdfilename) {
	ifstream infile(abdfilename, ifstream::in);
	if (!infile) {
		cerr << "Error: Open KO Abduncance file error : " << abdfilename << endl;
		exit(0);
	}

	string buffer;
	while(getline(infile, buffer)) {

		stringstream strin(buffer);
		string id;
		int index;
		float copy;

		strin >> id;

		while(strin >> index) {
			strin >> copy;
			//OTU_KO_Index[id].push_back(_KO_Index_Copy(index, copy));
			//debug
			OTU_KO_Index[id].push_back(_KO_Index_Copy(index - 1, copy));
		}

	}

	infile.close();
	infile.clear();

	return OTU_KO_Index.size();

}

int _KO_OTU_Table_All::Load_OTU_KO_Index_openMP(const char * abdfilename, int coren) {
	ifstream infile(abdfilename);
	if (!infile) {
		cerr << "Error: Open KO Abundance file error: " << abdfilename << endl;
		exit(1);
	}

	// read data
	vector<string> lines;
	string buffer;
	while (getline(infile, buffer)) {
		lines.push_back(buffer);
	}
	infile.close(); // close file

//	cout << "lines.size() = " << lines.size() << endl;

	// Build OTU\_KO\_Index sequentially using OpenMP
	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(dynamic, 1)
	for (size_t j = 0; j < lines.size(); ++j) {
		stringstream strin(lines[j]);
		string id;
		int index;
		float copy;

		strin >> id;

		vector<_KO_Index_Copy> local_data;

		while (strin >> index >> copy) {
			local_data.emplace_back(index - 1, copy);
		}

		#pragma omp critical
		{
			OTU_KO_Index[id].insert(OTU_KO_Index[id].end(), local_data.begin(), local_data.end());
		}

	}

	return OTU_KO_Index.size(); // return the size of OTU_KO_Index

}

int _KO_OTU_Table_All::Load_Sample_By_OTU_Table(_Table_Format * table, int coren, bool skipNormalization, double max_nsti) {

	hash_map <string, int, std_string_hash> * otu_seq_count = new hash_map <string, int, std_string_hash> [table->Get_Sample_Size()];

	vector <string> otus = table->Get_Feature_Names();
	vector <string> sam_names = table->Get_Sample_Names();

	for (int i = 0; i < table->Get_Sample_Size(); i ++) {

		Sample_names[i] = sam_names[i];
		//vector <float> abd= table->Get_Abd(i);

		for (int j = 0; j < table->Get_Feature_Size(); j ++)
			if (table->Get_Abd_By_Order(i, j) > 0) {
				string a_otu_j = Check_OTU(otus[j]);
				otu_seq_count[i][a_otu_j] = (int) table->Get_Abd_By_Order(i, j);
			}
	}

	my_timer timer;
	double t_calc = 0.0;

	timer.start();

	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(dynamic, 1)
	for(int i = 0; i < table->Get_Sample_Size(); i ++)
		Load_Sample_By_OTU(&(otu_seq_count[i]), i, skipNormalization, max_nsti);

	timer.stop();
	t_calc += timer.dt;
	printf("Calculation took %8.3f ms.\n",t_calc);

	return table->Get_Sample_Size();
}

void _KO_OTU_Table_All::matrixMultiplyS(float *B, float *A, size_t N, size_t K, size_t M, int coren) {

	// 1. get total GPU memory
	size_t total_mem = 0;
	cudaMemGetInfo(&total_mem, nullptr);

	// 2. calculate needed memory
	size_t required_mem = calculate_memory_size(N, K, M);
	cout << "Required memory: " << required_mem / (1024 * 1024) << " MB" << endl;
	cout << "Available memory: " << total_mem / (1024 * 1024) << " MB" << endl;

	// 3. whether batch splitting is needed
	size_t batchSize_B, batchSize_A;
	bool need_batching = false;

	size_t element_size = sizeof(float);

	// Check whether the size of database exceeds 70% of GPU memory
	if (N * K * element_size > total_mem * 0.7) {
		need_batching = true;
		batchSize_B = static_cast<size_t>(total_mem * 0.7 / K / element_size); // Split database into partitions occupying 70% of GPU memory

		size_t A_C_size = K * M * element_size + N * M * element_size;
		if (A_C_size > total_mem * 0.2) {
			batchSize_A = static_cast<size_t>(total_mem * 0.2 / element_size / (K + N)); // Split sample
		} else {
			batchSize_A = M;  // do not need split sample, calculate directly
		}
	} else {
		size_t B_A_C_size = N * K * element_size + K * M * element_size + N * M * element_size;
		if (B_A_C_size <= total_mem * 0.9) {
			need_batching = false;
			batchSize_B = N;
			batchSize_A = M;  // calculate directly
		} else {
			size_t available_mem_for_A_C = total_mem * 0.9 - (N * K * element_size);
			batchSize_A = static_cast<size_t>(available_mem_for_A_C / (K * element_size + N * element_size));
			batchSize_B = N;
		}
	}

	cout << "Partition-database Size: " << batchSize_B << ", Batch-sample Size: " << batchSize_A << endl;
	cout << "Need batching: " << need_batching << endl;

	// 4. allocate memory in GPU
	float *device_B, *device_A, *device_C;
	cublasHandle_t handle;
	CHECK_CUBLAS_ERROR(cublasCreate(&handle));

	// 5. allocate memory in GPU
	CHECK_CUDA_ERROR(cudaMalloc((void**)&device_B, batchSize_B * K * sizeof(float))); // database
	CHECK_CUDA_ERROR(cudaMalloc((void**)&device_A, K * batchSize_A * sizeof(float))); // sample
	CHECK_CUDA_ERROR(cudaMalloc((void**)&device_C, batchSize_B * batchSize_A * sizeof(float))); // result

	// time
	cudaEvent_t start, stop;
	float time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float dt;

	// 6. calculate directly
	if (!need_batching) {

		// allocate result memory in RAM
		float *C;
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&C, N * M * sizeof(float), cudaHostAllocDefault));
		// check result
		if (C != nullptr) {
			printf("result Memory allocation successful!\n");
		} else {
			fprintf(stderr, "result Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		// 6.1 transfer database from RAM to GPU
		CHECK_CUDA_ERROR(cudaMemcpy(device_B, B, batchSize_B * K * sizeof(float), cudaMemcpyHostToDevice));

		for (size_t j = 0; j < M; j += batchSize_A) {
			size_t cur_batchSize_A = min(batchSize_A, M - j);
			size_t offset = j * K;

			// 6.3 transfer batch from RAM to GPU
			CHECK_CUDA_ERROR(cudaMemcpy(device_A, A + offset, K * cur_batchSize_A * sizeof(float), cudaMemcpyHostToDevice));

			// 6.2 init result to 0
			CHECK_CUDA_ERROR(cudaMemset(device_C, 0, batchSize_B * batchSize_A * sizeof(float)));

			// 6.4 calculation
			const float alpha = 1.0f;
			const float beta = 0.0f;

			cudaEventRecord(start); // record start
			CHECK_CUBLAS_ERROR(cublasSgemm(handle,
			                               CUBLAS_OP_N, CUBLAS_OP_N,
			                               N, cur_batchSize_A, K,
			                               &alpha,
			                               device_B, N,
			                               device_A, K,
			                               &beta,
			                               device_C, N));
			cudaEventRecord(stop);  // record end
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&dt, start, stop);
			printf("Batch-sample %zu: Calculation took %8.3f ms.\n", j, dt);
			time += dt;

			// 6.5 transfer result from GPU to RAM
			CHECK_CUDA_ERROR(cudaMemcpy(C + j * N, device_C, N * cur_batchSize_A * sizeof(float), cudaMemcpyDeviceToHost));
		}
		printf("Calculation took %8.3f ms.\n", time);

		// normalization for output
		omp_set_num_threads(coren);
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				KO_Abd[i][j] = C[i * N + j];
			}
		}

		CHECK_CUDA_ERROR(cudaFreeHost(C));

	} else {
		// 7.1 allocate result in RAM
		float *result_C;
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&result_C, batchSize_B * batchSize_A * sizeof(float), cudaHostAllocDefault));
		if (result_C != nullptr) {
			printf("result Memory allocation successful!\n");
		} else {
			fprintf(stderr, "result Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		omp_set_num_threads(coren);
		for (size_t i = 0; i < N; i += batchSize_B) {
			size_t cur_batchSize_B = min(batchSize_B, N - i);
			// 7.2 transfer current Partition from RAM to GPU
			#pragma omp parallel for schedule(static)
			for (size_t k = 0; k < K; ++k) {
				CHECK_CUDA_ERROR(cudaMemcpy(device_B + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), cudaMemcpyHostToDevice));
			}
			for (size_t j = 0; j < M; j += batchSize_A) {
				size_t cur_batchSize_A = min(batchSize_A, M - j);

				// 7.3 transfer current Batch from RAM to GPU
				size_t offset = j * K;
				CHECK_CUDA_ERROR(cudaMemcpy(device_A, A + offset, K * cur_batchSize_A * sizeof(float), cudaMemcpyHostToDevice));

				// 7.4 init result to 0
				CHECK_CUDA_ERROR(cudaMemset(device_C, 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

				// 7.5 calculation
				float alpha = 1.0f;
				float beta = 0.0f;

				cudaEventRecord(start); // record start
				CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_batchSize_B, cur_batchSize_A, K,
				                               &alpha, device_B, cur_batchSize_B, device_A, K, &beta, device_C, cur_batchSize_B));
				cudaEventRecord(stop);  // record end
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&dt, start, stop);
				printf("Partition-database %zu, Batch-sample %zu: Calculation took %8.3f ms.\n", i, j, dt);
				time += dt;

				// 7.6 transfer result from GPU to RAM
				CHECK_CUDA_ERROR(cudaMemcpy(result_C, device_C, cur_batchSize_B * cur_batchSize_A * sizeof(float), cudaMemcpyDeviceToHost));

				// 7.7 normalization for output
				#pragma omp parallel for schedule(static)
				for (size_t ai = 0; ai < cur_batchSize_A; ++ai) {
					for (size_t bi = 0; bi < cur_batchSize_B; ++bi) {
						size_t C_index = (j + ai) * N + (i + bi);
						size_t result_index = ai * cur_batchSize_B + bi;
						size_t i = C_index / N;
						size_t j = C_index - i * N;
						KO_Abd[i][j] = result_C[result_index];
					}
				}
			}
		}
		printf("Calculation took %8.3f ms.\n", time);
		CHECK_CUDA_ERROR(cudaFreeHost(result_C));
	}

	// 8. release resource for GPU
	cublasDestroy(handle);
	cudaFree(device_B);
	cudaFree(device_A);
	cudaFree(device_C);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

void _KO_OTU_Table_All::matrixMultiplyS_more(float *B, float *A, size_t N, size_t K, size_t M, int coren) {

	float dt;
	gpu_times[0] = 0.0f;

	// if don't split, calculate directly
	if (!need_batching) {
		for (size_t j = 0; j < M; j += batchSize_A) {
			size_t cur_batchSize_A = min(batchSize_A, M - j);
			size_t offset = j * K;

			// 6.3 transfer Batch from RAM to GPU
			CHECK_CUDA_ERROR(cudaMemcpy(device_A[0], A + offset, K * cur_batchSize_A * sizeof(float), cudaMemcpyHostToDevice));

			// 6.2 init result to 0
			CHECK_CUDA_ERROR(cudaMemset(device_C[0], 0, batchSize_B * batchSize_A * sizeof(float)));

			// 6.4 calculation
			const float alpha = 1.0f;
			const float beta = 0.0f;

			cudaEventRecord(start_events[0]); // record start
			CHECK_CUBLAS_ERROR(cublasSgemm(handles[0],
			                               CUBLAS_OP_N, CUBLAS_OP_N,
			                               N, cur_batchSize_A, K,
			                               &alpha,
			                               device_B[0], N,
			                               device_A[0], K,
			                               &beta,
			                               device_C[0], N));
			cudaEventRecord(stop_events[0]);  // record end
			cudaEventSynchronize(stop_events[0]);
			cudaEventElapsedTime(&dt, start_events[0], stop_events[0]);
			printf("Batch-sample %zu: Calculation took %8.3f ms.\n", j, dt);
			gpu_times[0] += dt;

			// 6.5 transfer result from GPU to RAM
			CHECK_CUDA_ERROR(cudaMemcpy(result_C[0] + j * N, device_C[0], N * cur_batchSize_A * sizeof(float), cudaMemcpyDeviceToHost));
		}
		printf("Calculation took %8.3f ms.\n", gpu_times[0]);
		total_time += gpu_times[0];

		// normalization for output
		omp_set_num_threads(coren);
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				KO_Abd[i][j] = result_C[0][i * N + j];
			}
		}
	} else {
		// allocate result in RAM
		omp_set_num_threads(coren);
		for (size_t i = 0; i < N; i += batchSize_B) {
			size_t cur_batchSize_B = min(batchSize_B, N - i);
			// 7.2 transfer current Partition from RAM to GPU
			#pragma omp parallel for schedule(static)
			for (size_t k = 0; k < K; ++k) {
				CHECK_CUDA_ERROR(cudaMemcpy(device_B[0] + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), cudaMemcpyHostToDevice));
			}
			for (size_t j = 0; j < M; j += batchSize_A) {
				size_t cur_batchSize_A = min(batchSize_A, M - j);

				// 7.3 transfer current Batch from RAM to GPU
				size_t offset = j * K;
				CHECK_CUDA_ERROR(cudaMemcpy(device_A[0], A + offset, K * cur_batchSize_A * sizeof(float), cudaMemcpyHostToDevice));

				// 7.4 init result to 0
				CHECK_CUDA_ERROR(cudaMemset(device_C[0], 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

				// 7.5 calculation
				float alpha = 1.0f;
				float beta = 0.0f;

				cudaEventRecord(start_events[0]); // record start
				CHECK_CUBLAS_ERROR(cublasSgemm(handles[0], CUBLAS_OP_N, CUBLAS_OP_N, cur_batchSize_B, cur_batchSize_A, K,
				                               &alpha, device_B[0], cur_batchSize_B, device_A[0], K, &beta, device_C[0], cur_batchSize_B));
				cudaEventRecord(stop_events[0]);  // record end
				cudaEventSynchronize(stop_events[0]);
				cudaEventElapsedTime(&dt, start_events[0], stop_events[0]);
				printf("Partition-database %zu, Batch-sample %zu: Calculation took %8.3f ms.\n", i, j, dt);
				gpu_times[0] += dt;

				// 7.6 transfer result from GPU to RAM
				CHECK_CUDA_ERROR(cudaMemcpy(result_C[0], device_C[0], cur_batchSize_B * cur_batchSize_A * sizeof(float), cudaMemcpyDeviceToHost));

				// 7.7 normalization for output
				#pragma omp parallel for schedule(static)
				for (size_t ai = 0; ai < cur_batchSize_A; ++ai) {
					for (size_t bi = 0; bi < cur_batchSize_B; ++bi) {
						size_t C_index = (j + ai) * N + (i + bi);
						size_t result_index = ai * cur_batchSize_B + bi;
						size_t i = C_index / N;
						size_t j = C_index - i * N;
						KO_Abd[i][j] = result_C[0][result_index];
					}
				}
			}
		}
		total_time += gpu_times[0];
		printf("Calculation took %8.3f ms.\n", gpu_times[0]);
	}
}

void _KO_OTU_Table_All::matrixMultiplyM(float *B, float *A, size_t N, size_t K, size_t M, int coren, int gpu_number) {

	// Automatically get the number of GPUs
	int gpu_count = 0;
	CHECK_CUDA_ERROR(cudaGetDeviceCount(&gpu_count));
	cout << "Found " << gpu_count << " GPUs available." << endl;
	if (gpu_number > 0 && gpu_number <= gpu_count) {
		gpu_count = gpu_number;
	}
	if (gpu_count == 0) {
		cerr << "No GPU devices found!" << endl;
		exit(EXIT_FAILURE);
	}

	// 1. get total GPU memory
	size_t total_mem = 0;
	cudaMemGetInfo(&total_mem, nullptr);

	// 2. calculate needed memory
	size_t required_mem = calculate_memory_size(N, K, M);
	cout << "Required memory: " << required_mem / (1024 * 1024) << " MB" << endl;
	cout << "Available memory: " << total_mem / (1024 * 1024) << " MB" << endl;

	// 3. whether batch partitioning is needed
	size_t batchSize_B, batchSize_A;
	bool need_batching = false;
	size_t element_size = sizeof(float);

	// Check if the size of the database exceeds 70% of the GPU memory
	if (N * K * element_size > total_mem * 0.7) {
		need_batching = true;
		batchSize_B = static_cast<size_t>(total_mem * 0.7 / K / element_size); // Split database into 70% of the GPU memory

		size_t A_C_size = K * M * element_size + N * M * element_size;
		if (A_C_size > total_mem * 0.2) {
			batchSize_A = static_cast<size_t>(total_mem * 0.2 / element_size / (K + N)); // Split sample
		} else {
			batchSize_A = M;  // do not need split sample, calculate directly
		}
	} else {
		size_t B_A_C_size = N * K * element_size + K * M * element_size + N * M * element_size;
		if (B_A_C_size <= total_mem * 0.9) {
			need_batching = false;
			batchSize_B = N;
			batchSize_A = M;  // calculate directly
		} else {
			size_t available_mem_for_A_C = total_mem * 0.9 - (N * K * element_size);
			batchSize_A = static_cast<size_t>(available_mem_for_A_C / (K * element_size + N * element_size));
			batchSize_B = N;
		}
	}

//	size_t batch = (M + gpu_count - 1) / gpu_count;
	size_t batch = M / gpu_count;
	if (batch < 1) batch = 1;
	while (batch > batchSize_A) {
		batch = batch / gpu_count;
		if (batch < 1) batch = 1;
	}
//	cout << batch << endl;
	batchSize_A = batch;
	need_batching = true;

	cout << "Partition-database Size: " << batchSize_B << ", Batch-sample Size: " << batchSize_A << endl;
	cout << "Need batching: " << need_batching << endl;

	// calculate the batch number of sample
	size_t num_batches_A = (M + batchSize_A - 1) / batchSize_A; // round up
	cout << "number of Batch-sample: " << num_batches_A << endl;

	// If the number of sample batches is less than the number of GPUs, allocate memory only for the GPUs actually in use
	int max_gpus = min(gpu_count, static_cast<int>(num_batches_A));
	cout << "use " << max_gpus << " GPUs." << endl;

	vector<float*> device_B(max_gpus), device_A(max_gpus), device_C(max_gpus);
	vector<float*> result_C(max_gpus); // allocate result for all GPUs
	vector<cublasHandle_t> handles(max_gpus);  // allocate cublasHandle_t for all GPUs

	// === events and time ===
	vector<cudaEvent_t> start_events(max_gpus);
	vector<cudaEvent_t> stop_events(max_gpus);
	vector<float> gpu_times(max_gpus, 0.0f);

	// allocate memory and cublasHandle_t for all GPUs
	#pragma omp parallel for num_threads(max_gpus)
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_CUDA_ERROR(cudaSetDevice(i));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&device_B[i], batchSize_B * K * sizeof(float)));  // database
		CHECK_CUDA_ERROR(cudaMalloc((void**)&device_A[i], K * batchSize_A * sizeof(float)));  // sample
		CHECK_CUDA_ERROR(cudaMalloc((void**)&device_C[i], batchSize_B * batchSize_A * sizeof(float)));  // result
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&result_C[i], batchSize_B * batchSize_A * sizeof(float), cudaHostAllocDefault));

		// create cublas handle
		CHECK_CUBLAS_ERROR(cublasCreate(&handles[i]));

		// create cuda event
		CHECK_CUDA_ERROR(cudaEventCreate(&start_events[i]));
		CHECK_CUDA_ERROR(cudaEventCreate(&stop_events[i]));
	}

	int actual_coren = coren/max_gpus - max_gpus;
	cout << "actual_coren = " << actual_coren << endl;

	// 7 calculation
	for (size_t i = 0; i < N; i += batchSize_B) {
		// calculate current Partition size
		size_t cur_batchSize_B = min(batchSize_B, N - i);

		// transfer current Partiton from RAM to GPU
		if (batchSize_B >= N ) {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_CUDA_ERROR(cudaSetDevice(gpu_idx));
				CHECK_CUDA_ERROR(cudaMemcpy(device_B[gpu_idx], B, batchSize_B * K * sizeof(float), cudaMemcpyHostToDevice));
			}
		} else {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_CUDA_ERROR(cudaSetDevice(gpu_idx));
				#pragma omp parallel for num_threads(actual_coren) schedule(static)
				for (size_t k = 0; k < K; ++k) {
					CHECK_CUDA_ERROR(cudaMemcpy(device_B[gpu_idx] + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), cudaMemcpyHostToDevice));
				}
			}
		}

		#pragma omp parallel for num_threads(max_gpus)
		for (size_t j = 0; j < M; j += batchSize_A) {
			int gpu_idx = omp_get_thread_num();
			CHECK_CUDA_ERROR(cudaSetDevice(gpu_idx));

			// calculate curren Batch size of sample
			size_t cur_batchSize_A = min(batchSize_A, M - j);

			// transfer current Batch from RAM to GPU
			CHECK_CUDA_ERROR(cudaMemcpy(device_A[gpu_idx], A + j * K, K * cur_batchSize_A * sizeof(float), cudaMemcpyHostToDevice));

			// init result to 0
			CHECK_CUDA_ERROR(cudaMemset(device_C[gpu_idx], 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

			// record start
			CHECK_CUDA_ERROR(cudaEventRecord(start_events[gpu_idx], 0));

			// calculate
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_CUBLAS_ERROR(cublasSgemm(handles[gpu_idx], CUBLAS_OP_N, CUBLAS_OP_N,
			                               cur_batchSize_B, cur_batchSize_A, K, &alpha,
			                               device_B[gpu_idx], cur_batchSize_B,
			                               device_A[gpu_idx], K, &beta,
			                               device_C[gpu_idx], cur_batchSize_B));

			// record end
			CHECK_CUDA_ERROR(cudaEventRecord(stop_events[gpu_idx], 0));
			CHECK_CUDA_ERROR(cudaEventSynchronize(stop_events[gpu_idx]));

			float milliseconds = 0.0f;
			CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_events[gpu_idx], stop_events[gpu_idx]));
			gpu_times[gpu_idx] += milliseconds;

			// transfer result from GPU to RAM
			CHECK_CUDA_ERROR(cudaMemcpy(result_C[gpu_idx], device_C[gpu_idx], cur_batchSize_B * cur_batchSize_A * sizeof(float), cudaMemcpyDeviceToHost));

			// normalization for out put
			#pragma omp parallel for num_threads(actual_coren) schedule(static)
			for (size_t ai = 0; ai < cur_batchSize_A; ++ai) {
				for (size_t bi = 0; bi < cur_batchSize_B; ++bi) {
					size_t C_index = (j + ai) * N + (i + bi);
					size_t result_index = ai * cur_batchSize_B + bi;
					size_t i = C_index / N;
					size_t j = C_index - i * N;
					KO_Abd[i][j] = result_C[gpu_idx][result_index];
				}
			}
		}
	}
	// release resource
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_CUDA_ERROR(cudaFreeHost(result_C[i]));
	}

	// === output GPU computation time ===
	int gpu_index = 0;
	for (auto t : gpu_times) {
		cout << "GPU "<< gpu_index++ << " computation time: " << t << " ms" << endl;
	}

	for (auto t : gpu_times) total_time += t;
	cout << "Total GPU computation time: " << total_time << " ms" << endl;

	// ÇåÀíÄÚ´æ
	#pragma omp parallel for num_threads(max_gpus)
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_CUDA_ERROR(cudaSetDevice(i));
		CHECK_CUBLAS_ERROR(cublasDestroy(handles[i]));
		CHECK_CUDA_ERROR(cudaFree(device_B[i]));
		CHECK_CUDA_ERROR(cudaFree(device_A[i]));
		CHECK_CUDA_ERROR(cudaFree(device_C[i]));

		CHECK_CUDA_ERROR(cudaEventDestroy(start_events[i]));
		CHECK_CUDA_ERROR(cudaEventDestroy(stop_events[i]));
	}
}

void _KO_OTU_Table_All::matrixMultiplyM_more(float *B, float *A, size_t N, size_t K, size_t M, int coren) {

	float milliseconds = 0.0f;
	gpu_times.assign(max_gpus, 0.0f);

	// 7 calculation
	for (size_t i = 0; i < N; i += batchSize_B) {
		// calculate curreent Partition size of database
		size_t cur_batchSize_B = min(batchSize_B, N - i);

		// transfer current Partiton from RAM to GPU
		if (batchSize_B >= N ) {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_CUDA_ERROR(cudaSetDevice(gpu_idx));
				CHECK_CUDA_ERROR(cudaMemcpy(device_B[gpu_idx], B, batchSize_B * K * sizeof(float), cudaMemcpyHostToDevice));
			}
		} else {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_CUDA_ERROR(cudaSetDevice(gpu_idx));
				#pragma omp parallel for num_threads(actual_coren) schedule(static)
				for (size_t k = 0; k < K; ++k) {
					CHECK_CUDA_ERROR(cudaMemcpy(device_B[gpu_idx] + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), cudaMemcpyHostToDevice));
				}
			}
		}

		#pragma omp parallel for num_threads(max_gpus)
		for (size_t j = 0; j < M; j += batchSize_A) {
			int gpu_idx = omp_get_thread_num();
			CHECK_CUDA_ERROR(cudaSetDevice(gpu_idx));

			// calculate current Batch size of sample
			size_t cur_batchSize_A = min(batchSize_A, M - j);

			// transfer current Batch from RAM to GPU
			CHECK_CUDA_ERROR(cudaMemcpy(device_A[gpu_idx], A + j * K, K * cur_batchSize_A * sizeof(float), cudaMemcpyHostToDevice));

			// init result to 0
			CHECK_CUDA_ERROR(cudaMemset(device_C[gpu_idx], 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

			// record start
			CHECK_CUDA_ERROR(cudaEventRecord(start_events[gpu_idx], 0));

			// calculation
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_CUBLAS_ERROR(cublasSgemm(handles[gpu_idx], CUBLAS_OP_N, CUBLAS_OP_N,
			                               cur_batchSize_B, cur_batchSize_A, K, &alpha,
			                               device_B[gpu_idx], cur_batchSize_B,
			                               device_A[gpu_idx], K, &beta,
			                               device_C[gpu_idx], cur_batchSize_B));

			// record end
			CHECK_CUDA_ERROR(cudaEventRecord(stop_events[gpu_idx], 0));
			CHECK_CUDA_ERROR(cudaEventSynchronize(stop_events[gpu_idx]));

			CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_events[gpu_idx], stop_events[gpu_idx]));
			gpu_times[gpu_idx] += milliseconds;

			// transfer result from GPU to RAM
			CHECK_CUDA_ERROR(cudaMemcpy(result_C[gpu_idx], device_C[gpu_idx], cur_batchSize_B * cur_batchSize_A * sizeof(float), cudaMemcpyDeviceToHost));

			// normalization for output
			#pragma omp parallel for num_threads(actual_coren) schedule(static)
			for (size_t ai = 0; ai < cur_batchSize_A; ++ai) {
				for (size_t bi = 0; bi < cur_batchSize_B; ++bi) {
					size_t C_index = (j + ai) * N + (i + bi);
					size_t result_index = ai * cur_batchSize_B + bi;
					size_t i = C_index / N;
					size_t j = C_index - i * N;
					KO_Abd[i][j] = result_C[gpu_idx][result_index];
				}
			}
		}
	}

	// === output GPU computation time ===
	int gpu_index = 0;
	for (auto t : gpu_times) {
		cout << "GPU "<< gpu_index++ << " computation time: " << t << " ms" << endl;
	}

	for (auto t : gpu_times) total_time += t;
}

int _KO_OTU_Table_All::Load_Sample_By_OTU_Table_G(_Table_Format * table, int coren, bool skipNormalization, double max_nsti, string gpuMode, int gpu_number) {

	size_t N = KO_Index.size();            // Number of rows in database
	size_t K = table->Get_Feature_Size(); // Number of columns in database, also number of rows in sample
	size_t M = table->Get_Sample_Size();  // Number of columns in sample

	// allocate and int database
	float *B = nullptr;
//	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&B, N * K * sizeof(float), cudaHostAllocDefault));
	B = (float*)malloc(N * K * sizeof(float));
	// check and init database to 0
	if (B != nullptr) {
		printf("database Memory allocation successful!\n");
		memset(B, 0, N * K * sizeof(float));
	} else {
		fprintf(stderr, "database Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	float *A = table->GetA();

	vector<string> otus = table->Get_Feature_Names();
	vector<string> sam_names = table->Get_Sample_Names();

	omp_set_num_threads(coren);
	// normalization for sample
	if (!skipNormalization) {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < M; ++i) {
			Sample_names[i] = sam_names[i];
			for (int j = 0; j < K; ++j) {
				string a_otu_j = Check_OTU(otus[j]);
				float rna_cp = Otu_parser.Get_cp_by_OTU(a_otu_j);  // get RNA copy number
				A[i * K + j] = A[i * K + j] / rna_cp;
			}
		}
	} else {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < M; ++i) {
			Sample_names[i] = sam_names[i];
		}
	}

	// normalization for database
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < K; ++i) {
		string a_otu_j = Check_OTU(otus[i]);

		// Use find() to avoid implicit write with operator[], ensuring thread safety
		auto it = OTU_KO_Index.find(a_otu_j);
		if (it != OTU_KO_Index.end()) {
			vector<_KO_Index_Copy>& kos = it->second;
			double nsti = Otu_parser.Get_nsti_by_OTU(a_otu_j);
			if (nsti >= 0 && nsti <= max_nsti) {
				for (int j = 0; j < kos.size(); ++j) {
					B[i * N + kos[j].Index] = kos[j].Copy;
				}
			}
		} else {
			// Skip if OTU does not have an index
			continue;
		}
	}

	// calculation
	if (gpuMode == "S")
		matrixMultiplyS(B, A, N, K, M, coren);
	else
		matrixMultiplyM(B, A, N, K, M, coren, gpu_number);

	// release resource
//	CHECK_CUDA_ERROR(cudaFreeHost(B));
	free(B);
	B = nullptr;

	return table->Get_Sample_Size();
}

int _KO_OTU_Table_All::Load_Sample_By_OTU_Table_newG(_Table_Format * table, int coren, bool skipNormalization, string gpuMode, int num, bool over, int gpu_number) {

	size_t N = KO_Index.size();            // Number of rows in database
	size_t K = table->Get_Feature_Size(); // Number of columns in database, also number of rows in sample
	size_t M = table->Get_Sample_Size();  // Number of columns in sample


	Sample_count = M;

	vector<string> otus = table->Get_Feature_Names();
	vector<string> sam_names = table->Get_Sample_Names();

	float *A = table->GetA();

	omp_set_num_threads(coren);
	// normalization for sample
	if (!skipNormalization) {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < M; ++i) {
			Sample_names[i] = sam_names[i];
			for (int j = 0; j < K; ++j) {
				string a_otu_j = Check_OTU(otus[j]);
				float rna_cp = Otu_parser.Get_cp_by_OTU(a_otu_j);  // get RNA copy number
				A[i * K + j] = A[i * K + j] / rna_cp;
			}
		}
	} else {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < M; ++i) {
			Sample_names[i] = sam_names[i];
		}
	}


	float *B_ptr = B_host;

	if(num == 0) {
		if (gpuMode == "S")
			make_batch_S(table, N, K, M);
		else
			make_batch_M(table, N, K, M, coren, gpu_number);
	}

	// calculation
	if (gpuMode == "S")
		matrixMultiplyS_more(B_ptr, A, N, K, M, coren);
	else
		matrixMultiplyM_more(B_ptr, A, N, K, M, coren);

	cout << "---------------------------------over:" << over << endl;
	// release resource
	if (over) {
		printf("Calculation took %8.3f ms(total).\n", total_time);
		releaseMemory();
//		CHECK_CUDA_ERROR(cudaFreeHost(B_host));
		free(B_host);
		B_host = nullptr;
	}

	return table->Get_Sample_Size();
}

int _KO_OTU_Table_All::Output_By_Category(const char * outfilename, int level, float max, float min) { //2 output, abd and count


	hash_map <string, vector<string>, std_string_hash> pw_ko;
	hash_map <string, int, std_string_hash> pw_index;

	// open output
	//Abd file
	string outfilename_abd = outfilename;
	outfilename_abd += ".Abd";
	ofstream outfile_abd(outfilename_abd.c_str(), ofstream::out);
	if (!outfile_abd) {
		cerr << "Error: Cannot open output file : " << outfilename_abd << endl;
		return 0;
	}
	//Count file
	string outfilename_count = outfilename;
	outfilename_count += ".Count";
	ofstream outfile_count(outfilename_count.c_str(), ofstream::out);
	if (!outfile_count) {
		cerr << "Error: Cannot open output file : " << outfilename_count << endl;
		return 0;
	}

	//add_ko
	for (int i = 0; i < KOs.size(); i ++) {

		if (level >= 3) {

			pw_ko[KOs[i].Get_Name()].push_back(KOs[i].Get_Name());
			continue;

		}

		string pw = KOs[i].Get_Pathway();
		int l = 0;
		int begin = 0;
		int end = 0;
		while(end <= pw.size()) {

			if ((pw[end] == ';') || (end == pw.size())) {
				if (l == level) {
					string pw_name = pw.substr(begin, end - begin);
					pw_ko[pw_name].push_back(KOs[i].Get_Name());
				}
				begin = end + 1;
				l ++;
			} else if (pw[end] == '|') {
				l = 0;
				begin = end + 1;
			} else if ((pw[end] == ' ') || (pw[end] == '\''))
				pw[end] = '_';

			end ++;

		}
	}

	//make pw index
	int index = 0;
	for (hash_map <string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); miter ++) {
		pw_index[miter->first] = index;
		index ++;
	}

	//calc the pathway sum
	float ** pw_count = new float * [Sample_count];
	float ** pw_abd = new float * [Sample_count];

	for (int i = 0; i < Sample_count; i ++) {
		pw_count[i] = new float [pw_ko.size()];
		pw_abd[i] = new float [pw_ko.size()];

		for (int j = 0; j < pw_ko.size(); j ++) {
			pw_count[i][j] = 0;
			pw_abd[i][j] = 0;
		}

	}

	//count file
	for (int i = 0; i < Sample_count; i ++) {
		for (hash_map <string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); miter ++) {
			for (int j = 0; j < (miter->second).size(); j ++)
				pw_count[i][pw_index[miter->first]] += KO_Abd[i][KO_Index[(miter->second)[j]]];
		}
	}

	//abd file
	//norm for abd
	for (int i = 0; i < Sample_count; i ++) {
		float sum = 0;
		for (hash_map <string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); miter ++)
			sum += pw_count[i][pw_index[miter->first]];
		for (hash_map <string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); miter ++)
			if (sum > 0)
				pw_abd[i][pw_index[miter->first]] = pw_count[i][pw_index[miter->first]] / sum;
			else pw_abd[i][pw_index[miter->first]] = 0;
	}

	//check no zero
	bool * No_Zero_Check = new bool[pw_ko.size()];
	int count = 0;
	for (hash_map <string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); miter ++) {
		index = pw_index[miter->first];
		No_Zero_Check[index] = false;

		for (int i = 0; i < Sample_count; i ++)
			if (pw_abd[i][index] > 0)
				No_Zero_Check[index] = true;

		if (No_Zero_Check[index]) count ++;
	}

	//output
	outfile_abd << "Sample";
	outfile_count << "Sample";

	for (hash_map <string, vector<string>,std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); miter ++) {
		index = pw_index[miter->first];
		if (!No_Zero_Check[index]) continue;
		outfile_abd << "\t" << miter->first;
		outfile_count << "\t" << miter->first;
	}
	outfile_abd << endl;
	outfile_count << endl;

	for (int i = 0; i < Sample_count; i ++) {

		outfile_abd << Sample_names[i];
		outfile_count << Sample_names[i];

		for (hash_map <string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); miter ++) {

			index = pw_index[miter->first];
			if (!No_Zero_Check[index]) continue;

			outfile_count << "\t" << pw_count[i][index];

			outfile_abd << "\t" << pw_abd[i][index];

		}

		outfile_abd << endl;
		outfile_count << endl;
	}

	outfile_abd.close();
	outfile_abd.clear();

	outfile_count.close();
	outfile_count.clear();

	return count;
}

int _KO_OTU_Table_All::Output_By_Category_openMP(const char * outfilename, int level, float max, float min, int coren) { //2 output, abd and count
	hash_map <string, vector<string>, std_string_hash> pw_ko;
	hash_map <string, int, std_string_hash> pw_index;

	// open output file
	string outfilename_abd = outfilename;
	outfilename_abd += ".Abd";
	ofstream outfile_abd(outfilename_abd.c_str(), ofstream::out);
	if (!outfile_abd) {
		cerr << "Error: Cannot open output file : " << outfilename_abd << endl;
		return 0;
	}

	string outfilename_count = outfilename;
	outfilename_count += ".Count";
	ofstream outfile_count(outfilename_count.c_str(), ofstream::out);
	if (!outfile_count) {
		cerr << "Error: Cannot open output file : " << outfilename_count << endl;
		return 0;
	}

	// add KO
	for (int i = 0; i < KOs.size(); i++) {
		if (level >= 3) {
			pw_ko[KOs[i].Get_Name()].push_back(KOs[i].Get_Name());
			continue;
		}
		string pw = KOs[i].Get_Pathway();
		int l = 0;
		int begin = 0;
		int end = 0;
		while (end <= pw.size()) {
			if ((pw[end] == ';') || (end == pw.size())) {
				if (l == level) {
					string pw_name = pw.substr(begin, end - begin);
					pw_ko[pw_name].push_back(KOs[i].Get_Name());
				}
				begin = end + 1;
				l++;
			} else if (pw[end] == '|') {
				l = 0;
				begin = end + 1;
			} else if ((pw[end] == ' ') || (pw[end] == '\'')) {
				pw[end] = '_';
			}
			end++;
		}
	}

	// make pw index
	int index = 0;
	for (hash_map<string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
		pw_index[miter->first] = index++;
	}

	vector<vector<float>> pw_count(Sample_count, vector<float>(pw_ko.size(), 0));
	vector<vector<float>> pw_abd(Sample_count, vector<float>(pw_ko.size(), 0));

	omp_set_num_threads(coren);
	// count
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < Sample_count; i ++) {
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			for (size_t j = 0; j < miter->second.size(); j++) {
				pw_count[i][pw_index[miter->first]] += KO_Abd[i][KO_Index[miter->second[j]]];
			}
		}
	}

	// normalization
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < Sample_count; i++) {
		float sum = 0;
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			sum += pw_count[i][pw_index[miter->first]];
		}
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			if (sum > 0) {
				pw_abd[i][pw_index[miter->first]] = pw_count[i][pw_index[miter->first]] / sum;
			} else {
				pw_abd[i][pw_index[miter->first]] = 0;
			}
		}
	}

	//check no zero
	bool * No_Zero_Check = new bool[KO_Index.size()];
	#pragma omp parallel for schedule(dynamic, 1)
	for (int j = 0; j < KO_Index.size(); j ++) {
		No_Zero_Check[j] = false;
		for (int i = 0; i < Sample_count; i ++)
			if (pw_abd[i][j] > 0) {
				No_Zero_Check[j] = true;
				break;
			}
	}

	int count = 0;
	for (hash_map <string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
		int index = pw_index[miter->first];
		if (No_Zero_Check[index]) count ++;
	}

	// output file header
	outfile_abd << "Sample";
	outfile_count << "Sample";
	for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
		index = pw_index[miter->first];
		if (!No_Zero_Check[index]) continue;
		outfile_abd << "\t" << miter->first;
		outfile_count << "\t" << miter->first;
	}
	outfile_abd << endl;
	outfile_count << endl;

	for (int i = 0; i < Sample_count; i++) {
		ostringstream abd_buffer;
		ostringstream count_buffer;

		abd_buffer << Sample_names[i];
		count_buffer << Sample_names[i];

		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			index = pw_index[miter->first];
			if (!No_Zero_Check[index]) continue;
			count_buffer << "\t" << pw_count[i][index];
			abd_buffer << "\t" << pw_abd[i][index];
		}
		abd_buffer << endl;
		count_buffer << endl;

		// Flush the buffered content to file
		outfile_abd << abd_buffer.str();
		outfile_count << count_buffer.str();
	}

	// close file
	outfile_abd.close();
	outfile_abd.clear();
	outfile_count.close();
	outfile_count.clear();

	delete[] No_Zero_Check;

	return count;
}

int _KO_OTU_Table_All::Output_By_Category_app_openMP(const char * outfilename, int level, float max, float min, int coren, int num) { //2 output, abd and count
	hash_map <string, vector<string>, std_string_hash> pw_ko;
	hash_map <string, int, std_string_hash> pw_index;

	// open output file
	string outfilename_abd = outfilename;
	outfilename_abd += ".Abd";
	ofstream outfile_abd(outfilename_abd.c_str(), ofstream::out | ofstream::app);
	if (!outfile_abd) {
		cerr << "Error: Cannot open output file : " << outfilename_abd << endl;
		return 0;
	}

	string outfilename_count = outfilename;
	outfilename_count += ".Count";
	ofstream outfile_count(outfilename_count.c_str(), ofstream::out | ofstream::app);
	if (!outfile_count) {
		cerr << "Error: Cannot open output file : " << outfilename_count << endl;
		return 0;
	}

	// add KO
	for (int i = 0; i < KOs.size(); i++) {
		if (level >= 3) {
			pw_ko[KOs[i].Get_Name()].push_back(KOs[i].Get_Name());
			continue;
		}
		string pw = KOs[i].Get_Pathway();
		int l = 0;
		int begin = 0;
		int end = 0;
		while (end <= pw.size()) {
			if ((pw[end] == ';') || (end == pw.size())) {
				if (l == level) {
					string pw_name = pw.substr(begin, end - begin);
					pw_ko[pw_name].push_back(KOs[i].Get_Name());
				}
				begin = end + 1;
				l++;
			} else if (pw[end] == '|') {
				l = 0;
				begin = end + 1;
			} else if ((pw[end] == ' ') || (pw[end] == '\'')) {
				pw[end] = '_';
			}
			end++;
		}
	}

	// make pw index
	int index = 0;
	for (hash_map<string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
		pw_index[miter->first] = index++;
	}

	vector<vector<float>> pw_count(Sample_count, vector<float>(pw_ko.size(), 0));
	vector<vector<float>> pw_abd(Sample_count, vector<float>(pw_ko.size(), 0));

	omp_set_num_threads(coren);
	// count
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < Sample_count; i ++) {
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			for (size_t j = 0; j < miter->second.size(); j++) {
				pw_count[i][pw_index[miter->first]] += KO_Abd[i][KO_Index[miter->second[j]]];
			}
		}
	}

	// normalization
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < Sample_count; i++) {
		float sum = 0;
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			sum += pw_count[i][pw_index[miter->first]];
		}
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			if (sum > 0) {
				pw_abd[i][pw_index[miter->first]] = pw_count[i][pw_index[miter->first]] / sum;
			} else {
				pw_abd[i][pw_index[miter->first]] = 0;
			}
		}
	}

	// output file header
	if(num == 0) {
		outfile_abd << "Sample";
		outfile_count << "Sample";
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			index = pw_index[miter->first];
			outfile_abd << "\t" << miter->first;
			outfile_count << "\t" << miter->first;
		}
		outfile_abd << endl;
		outfile_count << endl;
	}

	for (int i = 0; i < Sample_count; i++) {
		ostringstream abd_buffer;
		ostringstream count_buffer;

		abd_buffer << Sample_names[i];
		count_buffer << Sample_names[i];

		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			index = pw_index[miter->first];
			count_buffer << "\t" << pw_count[i][index];
			abd_buffer << "\t" << pw_abd[i][index];
		}
		abd_buffer << endl;
		count_buffer << endl;

		// Flush the buffered content to file
		outfile_abd << abd_buffer.str();
		outfile_count << count_buffer.str();
	}

	// close file
	outfile_abd.close();
	outfile_abd.clear();
	outfile_count.close();
	outfile_count.clear();

	return KOs.size();
}

int _KO_OTU_Table_All::Output_By_KO_openMP(const char * outfilename, int level, float max, float min, int coren) { //2 output, abd and count

	// open output
	//Abd file
	string outfilename_abd = outfilename;
	outfilename_abd += ".Abd";
	ofstream outfile_abd(outfilename_abd.c_str(), ofstream::out);
	if (!outfile_abd) {
		cerr << "Error: Cannot open output file : " << outfilename_abd << endl;
		return 0;
	}

	//Count file
	string outfilename_count = outfilename;
	outfilename_count += ".Count";
	ofstream outfile_count(outfilename_count.c_str(), ofstream::out);
	if (!outfile_count) {
		cerr << "Error: Cannot open output file : " << outfilename_count << endl;
		return 0;
	}

	int index = 0;

	vector<vector<float>> ko_abd(Sample_count, vector<float>(KOs.size(), 0));

	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < Sample_count; i ++) {
		float sum = 0;
		for (hash_map <string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter)
			sum += KO_Abd[i][KO_Index[miter->first]];
		for (hash_map <string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter)
			if (sum > 0)
				ko_abd[i][KO_Index[miter->first]] = KO_Abd[i][KO_Index[miter->first]] / sum;
			else
				ko_abd[i][KO_Index[miter->first]] = 0;
	}

	//check no zero
	bool * No_Zero_Check = new bool[KO_Index.size()];
	#pragma omp parallel for schedule(dynamic, 1)
	for (int j = 0; j < KO_Index.size(); j ++) {
		No_Zero_Check[j] = false;
		for (int i = 0; i < Sample_count; i ++)
			if (ko_abd[i][j] > 0) {
				No_Zero_Check[j] = true;
				break;
			}
	}

	int count = 0;
	for (hash_map <string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {
		int index = KO_Index[miter->first];
		if (No_Zero_Check[index]) count++;
	}

	//output
	outfile_abd << "Sample";
	outfile_count << "Sample";

	for (hash_map <string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {
		index = KO_Index[miter->first];
		if (!No_Zero_Check[index]) continue;
		outfile_abd << "\t" << miter->first;
		outfile_count << "\t" << miter->first;
	}

	outfile_abd << endl;
	outfile_count << endl;

	for (int i = 0; i < Sample_count; i ++) {
		ostringstream abd_buffer, count_buffer; // Used for buffering output

		outfile_abd << Sample_names[i];
		outfile_count << Sample_names[i];

		for (hash_map <string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {

			index = KO_Index[miter->first];
			if (!No_Zero_Check[index]) continue;

			count_buffer << "\t" << KO_Abd[i][index];
			abd_buffer << "\t" << ko_abd[i][index];

		}
		// Write to file in a single operation
		outfile_abd << abd_buffer.str() << endl;
		outfile_count << count_buffer.str() << endl;
	}

	outfile_abd.close();
	outfile_abd.clear();

	outfile_count.close();
	outfile_count.clear();

	delete[] No_Zero_Check;

	return count;
}

int _KO_OTU_Table_All::Output_By_KO_app_openMP(const char * outfilename, int level, float max, float min, int coren, int num) { //2 output, abd and count
	// output file name
	std::string outfilename_abd = std::string(outfilename) + ".Abd";
	std::string outfilename_count = std::string(outfilename) + ".Count";

	// open Abd file
	ofstream outfile_abd;
	if (num == 0)
		outfile_abd.open(outfilename_abd.c_str(), ofstream::out | ofstream::trunc);  // new and clean
	else
		outfile_abd.open(outfilename_abd.c_str(), ofstream::out | ofstream::app);    // add

	if (!outfile_abd) {
		cerr << "Error: Cannot open output file: " << outfilename_abd << endl;
		return 0;
	}

	// open Count file
	ofstream outfile_count;
	if (num == 0)
		outfile_count.open(outfilename_count.c_str(), ofstream::out | ofstream::trunc);
	else
		outfile_count.open(outfilename_count.c_str(), ofstream::out | ofstream::app);

	if (!outfile_count) {
		cerr << "Error: Cannot open output file: " << outfilename_count << endl;
		return 0;
	}

	vector<vector<float>> ko_abd(Sample_count, vector<float>(KOs.size(), 0));

	omp_set_num_threads(coren);
	// normalization
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < Sample_count; i++) {
		float sum = 0;
		for (hash_map<string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {
			sum += KO_Abd[i][miter->second];
		}
		for (hash_map<string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {
			if (sum > 0) {
				ko_abd[i][miter->second] = KO_Abd[i][miter->second] / sum;
			} else {
				ko_abd[i][miter->second] = 0;
			}
		}
	}

	// output file header
	if(num == 0) {
		outfile_abd << "Sample";
		outfile_count << "Sample";
		int index = 0;
		for (hash_map <string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {
			index = miter->second;
			outfile_abd << "\t" << miter->first;
			outfile_count << "\t" << miter->first;
		}
		outfile_abd << endl;
		outfile_count << endl;
	}

	for (int i = 0; i < Sample_count; i++) {
		ostringstream abd_buffer;
		ostringstream count_buffer;

		abd_buffer << Sample_names[i];
		count_buffer << Sample_names[i];

		int index = 0;

		for (hash_map<string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {
			index = miter->second;
			count_buffer << "\t" << KO_Abd[i][index];
			abd_buffer << "\t" << ko_abd[i][index];
		}
		abd_buffer << endl;
		count_buffer << endl;

		// Flush the buffered content to file
		outfile_abd << abd_buffer.str();
		outfile_count << count_buffer.str();
	}

	// close file
	outfile_abd.close();
	outfile_abd.clear();
	outfile_count.close();
	outfile_count.clear();

	return KOs.size();
}

int _KO_OTU_Table_All::Load_Sample_By_OTU(hash_map <string, int, std_string_hash> * otu_seq_count, int sample, bool skipNormalization, double max_nsti) {


	for (hash_map <string, int, std_string_hash> ::iterator miter = otu_seq_count->begin(); miter != otu_seq_count->end(); ++miter) {

		vector<_KO_Index_Copy> kos = OTU_KO_Index[miter->first];
		int seq_count = miter->second;

		float rna_cp = skipNormalization ? 1.0f : Otu_parser.Get_cp_by_OTU(miter->first);
		double nsti = Otu_parser.Get_nsti_by_OTU(miter->first);

		if(nsti >=0 && nsti<= max_nsti) {
			for (int i = 0; i < kos.size(); i ++) {
				int index = kos[i].Index;
				float copy = kos[i].Copy;
				KO_Abd[sample][index] += (float) seq_count * copy / rna_cp;
			}
		} else {
			continue;
		}
	}

	return otu_seq_count->size();
}
#endif /* class_func_h */

