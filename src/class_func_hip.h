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
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hipblas.h>
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
		int Load_Sample_Multi(vector<string> infilename, vector <string> sample_name, int coren, bool skipNormalization, double max_nsti);
		int Load_Sample_By_OTU_Table(_Table_Format * table, int coren, bool skipNormalization, double max_nsti);
		int Load_Sample_By_OTU_Table_G(_Table_Format * table, int coren, bool skipNormalization, double max_nsti, string gpuMode, int gpu_number);
		int Load_Sample_By_OTU_Table_newG(_Table_Format * table, int coren, bool skipNormalization, string gpuMode, int num, bool over, int gpu_number);
		int Load_Sample_By_Single_KO_Table(const char * infilename, string sample_name, int sample);
		int Load_Sample_By_KO_Table(_Table_Format * table);
		int Output(const char * outfilename);
		int Output_Multi(vector <string> outfilename);
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

		float *B;

		vector<float*> device_B;
		vector<float*> device_A;
		vector<float*> device_C;
		vector<float*> result_C;
		vector<hipblasHandle_t> handles;

		// === 新增：定义事件和时间统计变量 ===
		vector<hipEvent_t> start_events;
		vector<hipEvent_t> stop_events;
		vector<float> gpu_times;  // 各 GPU 累计时间

		// B, A batch size
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
	// If it's the first write, delete the old file and create a new one
	if (num == 0) {
		ofstream outfile(outfilename, ios::out | ios::trunc);
		if (!outfile) {
			cerr << "Error: Cannot open output file: " << outfilename << endl;
			return;
		}
		outfile << "sample\tweighted_NSTI" << endl;
		outfile.close();
	}

	// Afterwards, write using append mode uniformly
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

#define CHECK_HIP_ERROR(err) if (err != hipSuccess) { \
		std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE); \
	}

#define CHECK_HIPBLAS_ERROR(err) if (err != HIPBLAS_STATUS_SUCCESS) { \
		std::cerr << "HIPBLAS Error" << std::endl; \
		exit(EXIT_FAILURE); \
	}

// calculate needed GPU memory
size_t calculate_memory_size(size_t N, size_t K, size_t M) {
	size_t element_size = sizeof(float);
	size_t size_B = N * K * element_size;  // B 的大小
	size_t size_A = K * M * element_size;  // A 的大小
	size_t size_C = N * M * element_size;  // C 的大小
	return size_B + size_A + size_C;       // 总内存大小
}

void _KO_OTU_Table_All::make_batch_S(_Table_Format * table, size_t N, size_t K, size_t M) {
	// 1. Get the total GPU memory size
	size_t total_mem = 0;
	hipMemGetInfo(&total_mem, nullptr);

	// 2. Calculate the total required GPU memory size
	size_t required_mem = calculate_memory_size(N, K, M);
	cout << "Required memory: " << required_mem / (1024 * 1024) << " MB" << endl;
	cout << "Available memory: " << total_mem / (1024 * 1024) << " MB" << endl;

	// 3. Determine whether split processing is needed
	need_batching = false;

	size_t element_size = sizeof(float);

	// Check if the size of database exceeds 70% of the GPU memory
	if (N * K * element_size > total_mem * 0.7) {
		need_batching = true;
		batchSize_B = static_cast<size_t>(total_mem * 0.7 / K / element_size); // Split database to occupy 70% of the GPU memory

		size_t A_C_size = K * M * element_size + N * M * element_size;
		if (A_C_size > total_mem * 0.2) {
			batchSize_A = static_cast<size_t>(total_mem * 0.2 / element_size / (K + N)); // split sample
		} else {
			batchSize_A = M;  // do not need split sample
		}
	} else {
		size_t B_A_C_size = N * K * element_size + K * M * element_size + N * M * element_size;
		if (B_A_C_size <= total_mem * 0.9) {
			batchSize_B = N;
			batchSize_A = M;  // Compute directly without any spliting
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
	CHECK_HIP_ERROR(hipMalloc((void**)&device_B[0], batchSize_B * K * sizeof(float))); // database
	CHECK_HIP_ERROR(hipMalloc((void**)&device_A[0], K * batchSize_A * sizeof(float))); // samples
	CHECK_HIP_ERROR(hipMalloc((void**)&device_C[0], batchSize_B * batchSize_A * sizeof(float))); // results
	CHECK_HIPBLAS_ERROR(hipblasCreate(&handles[0]));

	// create hip events
	hipEventCreate(&start_events[0]);
	hipEventCreate(&stop_events[0]);

	if (!need_batching) {
		// allocate RAM for results
		CHECK_HIP_ERROR(hipHostMalloc((void**)&result_C[0], N * M * sizeof(float)));
		// check
		if (result_C[0] != nullptr) {
			printf("result Memory allocation successful!\n");
		} else {
			fprintf(stderr, "result Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		// Transfer the database data from RAM to GPU
		CHECK_HIP_ERROR(hipMemcpy(device_B[0], B, batchSize_B * K * sizeof(float), hipMemcpyHostToDevice));
	} else {
		// need split
		// Allocate memory for storing the resulting submatrix results
		CHECK_HIP_ERROR(hipHostMalloc((void**)&result_C[0], batchSize_B * batchSize_A * sizeof(float)));
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
		// release memory for single GPU
		hipFree(device_A[0]);
		hipFree(device_B[0]);
		hipFree(device_C[0]);
		hipHostFree(result_C[0]);
		hipblasDestroy(handles[0]);

		hipEventDestroy(start_events[0]);
		hipEventDestroy(stop_events[0]);
	} else {
		// release memory for multiple GPU
		for (int i = 0; i < max_gpus; ++i) {
			hipSetDevice(i);
			hipFree(device_A[i]);
			hipFree(device_B[i]);
			hipFree(device_C[i]);
			hipHostFree(result_C[i]);
			hipblasDestroy(handles[i]);

			hipEventDestroy(start_events[i]);
			hipEventDestroy(stop_events[i]);
		}
	}
}

void _KO_OTU_Table_All::make_batch_M(_Table_Format * table, size_t N, size_t K, size_t M, int coren, int gpu_number) {
//	// Automatically get the number of GPUs
//	int gpu_count = 0;
//	CHECK_HIP_ERROR(hipGetDeviceCount(&gpu_count));
//	if (gpu_count == 0) {
//		cerr << "No GPU devices found!" << endl;
//		exit(EXIT_FAILURE);
//	}
//	cout << "Found " << gpu_count << " GPUs available." << endl;

	// Automatically get the number of GPUs
	int gpu_count = 0;
	CHECK_HIP_ERROR(hipGetDeviceCount(&gpu_count));
	cout << "Found " << gpu_count << " GPUs available." << endl;
	if (gpu_number > 0 && gpu_number <= gpu_count) {
		gpu_count = gpu_number;
	}
	if (gpu_count == 0) {
		cerr << "No GPU devices found!" << endl;
		exit(EXIT_FAILURE);
	}

	// 1. Get the total GPU memory size
	size_t total_mem = 0;
	hipMemGetInfo(&total_mem, nullptr);

	// 2. Calculate the required GPU memory size
	size_t required_mem = calculate_memory_size(N, K, M);
	cout << "Required memory: " << required_mem / (1024 * 1024) << " MB" << endl;
	cout << "Available memory: " << total_mem / (1024 * 1024) << " MB" << endl;

	// 3. Determine whether split processing is needed
	size_t element_size = sizeof(float);

	// Check if the size of database exceeds 70% of the GPU memory
	if (N * K * element_size > total_mem * 0.7) {
		need_batching = true;
		batchSize_B = static_cast<size_t>(total_mem * 0.7 / K / element_size); // B 划分为显存的 70%

		size_t A_C_size = K * M * element_size + N * M * element_size;
		if (A_C_size > total_mem * 0.2) {
			batchSize_A = static_cast<size_t>(total_mem * 0.2 / element_size / (K + N)); // 对 A 进行划分
		} else {
			batchSize_A = M;  // do not need split sample, calculate directly
		}
	} else {
		size_t B_A_C_size = N * K * element_size + K * M * element_size + N * M * element_size;
		if (B_A_C_size <= total_mem * 0.9) {
			batchSize_B = N;
			batchSize_A = M;  // do not need split, calculate directly
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

	cout << "Partition-database Size: " << batchSize_B << ", Batch-sample Size: " << batchSize_A << endl;
	cout << "Need batching: " << need_batching << endl;

	// calculate the number of batches (sample)
	size_t num_batches_A = (M + batchSize_A - 1) / batchSize_A; // Round up
	cout << "number of Batch-sample: " << num_batches_A << endl;

	// If the number of sample batches is less than the number of GPUs, allocate memory only for the GPUs actually used
	max_gpus = min(gpu_count, static_cast<int>(num_batches_A));
	cout << "use " << max_gpus << " GPUs." << endl;

	// Adjust the size of the vector according to the number of GPUs
	device_A.resize(max_gpus);
	device_B.resize(max_gpus);
	device_C.resize(max_gpus);
	result_C.resize(max_gpus);
	handles.resize(max_gpus);

	start_events.resize(max_gpus);
	stop_events.resize(max_gpus);
	gpu_times.resize(max_gpus);

	// Allocate memory for each GPU and create hipblasHandle_t
	#pragma omp parallel for num_threads(max_gpus)
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_HIP_ERROR(hipSetDevice(i));
		CHECK_HIP_ERROR(hipMalloc((void**)&device_B[i], batchSize_B * K * sizeof(float)));  // database
		CHECK_HIP_ERROR(hipMalloc((void**)&device_A[i], K * batchSize_A * sizeof(float)));  // samples
		CHECK_HIP_ERROR(hipMalloc((void**)&device_C[i], batchSize_B * batchSize_A * sizeof(float)));  // results
		CHECK_HIP_ERROR(hipHostMalloc((void**)&result_C[i], batchSize_B * batchSize_A * sizeof(float)));

		// create hipblas handle
		CHECK_HIPBLAS_ERROR(hipblasCreate(&handles[i]));

		// create hip events
		CHECK_HIP_ERROR(hipEventCreate(&start_events[i]));
		CHECK_HIP_ERROR(hipEventCreate(&stop_events[i]));
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

//	cout << "-=-=-=-=-=-=-" << ko_abd_file << "-=-=-=-=-=-=-=-" << endl;

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
	size_t N = KO_Index.size();    // Number of rows in the database
	size_t K = table->Get_Feature_Size();  // Number of columns in the database, number of rows in the sample
	size_t M = table->Get_Sample_Size();    // Number of columns in the sample

	Sample_count = M;

	vector<string> otus = table->Get_Feature_Names();

	// Allocate memory for B if it is nullptr
//	if (hipHostMalloc((void**)&B, N * K * sizeof(float)) != hipSuccess) {
//		fprintf(stderr, "Error allocating memory for B\n");
//		exit(EXIT_FAILURE);
//	}
	B = (float*)malloc(N * K * sizeof(float));

	// check and init database to 0
	if (B != nullptr) {
		printf("database Memory allocation successful!\n");
		memset(B, 0, N * K * sizeof(float));
	} else {
		fprintf(stderr, "database Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	omp_set_num_threads(coren);
	// init matrix database
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
					B[i * N + kos[j].Index] = kos[j].Copy;
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

	// Read all rows and their OTUs into a vector
	vector<string> lines;
	string buffer;
	while (getline(infile, buffer)) {
		lines.push_back(buffer);
	}
	infile.close(); // close file

//	cout << "lines.size() = " << lines.size() << endl;

	// Construct OTU_KO_Index sequentially using OpenMP
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

	double start_time = omp_get_wtime();  // Record the start time

	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(dynamic, 1)
	for(int i = 0; i < table->Get_Sample_Size(); i ++)
		Load_Sample_By_OTU(&(otu_seq_count[i]), i, skipNormalization, max_nsti);

	double end_time = omp_get_wtime();    // Record the end time
	double elapsed_ms = (end_time - start_time) * 1000.0;  // ms
	cout << "Parallel calculation took " << elapsed_ms << " ms." << endl;

	return table->Get_Sample_Size();
}

void _KO_OTU_Table_All::matrixMultiplyS(float *B, float *A, size_t N, size_t K, size_t M, int coren) {

	// 1. get total GPU memory
	size_t total_mem = 0;
	hipMemGetInfo(&total_mem, nullptr);

	// 2. calculate needed memory
	size_t required_mem = calculate_memory_size(N, K, M);
	cout << "Required memory: " << required_mem / (1024 * 1024) << " MB" << endl;
	cout << "Available memory: " << total_mem / (1024 * 1024) << " MB" << endl;

	// 3. Check whether splitting is needed
	size_t batchSize_B, batchSize_A;
	bool need_batching = false;

	size_t element_size = sizeof(float);

	// Check if the size of the database exceeds 70% of the GPU memory
	if (N * K * element_size > total_mem * 0.7) {
		need_batching = true;
		batchSize_B = static_cast<size_t>(total_mem * 0.7 / K / element_size); // Split database into 70% of the GPU memory

		size_t A_C_size = K * M * element_size + N * M * element_size;
		if (A_C_size > total_mem * 0.2) {
			batchSize_A = static_cast<size_t>(total_mem * 0.2 / element_size / (K + N)); // Split the samples
		} else {
			batchSize_A = M;  // Do not need split samples, calculate directly
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

	// 4. allocation GPU valuable name
	float *device_B, *device_A, *device_C;
	hipblasHandle_t handle;
	CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

	// 5. allocation GPU memory
	CHECK_HIP_ERROR(hipMalloc((void**)&device_B, batchSize_B * K * sizeof(float))); // database
	CHECK_HIP_ERROR(hipMalloc((void**)&device_A, K * batchSize_A * sizeof(float))); // sample
	CHECK_HIP_ERROR(hipMalloc((void**)&device_C, batchSize_B * batchSize_A * sizeof(float))); // result

	// time
	hipEvent_t start, stop;
	float time = 0.0f;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	float dt;

	// 6. calculate diractly
	if (!need_batching) {

		// allocation result memory
		float *C;
		CHECK_HIP_ERROR(hipHostMalloc((void**)&C, N * M * sizeof(float)));
		// check h_sample_ko
		if (C != nullptr) {
			printf("result Memory allocation successful!\n");
		} else {
			fprintf(stderr, "result Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		// 6.1 transfer database from RAM to GPU
		CHECK_HIP_ERROR(hipMemcpy(device_B, B, batchSize_B * K * sizeof(float), hipMemcpyHostToDevice));

		for (size_t j = 0; j < M; j += batchSize_A) {
			size_t cur_batchSize_A = min(batchSize_A, M - j);
			size_t offset = j * K;  // the begin position of sample in RAM

			// 6.3 transfer current batch from RAM to GPU
			CHECK_HIP_ERROR(hipMemcpy(device_A, A + offset, K * cur_batchSize_A * sizeof(float), hipMemcpyHostToDevice));

			// 6.2 init result to 0
			CHECK_HIP_ERROR(hipMemset(device_C, 0, batchSize_B * batchSize_A * sizeof(float)));

			// 6.4 calculation
			const float alpha = 1.0f;
			const float beta = 0.0f;

			hipEventRecord(start); // record start
			CHECK_HIPBLAS_ERROR(hipblasSgemm(handle,
			                                 HIPBLAS_OP_N, HIPBLAS_OP_N,
			                                 N, cur_batchSize_A, K,
			                                 &alpha,
			                                 device_B, N,
			                                 device_A, K,
			                                 &beta,
			                                 device_C, N));
			hipEventRecord(stop);  // record end
			hipEventSynchronize(stop);
			hipEventElapsedTime(&dt, start, stop);
			printf("Batch-sample %zu: Calculation took %8.3f ms.\n", j, dt);
			time += dt;

			// 6.5 transfer result frm GPU to RAM
			CHECK_HIP_ERROR(hipMemcpy(C + j * N, device_C, N * cur_batchSize_A * sizeof(float), hipMemcpyDeviceToHost));
		}
		printf("Calculation took %8.3f ms.\n", time);

		// normalize result for output
		omp_set_num_threads(coren);
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				KO_Abd[i][j] = C[i * N + j];
			}
		}

		CHECK_HIP_ERROR(hipHostFree(C));

	} else {
		// 7 calculation for Partitions and Batches
		// 7.1 allocation result memory
		float *result_C;
		CHECK_HIP_ERROR(hipHostMalloc((void**)&result_C, batchSize_B * batchSize_A * sizeof(float)));
		if (result_C != nullptr) {
			printf("result Memory allocation successful!\n");
		} else {
			fprintf(stderr, "result Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		omp_set_num_threads(coren);
		for (size_t i = 0; i < N; i += batchSize_B) {
			size_t cur_batchSize_B = min(batchSize_B, N - i);
			// 7.2 transfer currnet Partition from RAM to GPU
			#pragma omp parallel for schedule(static)
			for (size_t k = 0; k < K; ++k) {
				CHECK_HIP_ERROR(hipMemcpy(device_B + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), hipMemcpyHostToDevice));
			}
			for (size_t j = 0; j < M; j += batchSize_A) {
				size_t cur_batchSize_A = min(batchSize_A, M - j);

				// 7.3 transfer currnet Batch from RAM to GPU
				size_t offset = j * K;
				CHECK_HIP_ERROR(hipMemcpy(device_A, A + offset, K * cur_batchSize_A * sizeof(float), hipMemcpyHostToDevice));

				// 7.4 init result to 0
				CHECK_HIP_ERROR(hipMemset(device_C, 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

				// 7.5 calculation
				float alpha = 1.0f;
				float beta = 0.0f;

				hipEventRecord(start); // record start
				CHECK_HIPBLAS_ERROR(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, cur_batchSize_B, cur_batchSize_A, K,
				                                 &alpha, device_B, cur_batchSize_B, device_A, K, &beta, device_C, cur_batchSize_B));
				hipEventRecord(stop);  // record end
				hipEventSynchronize(stop);
				hipEventElapsedTime(&dt, start, stop);
				printf("Partition-database %zu, Batch-sample %zu: Calculation took %8.3f ms.\n", i, j, dt);
				time += dt;

				// 7.6 transfer result from GPU to RAM
				CHECK_HIP_ERROR(hipMemcpy(result_C, device_C, cur_batchSize_B * cur_batchSize_A * sizeof(float), hipMemcpyDeviceToHost));

				// 7.7 normalization for oputput
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
		CHECK_HIP_ERROR(hipHostFree(result_C));
	}

	// 8. release source
	hipblasDestroy(handle);
	hipFree(device_B);
	hipFree(device_A);
	hipFree(device_C);

	hipEventDestroy(start);
	hipEventDestroy(stop);

}

void _KO_OTU_Table_All::matrixMultiplyS_more(float *B, float *A, size_t N, size_t K, size_t M, int coren) {

	float dt;
	gpu_times[0] = 0.0f;

	if (!need_batching) { // calculate directly
		for (size_t j = 0; j < M; j += batchSize_A) {
			size_t cur_batchSize_A = min(batchSize_A, M - j);
			size_t offset = j * K;

			// 6.3 transfer sample data from RAM to GPU
			CHECK_HIP_ERROR(hipMemcpy(device_A[0], A + offset, K * cur_batchSize_A * sizeof(float), hipMemcpyHostToDevice));

			// 6.2 init result to 0
			CHECK_HIP_ERROR(hipMemset(device_C[0], 0, batchSize_B * batchSize_A * sizeof(float)));

			// 6.4 calculation
			const float alpha = 1.0f;
			const float beta = 0.0f;

			hipEventRecord(start_events[0]); // record start
			CHECK_HIPBLAS_ERROR(hipblasSgemm(handles[0],
			                                 HIPBLAS_OP_N, HIPBLAS_OP_N,
			                                 N, cur_batchSize_A, K,
			                                 &alpha,
			                                 device_B[0], N,
			                                 device_A[0], K,
			                                 &beta,
			                                 device_C[0], N));
			hipEventRecord(stop_events[0]);  // record end
			hipEventSynchronize(stop_events[0]);
			hipEventElapsedTime(&dt, start_events[0], stop_events[0]);
			printf("Batch-sample %zu: Calculation took %8.3f ms.\n", j, dt);
			gpu_times[0] += dt;

			// 6.5 transfer result from GPU to RAM
			CHECK_HIP_ERROR(hipMemcpy(result_C[0] + j * N, device_C[0], N * cur_batchSize_A * sizeof(float), hipMemcpyDeviceToHost));
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
		// allocation result memory
		omp_set_num_threads(coren);
		for (size_t i = 0; i < N; i += batchSize_B) {
			size_t cur_batchSize_B = min(batchSize_B, N - i);
			// 7.2 transfer partition from RAM to GPU
			#pragma omp parallel for schedule(static)
			for (size_t k = 0; k < K; ++k) {
				CHECK_HIP_ERROR(hipMemcpy(device_B[0] + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), hipMemcpyHostToDevice));
			}
			for (size_t j = 0; j < M; j += batchSize_A) {
				size_t cur_batchSize_A = min(batchSize_A, M - j);

				// 7.3 transfer batch from RAM to GPU
				size_t offset = j * K;
				CHECK_HIP_ERROR(hipMemcpy(device_A[0], A + offset, K * cur_batchSize_A * sizeof(float), hipMemcpyHostToDevice));

				// 7.4 init result to 0
				CHECK_HIP_ERROR(hipMemset(device_C[0], 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

				// 7.5 calculation
				float alpha = 1.0f;
				float beta = 0.0f;

				hipEventRecord(start_events[0]); // record start
				CHECK_HIPBLAS_ERROR(hipblasSgemm(handles[0], HIPBLAS_OP_N, HIPBLAS_OP_N, cur_batchSize_B, cur_batchSize_A, K,
				                                 &alpha, device_B[0], cur_batchSize_B, device_A[0], K, &beta, device_C[0], cur_batchSize_B));
				hipEventRecord(stop_events[0]);  // record end
				hipEventSynchronize(stop_events[0]);
				hipEventElapsedTime(&dt, start_events[0], stop_events[0]);
				printf("Partition-database %zu, Batch-sample %zu: Calculation took %8.3f ms.\n", i, j, dt);
				gpu_times[0] += dt;

				// 7.6 transfer result from GPU to RAM
				CHECK_HIP_ERROR(hipMemcpy(result_C[0], device_C[0], cur_batchSize_B * cur_batchSize_A * sizeof(float), hipMemcpyDeviceToHost));

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

//	// Automatically get the number of available GPUs
//	int gpu_count = 0;
//	CHECK_HIP_ERROR(hipGetDeviceCount(&gpu_count));
//	if (gpu_count == 0) {
//		cerr << "No GPU devices found!" << endl;
//		exit(EXIT_FAILURE);
//	}
//	cout << "Found " << gpu_count << " GPUs available." << endl;

	// Automatically get the number of available GPUs
	int gpu_count = 0;
	CHECK_HIP_ERROR(hipGetDeviceCount(&gpu_count));
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
	hipMemGetInfo(&total_mem, nullptr);

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
//	cout << "First batchSize_A: " << batchSize_A << endl;
//	cout << "First batch: " << batch << endl;
	if (batch < 1) batch = 1;
	while (batch > batchSize_A) {
		batch = batch / gpu_count;
		if (batch < 1) batch = 1;
	}
//	cout << "End batch: " << batch << endl;
	batchSize_A = batch;
	need_batching = true;

	cout << "Partition-database Size: " << batchSize_B << ", Batch-sample Size: " << batchSize_A << endl;
	cout << "Need batching: " << need_batching << endl;

	// calculate the number of batch (sample)
	size_t num_batches_A = (M + batchSize_A - 1) / batchSize_A; // round up
	cout << "number of Batch-sample: " << num_batches_A << endl;

	int max_gpus = min(gpu_count, static_cast<int>(num_batches_A));
	cout << "use " << max_gpus << " GPUs." << endl;

	vector<float*> device_B(max_gpus), device_A(max_gpus), device_C(max_gpus);
	vector<float*> result_C(max_gpus); // result
	vector<hipblasHandle_t> handles(max_gpus);  // hipblasHandle_t

	// === events and time ===
	vector<hipEvent_t> start_events(max_gpus);
	vector<hipEvent_t> stop_events(max_gpus);
	vector<float> gpu_times(max_gpus, 0.0f);  // GPUs calculation time

	// allocate memory for all GPUs and hipblasHandle_t
	#pragma omp parallel for num_threads(max_gpus)
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_HIP_ERROR(hipSetDevice(i));
		CHECK_HIP_ERROR(hipMalloc((void**)&device_B[i], batchSize_B * K * sizeof(float)));  // database
		CHECK_HIP_ERROR(hipMalloc((void**)&device_A[i], K * batchSize_A * sizeof(float)));  // sample
		CHECK_HIP_ERROR(hipMalloc((void**)&device_C[i], batchSize_B * batchSize_A * sizeof(float)));  // result
		CHECK_HIP_ERROR(hipHostMalloc((void**)&result_C[i], batchSize_B * batchSize_A * sizeof(float)));

		// create hipblas handle
		CHECK_HIPBLAS_ERROR(hipblasCreate(&handles[i]));

		// create hip event
		CHECK_HIP_ERROR(hipEventCreate(&start_events[i]));
		CHECK_HIP_ERROR(hipEventCreate(&stop_events[i]));
	}

	int actual_coren = coren/max_gpus - max_gpus;
	cout << "actual_coren = " << actual_coren << endl;

	for (size_t i = 0; i < N; i += batchSize_B) {
		size_t cur_batchSize_B = min(batchSize_B, N - i);  // the size of current partition

		// transfer partition from RAM to GPU
		if (batchSize_B >= N ) {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_HIP_ERROR(hipSetDevice(gpu_idx));
				CHECK_HIP_ERROR(hipMemcpy(device_B[gpu_idx], B, batchSize_B * K * sizeof(float), hipMemcpyHostToDevice));
			}
		} else {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_HIP_ERROR(hipSetDevice(gpu_idx));
				#pragma omp parallel for num_threads(actual_coren) schedule(static)
				for (size_t k = 0; k < K; ++k) {
					CHECK_HIP_ERROR(hipMemcpy(device_B[gpu_idx] + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), hipMemcpyHostToDevice));
				}
			}
		}

		#pragma omp parallel for num_threads(max_gpus)
		for (size_t j = 0; j < M; j += batchSize_A) {
			int gpu_idx = omp_get_thread_num();
			CHECK_HIP_ERROR(hipSetDevice(gpu_idx));

			size_t cur_batchSize_A = min(batchSize_A, M - j);  // the size of current batch

			// transfer batch from GPU to RAM
			CHECK_HIP_ERROR(hipMemcpy(device_A[gpu_idx], A + j * K, K * cur_batchSize_A * sizeof(float), hipMemcpyHostToDevice));

			// init result to 0
			CHECK_HIP_ERROR(hipMemset(device_C[gpu_idx], 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

			// record start
			CHECK_HIP_ERROR(hipEventRecord(start_events[gpu_idx], 0));

			// calculation
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_HIPBLAS_ERROR(hipblasSgemm(handles[gpu_idx], HIPBLAS_OP_N, HIPBLAS_OP_N,
			                                 cur_batchSize_B, cur_batchSize_A, K, &alpha,
			                                 device_B[gpu_idx], cur_batchSize_B,
			                                 device_A[gpu_idx], K, &beta,
			                                 device_C[gpu_idx], cur_batchSize_B));

			// record end
			CHECK_HIP_ERROR(hipEventRecord(stop_events[gpu_idx], 0));
			CHECK_HIP_ERROR(hipEventSynchronize(stop_events[gpu_idx]));

			float milliseconds = 0.0f;
			CHECK_HIP_ERROR(hipEventElapsedTime(&milliseconds, start_events[gpu_idx], stop_events[gpu_idx]));
//			cout << gpu_idx << "----" << milliseconds << endl;
			gpu_times[gpu_idx] += milliseconds;  // add time

			// transfer result from GPU to RAM
			CHECK_HIP_ERROR(hipMemcpy(result_C[gpu_idx], device_C[gpu_idx], cur_batchSize_B * cur_batchSize_A * sizeof(float), hipMemcpyDeviceToHost));

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
	// release result memory for GPUs
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_HIP_ERROR(hipHostFree(result_C[i]));
	}

	// === output GPU computaton time ===
	int gpu_index = 0;
	for (auto t : gpu_times) {
		cout << "GPU "<< gpu_index++ << " computation time: " << t << " ms" << endl;
	}

	for (auto t : gpu_times) total_time += t;
	cout << "Total GPU computation time: " << total_time << " ms" << endl;

	// release source
	#pragma omp parallel for num_threads(max_gpus)
	for (int i = 0; i < max_gpus; ++i) {
		CHECK_HIP_ERROR(hipSetDevice(i));
		CHECK_HIPBLAS_ERROR(hipblasDestroy(handles[i]));
		CHECK_HIP_ERROR(hipFree(device_B[i]));
		CHECK_HIP_ERROR(hipFree(device_A[i]));
		CHECK_HIP_ERROR(hipFree(device_C[i]));

		CHECK_HIP_ERROR(hipEventDestroy(start_events[i]));
		CHECK_HIP_ERROR(hipEventDestroy(stop_events[i]));
	}
}

void _KO_OTU_Table_All::matrixMultiplyM_more(float *B, float *A, size_t N, size_t K, size_t M, int coren) {

	float milliseconds = 0.0f;
	gpu_times.assign(max_gpus, 0.0f);

	// Calculate
	for (size_t i = 0; i < N; i += batchSize_B) {
		size_t cur_batchSize_B = min(batchSize_B, N - i); // Actual size of the current partition

		// transfer current Partition from RAM to all GPU
		if (batchSize_B >= N ) {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_HIP_ERROR(hipSetDevice(gpu_idx));
				CHECK_HIP_ERROR(hipMemcpy(device_B[gpu_idx], B, batchSize_B * K * sizeof(float), hipMemcpyHostToDevice));
			}
		} else {
			#pragma omp parallel for num_threads(max_gpus)
			for (int gpu_idx = 0; gpu_idx < max_gpus; ++gpu_idx) {
				CHECK_HIP_ERROR(hipSetDevice(gpu_idx));
				#pragma omp parallel for num_threads(actual_coren) schedule(static)
				for (size_t k = 0; k < K; ++k) {
					CHECK_HIP_ERROR(hipMemcpy(device_B[gpu_idx] + k * cur_batchSize_B, B + k * N + i, cur_batchSize_B * sizeof(float), hipMemcpyHostToDevice));
				}
			}
		}

		#pragma omp parallel for num_threads(max_gpus)
		for (size_t j = 0; j < M; j += batchSize_A) {
			int gpu_idx = omp_get_thread_num();
			CHECK_HIP_ERROR(hipSetDevice(gpu_idx));

			// calculate current batch size
			size_t cur_batchSize_A = min(batchSize_A, M - j);

			// transfer batch from RAM to GPU
			CHECK_HIP_ERROR(hipMemcpy(device_A[gpu_idx], A + j * K, K * cur_batchSize_A * sizeof(float), hipMemcpyHostToDevice));

			// init result to 0
			CHECK_HIP_ERROR(hipMemset(device_C[gpu_idx], 0, cur_batchSize_B * cur_batchSize_A * sizeof(float)));

			// record start
			CHECK_HIP_ERROR(hipEventRecord(start_events[gpu_idx], 0));

			// computation
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_HIPBLAS_ERROR(hipblasSgemm(handles[gpu_idx], HIPBLAS_OP_N, HIPBLAS_OP_N,
			                                 cur_batchSize_B, cur_batchSize_A, K, &alpha,
			                                 device_B[gpu_idx], cur_batchSize_B,
			                                 device_A[gpu_idx], K, &beta,
			                                 device_C[gpu_idx], cur_batchSize_B));

			// record end
			CHECK_HIP_ERROR(hipEventRecord(stop_events[gpu_idx], 0));
			CHECK_HIP_ERROR(hipEventSynchronize(stop_events[gpu_idx]));

			CHECK_HIP_ERROR(hipEventElapsedTime(&milliseconds, start_events[gpu_idx], stop_events[gpu_idx]));
			gpu_times[gpu_idx] += milliseconds;

			// transfer result from GPU to RAM
			CHECK_HIP_ERROR(hipMemcpy(result_C[gpu_idx], device_C[gpu_idx], cur_batchSize_B * cur_batchSize_A * sizeof(float), hipMemcpyDeviceToHost));

			// normalize result for output
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

	size_t N = KO_Index.size();    // B的行数
	size_t K = table->Get_Feature_Size();  // B的列数，A的行数
	size_t M = table->Get_Sample_Size();    // A的列数

	// allcation RAM for database
	float *B;
//	CHECK_HIP_ERROR(hipHostMalloc((void**)&B, N * K * sizeof(float)));
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
	// copy number
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

	// init database
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < K; ++i) {
		string a_otu_j = Check_OTU(otus[i]);

		// Use find() instead of operator[] to avoid implicit insertion and ensure thread safety
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
			continue;
		}
	}

	// calculation
	if (gpuMode == "S")
		matrixMultiplyS(B, A, N, K, M, coren);
	else
		matrixMultiplyM(B, A, N, K, M, coren, gpu_number);

	// release source
//	CHECK_HIP_ERROR(hipHostFree(B));
	free(B);
	B = nullptr;

	return table->Get_Sample_Size();
}

int _KO_OTU_Table_All::Load_Sample_By_OTU_Table_newG(_Table_Format * table, int coren, bool skipNormalization, string gpuMode, int num, bool over, int gpu_number) {

	size_t N = KO_Index.size();             // Number of rows in database
	size_t K = table->Get_Feature_Size();  // Number of columns in database, number of rows in sample
	size_t M = table->Get_Sample_Size();   // Number of columns in sample


	Sample_count = M;

	vector<string> otus = table->Get_Feature_Names();
	vector<string> sam_names = table->Get_Sample_Names();

	float *A = table->GetA();

	omp_set_num_threads(coren);
	// copy number
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


	float *B_ptr = B;

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
	// release source
	if (over) {
		printf("Calculation took %8.3f ms(total).\n", total_time);
		releaseMemory();
//		CHECK_HIP_ERROR(hipHostFree(B));
		free(B);
		B = nullptr;
	}

	return table->Get_Sample_Size();
}

int _KO_OTU_Table_All::Load_Sample_By_Single_KO_Table(const char * infilename, string sample_name, int sample) {

	ifstream infile(infilename, ifstream::in);
	if (!infile) {
		cerr << "Error: Open input file error : " << infilename << endl;
		exit(0);
	}

	int count = 0;

	Sample_names[sample] = sample_name;
	string buffer;
	getline(infile, buffer);
	while(getline(infile, buffer)) {

		stringstream strin(buffer);
		string ko;
		float ko_count;

		strin >> ko >> ko_count;

		if (KO_Index.count(ko) == 0) continue;
		int index = KO_Index[ko];

		KO_Abd[sample][index] = ko_count;

		count ++;
	}

	infile.close();
	infile.clear();

	return count;;
}

int _KO_OTU_Table_All::Load_Sample_By_KO_Table(_Table_Format * table) {

	vector <string> kos = table->Get_Feature_Names();
	vector <string> sam_names = table->Get_Sample_Names();

	for (int i = 0; i < table->Get_Sample_Size(); i ++) {

		Sample_names[i] = sam_names[i];
		//vector <float> abd= table->Get_Abd(i);

		for (int j = 0; j < table->Get_Feature_Size(); j ++) {
			if (table->Get_Abd_By_Order(i, j) <= 0) continue;
			if (KO_Index.count(kos[j]) == 0) continue;
			KO_Abd[i][KO_Index[kos[j]]] = table->Get_Abd_By_Order(i, j);
		}
	}

	return table->Get_Sample_Size();
}

int _KO_OTU_Table_All::Output(const char * outfilename) {

	ofstream outfile(outfilename, ofstream::out);
	if (!outfile) {
		cerr << "Error: Cannot open output file : " << outfilename << endl;
		return 0;
	}

	int count = 0;

	outfile << "Gene";
	for (int i = 0; i < Sample_count; i ++)
		if (Sample_count == 1) outfile << "\tGene_Count";
		else outfile << "\t" << Sample_names[i] << "_Gene_Count";

	outfile << "\tGene_Description\tKEGG_Pathway" << endl;


	for (int i = 0; i < KOs.size(); i ++) {

		bool No_Zero_Check = false;
		for (int j = 0; j < Sample_count; j ++)
			if (KO_Abd[j][i] > 0) No_Zero_Check = true;
		if (!No_Zero_Check) continue;
		count ++;
		outfile << KOs[i].Get_Name();

		for (int j = 0; j < Sample_count; j ++)
			outfile << "\t" << KO_Abd[j][i];

		outfile << "\t" << KOs[i].Get_Des() << "\t" << KOs[i].Get_Pathway() << endl;
	}


	outfile.close();
	outfile.clear();

	return count;
}

int _KO_OTU_Table_All::Output_Multi(vector <string> outfilename) {

	int count = 0;

	for (int i = 0; i < KOs.size(); i ++) {

		bool No_Zero_Check = false;
		for (int j = 0; j < Sample_count; j ++)
			if (KO_Abd[j][i] > 0) No_Zero_Check = true;
		if (!No_Zero_Check) continue;
		count ++;
	}

	for (int i = 0; i < Sample_count; i ++) {

		ofstream outfile(outfilename[i].c_str(), ofstream::out);
		if (!outfile) {
			cerr << "Error: Cannot open output file : " << outfilename[i] << endl;
			return 0;
		}
		outfile << "Gene\tGene_Count\tGene_Description\tKEGG_Pathway" << endl;

		for (int j = 0; j < KOs.size(); j ++) {

			if (KO_Abd[i][j] <= 0) continue;

			outfile << KOs[j].Get_Name() << "\t" << KO_Abd[i][j];
			outfile << "\t" << KOs[j].Get_Des() << "\t" << KOs[j].Get_Pathway() << endl;
		}

		outfile.close();
		outfile.clear();
	}

	return count;
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

	// open file
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

	// create path index
	int index = 0;
	for (hash_map<string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
		pw_index[miter->first] = index++;
	}

	vector<vector<float>> pw_count(Sample_count, vector<float>(pw_ko.size(), 0));
	vector<vector<float>> pw_abd(Sample_count, vector<float>(pw_ko.size(), 0));

	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < Sample_count; i ++) {
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			for (size_t j = 0; j < miter->second.size(); j++) {
				pw_count[i][pw_index[miter->first]] += KO_Abd[i][KO_Index[miter->second[j]]];
			}
		}
	}

	// Normalize
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

	// Output the file header
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

		// Write the buffered content to the file
		outfile_abd << abd_buffer.str();
		outfile_count << count_buffer.str();
	}

	// close file
	outfile_abd.close();
	outfile_abd.clear();
	outfile_count.close();
	outfile_count.clear();

//	delete[] pw_count;
//	delete[] pw_abd;
	delete[] No_Zero_Check;

	return count;
}

int _KO_OTU_Table_All::Output_By_Category_app_openMP(const char * outfilename, int level, float max, float min, int coren, int num) { //2 output, abd and count
	hash_map <string, vector<string>, std_string_hash> pw_ko;
	hash_map <string, int, std_string_hash> pw_index;

	// open file
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

	//  add KO
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

	// create path index
	int index = 0;
	for (hash_map<string, vector<string>, std_string_hash> ::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
		pw_index[miter->first] = index++;
	}

	vector<vector<float>> pw_count(Sample_count, vector<float>(pw_ko.size(), 0));
	vector<vector<float>> pw_abd(Sample_count, vector<float>(pw_ko.size(), 0));

	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < Sample_count; i ++) {
		for (hash_map<string, vector<string>, std_string_hash>::iterator miter = pw_ko.begin(); miter != pw_ko.end(); ++miter) {
			for (size_t j = 0; j < miter->second.size(); j++) {
				pw_count[i][pw_index[miter->first]] += KO_Abd[i][KO_Index[miter->second[j]]];
			}
		}
	}

	// Normalize
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

	// Output the file header
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

		// Write the buffered content to the file
		outfile_abd << abd_buffer.str();
		outfile_count << count_buffer.str();
	}

	// close file
	outfile_abd.close();
	outfile_abd.clear();
	outfile_count.close();
	outfile_count.clear();

//	delete[] pw_count;
//	delete[] pw_abd;

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
		ostringstream abd_buffer, count_buffer; // Used to buffer output

		outfile_abd << Sample_names[i];
		outfile_count << Sample_names[i];

		for (hash_map <string, int, std_string_hash>::iterator miter = KO_Index.begin(); miter != KO_Index.end(); ++miter) {

			index = KO_Index[miter->first];
			if (!No_Zero_Check[index]) continue;

			count_buffer << "\t" << KO_Abd[i][index];
			abd_buffer << "\t" << ko_abd[i][index];

		}
		// output to file
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
	// file name
	std::string outfilename_abd = std::string(outfilename) + ".Abd";
	std::string outfilename_count = std::string(outfilename) + ".Count";

	// Open the Abd file
	ofstream outfile_abd;
	if (num == 0)
		outfile_abd.open(outfilename_abd.c_str(), ofstream::out | ofstream::trunc);  // new and clean
	else
		outfile_abd.open(outfilename_abd.c_str(), ofstream::out | ofstream::app);    // add

	if (!outfile_abd) {
		cerr << "Error: Cannot open output file: " << outfilename_abd << endl;
		return 0;
	}

	// Open the Count file
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
	// Normalize
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

	// Output the file header
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

		// Write the buffered content to the file
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

