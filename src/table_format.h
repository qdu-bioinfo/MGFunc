// Updated at Feb. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0

#ifndef table_format_h
#define table_format_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cstdlib>

using namespace std;

class _Table_Format {

	public:

		friend class _Table_Format_Seq;

		_Table_Format() {}
		_Table_Format(const char * infilename); //default is true
		_Table_Format(const char * infilename, int coren);
		_Table_Format(const char * infilename, int coren, int num, int batchSize, bool& over);
		_Table_Format(vector <string> features); //init by features

		int Load_Table(const char * infilename); //row: sample, column: feature
		int Load_Table_openMP(const char * infilename, int coren); //row: sample, column: feature
		int Load_Table_openMP(const char * infilename, int coren, int num, int batchSize, bool& over); //row: sample, column: feature

		unsigned int Output_Table(ostream * outfile); //row: sample, column: feature
		unsigned int Output_Table(const char * outfilename);

		vector <string> Get_Sample_Names();
		vector <string> Get_Feature_Names();
		int Get_Sample_Size();//Get the size of Samples
		int Get_Feature_Size();//Get the size of Features

		float Get_Abd_By_Order(unsigned int s, unsigned int i);
		vector <float> Get_Abd(unsigned int s);

		void ReleaseMemory() {
			if (A) {
				free(A);
//				hipHostFree(A); // free A
				A = nullptr;
			}
		}

		float* GetA() {
			return A;
		}

	protected:
		vector <string> Samples;
		vector <string> Features;
		vector < vector <float> > Abd;

		float *A = nullptr;
		streampos last_pos = 0;  // keep file read position

};

_Table_Format::_Table_Format(const char * infilename) {
	Load_Table(infilename);

} //default true

_Table_Format::_Table_Format(const char * infilename, int coren) {
	Load_Table_openMP(infilename, coren);
}

_Table_Format::_Table_Format(const char * infilename, int coren, int num, int batchSize, bool& over) {
	Load_Table_openMP(infilename, coren, num, batchSize, over);
}

_Table_Format::_Table_Format(vector <string> features) {
	Features = features;
} //default true

int _Table_Format::Load_Table(const char * infilename) {

	ifstream infile(infilename, ifstream::in);
	if (!infile) {
		cerr << "Error: Cannot open file : " << infilename << endl;
		exit(0);
		return 0;
	}

	string buffer;

	//Samples
	getline(infile, buffer);
	stringstream strin(buffer);
	string feature_name;
	strin >> feature_name; // Title
	while(strin >> feature_name) {
		//cout << feature_name<< "\t" ;
		Features.push_back(feature_name);
	}
	//cout<<endl;


	//Data
	while(getline(infile, buffer)) {
		stringstream strin(buffer);
		string sample_id;
		strin >> sample_id;

		vector <float> abd;
		float a_abd;
		while(strin >> a_abd) {
			//cout << a_abd << "\t";
			abd.push_back(a_abd);
		}
		//cout<<endl;

		//cout<< abd.size() <<endl;
		//cout<< Features.size() <<endl;
		//check feature number
		if (abd.size() != Features.size()) {
			cerr << "Error: Sample: " << sample_id << " does not have " << Features.size() << " features" << endl;
			continue;
		}

		Samples.push_back(sample_id);
		Abd.push_back(abd);
	}

	infile.close();
	infile.clear();

	return Samples.size();
}

int _Table_Format::Load_Table_openMP(const char * infilename, int coren) {
	ifstream infile(infilename, ifstream::in);
	if (!infile) {
		cerr << "error:can not open file: " << infilename << endl;
		exit(0);
		return -1;
	}

	string buffer;

	// read feature
	if (!getline(infile, buffer)) {
		cerr << "error:file is empty" << endl;
		exit(0);
		return -1;
	}

	stringstream strin(buffer);
	string feature_name;
	strin >> feature_name; // Title
	while (strin >> feature_name) {
		Features.push_back(feature_name);
	}

	size_t K = Features.size(); // feature number

	// read sample
	vector<string> sample_lines;
	while (getline(infile, buffer)) {
		sample_lines.push_back(buffer);
	}
	infile.close(); // close file

	size_t M = sample_lines.size(); // sample number

	cout << "sample number is " << M << endl;
	cout << "feature number is " << K << endl;

	// allocate A, size is K * M
//	hipHostMalloc((void**)&A, K * M * sizeof(float));
	A = (float*)malloc(K * M * sizeof(float));

	if (A != nullptr) {
		printf("sample Memory allocation successful!\n");
	} else {
		fprintf(stderr, "sample Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	// read and make A
	Samples.resize(M);
	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(static)
	for (size_t j = 0; j < M; ++j) {
		stringstream strin(sample_lines[j]);
		string sample_id;
		strin >> sample_id; // get sample ID
		Samples[j] = sample_id;

		for (size_t i = 0; i < K; ++i) {
			float value;
			strin >> value;
			A[j * K + i] = value; // col first
		}
	}

	return Samples.size();
}

int _Table_Format::Load_Table_openMP(const char * infilename, int coren, int num, int batchSize, bool& over) {
	ifstream infile(infilename, ifstream::in);
	if (!infile) {
		cerr << "error:can not open file:" << infilename << endl;
		exit(0);
		return -1;
	}

	// recover last_pos
	infile.seekg(last_pos);

	string buffer;

	if(over) {
		// read feature
		if (!getline(infile, buffer)) {
			cerr << "error:file is empty" << endl;
			exit(0);
			return -1;
		}
		stringstream strin(buffer);
		string feature_name;
		strin >> feature_name; // Title
		while (strin >> feature_name) {
			Features.push_back(feature_name);
		}
	}

	size_t K = Features.size(); // feature number

	vector<string> sample_lines;
	size_t current_row = 0;

	while (current_row < batchSize && getline(infile, buffer)) {
		sample_lines.push_back(buffer);
		current_row++;
	}

	if (current_row == 0) {
		cerr << "error:line is not enough, plsease check -b" << endl;
		exit(0);
		return -1;
	}

	size_t M = sample_lines.size();

	cout << "sample number is " << M << endl;
	cout << "feature number is " << K << endl;
	if(over) {
		// allocate A, size is K * M
		A = (float*)malloc(K * M * sizeof(float));
//		hipHostMalloc((void**)&A, K * M * sizeof(float));

		if (A != nullptr) {
			printf("sample Memory allocation successful!\n");
		} else {
			fprintf(stderr, "sample Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}
	}

	// update last_pos
	last_pos = infile.tellg();

	// judge file is over
	if (getline(infile, buffer)) {
		// not
		over = false;
		cout << "file not over" << endl;
	} else {
		// yes
		over = true;
		cout << "file over" << endl;
	}

	infile.close(); // close file

	Samples.clear();
	Samples.resize(M);

	omp_set_num_threads(coren);
	#pragma omp parallel for schedule(static)
	for (size_t j = 0; j < M; ++j) {
		stringstream strin(sample_lines[j]);
		string sample_id;
		strin >> sample_id;
		Samples[j] = sample_id;

		for (size_t i = 0; i < K; ++i) {
			float value;
			strin >> value;
			A[j * K + i] = value; // col first
		}
	}

	return Samples.size();
}

// #############################################################################

vector <string> _Table_Format::Get_Sample_Names() {
	return Samples;
}
vector <string> _Table_Format::Get_Feature_Names() {
	return Features;
}
int _Table_Format::Get_Sample_Size() {
	return Samples.size();
}

int _Table_Format::Get_Feature_Size() {
	return Features.size();
}

float _Table_Format::Get_Abd_By_Order(unsigned int s, unsigned int i) {

	if (s >= Samples.size() || (i >= Features.size()))
		return 0;
	return Abd[s][i];
}

vector <float> _Table_Format::Get_Abd(unsigned int s) {

	if (s < Samples.size())
		return Abd[s];
}

// #############################################################################
#endif /* table_format_hpp */
