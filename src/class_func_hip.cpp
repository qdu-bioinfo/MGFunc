// Updated at Feb. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0

#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#include <string>
#include <cctype>
#include <cstdlib>

int getAvailableGPUCount() {
	int count = 0;
	hipGetDeviceCount(&count);
	return count;
}


#include "class_func_hip.h"
#include "timer.h"
#include <iostream>

using namespace std;

// Parameters Def
_PMDB Database;

string Tablefile;

vector <string> Infilename;
vector <string> Sam_name;
vector <string> Outfilename;

string Out_path = "result";

int Mode = -1;

int Coren = 0;
int batchSize = 0;

bool over = true;	// read file flag
int num = 0; // read file number

bool useGPU = true;	// use GPU
string gpuMode = "S";	// gpu mode
bool skipNormalization = false;  // skipNormalization

int gpu_number = 1;
double max_nsti = 2.0;

int printhelp() {
	// print version and description
	cout << "MGFunc version: " << Version << endl;
	cout << "Usage:" << endl;
	cout << "\tMGFunc [Option] Value" << endl;
	cout << "Options: " << endl;

	cout << "\n[Input options]" << endl;
	cout << "\t-T    Input OTU count table (*.OTU.Count)" << endl;
	cout << "\t-b    Number of iteration size, default: no iteration" << endl;
	cout << "\t-D    Reference database, options: " << _PMDB::Get_Func_Args() << endl;

	cout << "\n[GPU options]" << endl;
	cout << "\t-G    Use GPUs: 'a' for all available GPUs, or number of GPUs (default is 1)" << endl;

	cout << "\n[Data processing options]" << endl;
	cout << "\t-N    Skip OTU count normalization: T for true, F for false (default is F)" << endl;
	cout << "\t-S    Max NSTI value, OTUs above this value will be excluded (default is 2.0)" << endl;

	cout << "\n[Output options]" << endl;
	cout << "\t-o    Output path (default is ./result)" << endl;

	cout << "\n[Other options]" << endl;
	cout << "\t-t    Number of CPU threads, default: auto" << endl;
	cout << "\t-h    Help" << endl;

	return 0;
}

// Determine whether the string is a positive integer (greater than 0)
bool isPositiveInteger(const std::string& s) {
	if (s.empty()) return false;
	if (s[0] == '0' && s.size() > 1) return false; // Prohibit leading zeros, such as "01"
	return std::all_of(s.begin(), s.end(), ::isdigit);
}

// Determine whether the string is a positive floating-point number (greater than 0)
bool isPositiveFloat(const std::string& s) {
	if (s.empty()) return false;
	bool decimalPointSeen = false;
	int digitCount = 0;
	for (size_t i = 0; i < s.size(); ++i) {
		if (s[i] == '.') {
			if (decimalPointSeen) return false; // There cannot be multiple decimal points
			decimalPointSeen = true;
		} else if (isdigit(s[i])) {
			++digitCount;
		} else {
			return false; // It's neither a number nor a decimal point
		}
	}
	if (digitCount == 0) return false; // There is at least one digit
	try {
		double val = std::stod(s);
		if (val <= 0) return false;
	} catch (...) {
		return false;
	}
	return true;
}

void Parse_Para(int argc, char* argv[]) {
	if (argc == 1) {
		printhelp();
		exit(0);
	}

	int i = 1;
	Mode = 2;

	while (i < argc) {
		if (argv[i][0] != '-') {
			printf("Argument # %d Error: Arguments must start with -\n", i);
			exit(1);
		}
		switch (argv[i][1]) {
			case 'N':  // -N T/F
				if (i + 1 >= argc) {
					cerr << "Error: Missing argument after -N" << endl;
					exit(1);
				}
				if (argv[i + 1][0] == 'T' || argv[i + 1][0] == 't') {
					skipNormalization = true;
				} else if (argv[i + 1][0] == 'F' || argv[i + 1][0] == 'f') {
					skipNormalization = false;
				} else {
					cerr << "Error: Invalid argument for -N. Use T or F." << endl;
					exit(1);
				}
				i += 2;
				break;

			case 'S':  // -S <positive float>
				if (i + 1 >= argc) {
					std::cerr << "Error: Missing argument after -S" << std::endl;
					exit(1);
				}
				{
					std::string arg = argv[i + 1];
					if (!isPositiveFloat(arg)) {
						std::cerr << "Error: Invalid argument for -S: must be a positive float." << std::endl;
						exit(1);
					}
					max_nsti = std::stod(arg);
					i += 2;
				}
				break;

			case 't':  // -t <positive integer>
				if (i + 1 >= argc) {
					std::cerr << "Error: Missing argument after -t" << std::endl;
					exit(1);
				}
				{
					std::string arg = argv[i + 1];
					if (!isPositiveInteger(arg)) {
						std::cerr << "Error: Invalid argument for -t: must be a positive integer." << std::endl;
						exit(1);
					}
					Coren = std::stoi(arg);
					if (Coren <= 0) {
						std::cerr << "Error: -t argument must be greater than 0." << std::endl;
						exit(1);
					}
					i += 2;
				}
				break;
			case 'D':
				Database.Set_DB(argv[i + 1][0]);
				i += 2;
				break;
			case 'T':
				Tablefile = argv[i + 1];
				i += 2;
				break;
			case 'G':
				if (i + 1 < argc) {
					string arg = argv[i + 1];
					if (arg == "a") {
						gpuMode = "M";
						gpu_number = getAvailableGPUCount();
						i += 2;
					} else {
						// check > 0 ?
						bool is_number = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
						if (is_number) {
							int num = stoi(arg);
							if (num == 1) {
								gpuMode = "S";
							} else if (num > 1) {
								gpuMode = "M";
								gpu_number = num;
								int gpu_count = getAvailableGPUCount();
								if (gpu_number > gpu_count) {
									cerr << "Error: only " << gpu_count << " GPU(s) are available" << endl;
									exit(1);
								}
							} else {
								cerr << "Error: Invalid number for -G: " << arg << endl;
								exit(1);
							}
							i += 2;
						} else {
							cerr << "Error: Unrecognized argument for -G: " << arg << endl;
							printhelp();
							exit(1);
						}
					}
				} else {
					cerr << "Error: Missing argument after -G" << endl;
					printhelp();
					exit(1);
				}
				break;
			case 'b': {
				// First, check whether the parameters exist
				if (i + 1 >= argc) {
					cerr << "Error: Missing argument after -b" << endl;
					exit(1);
				}
				string arg = argv[i + 1];
				// Determine whether all are numbers
				bool is_number = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
				if (!is_number) {
					cerr << "Error: Argument for -b must be a positive integer >= 1" << endl;
					exit(1);
				}
				int val = stoi(arg);
				if (val < 1) {
					cerr << "Error: Argument for -b must be >= 1" << endl;
					exit(1);
				}
				batchSize = val;
				Mode = 3;
				i += 2;
				break;
			}
			case 'o':
				Out_path = argv[i + 1];
				i += 2;
				break;
			case 'h':
				printhelp();
				exit(0);
				break;
			default:
				printf("Error: Unrecognized argument %s\n", argv[i]);
				exit(1);
				break;
		}
	}
	
	int max_core_number = sysconf(_SC_NPROCESSORS_CONF);

	if ((Coren <= 0) || (Coren > max_core_number)) {
		//cerr << "Core number must be larger than 0, change to automatic mode" << endl;
		Coren = max_core_number;
	}

	// Print the specified relevant parameter Settings
	cout << "Using GPU mode: " << (gpuMode == "S" ? "Single GPU" : "Multiple GPUs") << endl;

	if(gpu_number > 1)
		cout << "Using number of GPUs is " << gpu_number << endl;


	if (skipNormalization) {
		cout << "Skipping data normalization" << endl;
	}	else {
		cout << "Not skipping data normalization" << endl;
	}

	cout << "max nsti is " << max_nsti << endl;
}

int main(int argc, char * argv[]) {

	Parse_Para(argc, argv);
	int sam_num = 0;

	_KO_OTU_Table_All KOs;

	cout << endl << "Functional Annotation Starts" << endl;
//	cout << "Mode:"<< Mode<< endl;

	switch (Mode) {
		case 2: {
			if (useGPU) {
				_Table_Format table(Tablefile.c_str(), Coren);
				sam_num = table.Get_Sample_Size();
				KOs = _KO_OTU_Table_All(Database, sam_num, 0, Coren, skipNormalization);
				KOs.Load_Sample_By_OTU_Table_G(&table, Coren, skipNormalization, max_nsti, gpuMode, gpu_number);
				KOs.Output_Nsti(useGPU, (Out_path + ".nsti").c_str(), &table, Coren, skipNormalization, max_nsti, num, over);
			} else {
				_Table_Format table(Tablefile.c_str());
				sam_num = table.Get_Sample_Size();
				KOs = _KO_OTU_Table_All(Database, sam_num, 0, skipNormalization);
				KOs.Load_Sample_By_OTU_Table(&table, Coren, skipNormalization, max_nsti);
				KOs.Output_Nsti(useGPU, (Out_path + ".nsti").c_str(), &table, Coren, skipNormalization, max_nsti, num, over);
			}

			cout << "Total Sample Number is " << sam_num << endl;
			cout << KOs.Output_By_KO_openMP((Out_path + ".KO").c_str(), 3, 0, 0, Coren) << " KOs have been parsed out" << endl;
			//cout << KOs.Output_By_Category_openMP((Out_path + ".KO").c_str(), 3, 0, 0, Coren) << " KOs have been parsed out" << endl;
			break;
		}
		case 3: {
			cout << "Iteration Size:" << batchSize << endl;
			float *B = nullptr;
			_Table_Format* table = nullptr; // Declare table using a pointer
			do {
				if (num == 0)
					table = new _Table_Format(Tablefile.c_str(), Coren, num, batchSize, over);
				else
					table->Load_Table_openMP(Tablefile.c_str(), Coren, num, batchSize, over);

				sam_num = table->Get_Sample_Size();
				cout << "current sample number is " << sam_num << endl;

				if (num == 0) {
					KOs = _KO_OTU_Table_All(Database, sam_num, 0, Coren, skipNormalization);
					KOs.make_B(table, max_nsti, Coren);
				}

				KOs.Load_Sample_By_OTU_Table_newG(table, Coren, skipNormalization, gpuMode, num, over, gpu_number);
				KOs.Output_Nsti(useGPU, (Out_path + ".nsti").c_str(), table, Coren, skipNormalization, max_nsti, num, over);

				cout << "Total Sample Number is " << sam_num << endl;

				//cout << KOs.Output_By_Category_app_openMP((Out_path + ".KO").c_str(), 3, 0, 0, Coren, num) << " KOs have been parsed out" << endl;
				cout << KOs.Output_By_KO_app_openMP((Out_path + ".KO").c_str(), 3, 0, 0, Coren, num) << " KOs have been parsed out" << endl;

				num++;
			} while(!over);

			delete table;

			break;
		}
		default:
			cerr << "Error: Incorrect mode";
			exit(0);
	}

	cout << endl << "Functional Annotation Finished"<< endl;

	return 0;
}
