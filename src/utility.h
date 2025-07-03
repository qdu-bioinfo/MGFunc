// Updated at Jan. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0

#ifndef _UTILITY_H
#define _UTILITY_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <stdlib.h>
#include <string.h>
#include <sys/dir.h>
#include <sys/stat.h>

#define BUFFER_SIZE 5000

#include "hash.h"
using namespace std;

string Check_Env() {

	if (getenv("MGFunc") == NULL) {

		cerr << "Error: Please set the environment variable \"MGFunc\" to the directory" << endl;
		exit(0);

	}

	string path =  getenv("MGFunc");
	return path;

}

string Check_OTU(string otu) {

	string a_otu = otu;
	if (a_otu.size() > 4 ) {
		string prefix_4 = a_otu.substr(0, 4);
		if (( prefix_4 == "otu_") || ( prefix_4 == "OTU_"))
			a_otu = a_otu.substr(4, a_otu.size() - 4);
	}
	return a_otu;
}

#endif
