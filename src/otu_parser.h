// Updated at Feb. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0


#ifndef otu_parser_h
#define otu_parser_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "hash.h"
#include "db.h"

#define BUFF 10000000
#define STRLEN 1000
#define TAXA_LEVEL 8 //K, P, C, O, F, G, S, OTU

using namespace std;

class _OTU_Parser {

	public:

		_OTU_Parser() {};

		_OTU_Parser(_PMDB db, bool skipNormalization) {
			Database = db;
			if (!skipNormalization)
				Database.Load_Copy_Number(Cp_number);
			
			Database.Load_Nsti(nsti_table);
		};
		_OTU_Parser(_PMDB db, int coren, bool skipNormalization) {
			Database = db;
			if (!skipNormalization)
				Database.Load_Copy_Number_openMP(Cp_number, coren);
			
			Database.Load_Nsti(nsti_table);
		};
		
		float Get_cp_by_OTU(string otu);
		double Get_nsti_by_OTU(string otu);

	private:
		_PMDB Database;
		hash_map<string, string, std_string_hash> OTU_taxa_table;
		hash_map <string, float, std_string_hash> Cp_number;
		hash_map <string, double, std_string_hash> nsti_table; 
};

float _OTU_Parser::Get_cp_by_OTU(string otu) {
	if (Cp_number.count(otu) != 0)
		return Cp_number[otu];
	return 1;
}

double _OTU_Parser::Get_nsti_by_OTU(string otu) {
	if (nsti_table.count(otu) != 0)
		return nsti_table[otu];
	return -1;
}
#endif
