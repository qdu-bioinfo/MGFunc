// Updated at Jan. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0

#include <iostream>
#include <sstream>
#include "utility.h"
#include "hash.h"

using namespace std;

#ifndef _DB_H
#define _DB_H

#define DB_CONFIG_COUNT 8
#define DEFAULT_DB 'G'

class _PMDB {

	public:
		_PMDB() {
			Set_DB(DEFAULT_DB);
		}

		_PMDB(char db) {
			Set_DB(db);
		}

		void Set_DB(char db) {

			Is_taxa = false;
			Is_tree = false;
			Is_cp = false;
			Is_func = false;

			if (DB_config.size() == 0) {
				cerr << "Error: Cannot open reference database configuration" << endl;
				return;
			}

			if (DB_config.count(db) == 0) {
				cerr << "Error: Invalid database option '" << db << endl;
				exit(1);
			}
			vector <string> a_config = DB_config[db];

			//debug
			//cout << a_config.size() << endl;
			//for (int i = 0; i < a_config.size(); i ++)
			//    cout << a_config[i] << endl;

			//make config
			DB_id = a_config[0][0];
			DB_domain = atoi(a_config[1].c_str());
			DB_name = a_config[2];
			DB_description = a_config[3];
			if ((a_config[4][0] == 'Y') || (a_config[4][0] == 'y')) Is_taxa = true;
			if ((a_config[5][0] == 'Y') || (a_config[5][0] == 'y'))  Is_tree = true;
			if ((a_config[6][0] == 'Y') || (a_config[6][0] == 'y')) Is_cp = true;
			if ((a_config[7][0] == 'Y') || (a_config[7][0] == 'y')) Is_func = true;

			Make_base();
			Make_taxa();
			Make_tree();
			Make_func();

		}

		char Get_Id() {
			return DB_id;
		}

		int Get_Domain() {
			return DB_domain;
		}

		string Get_Path() {
			return DB_path;
		}
		string Get_Description() {
			return DB_description;
		}

		//taxa
		unsigned int Read_Taxonomy(hash_map<string, string, std_string_hash> & table);
		unsigned int Read_Taxonomy_openMP(hash_map<string, string, std_string_hash> & table, int coren);
		//cp
		int Load_Copy_Number(hash_map <string, float, std_string_hash> & cp_number);
		int Load_Copy_Number_openMP(hash_map <string, float, std_string_hash> & cp_number, int coren);
		//nsti
		int Load_Nsti(hash_map <string, double, std_string_hash> & nsti_table);

		//tree
		string Get_Tree_Id() {
			return DB_tree_id;
		}

		string Get_Tree_Order() {
			return DB_tree_order;
		}

		//func
		string Get_Func_Id() {
			return DB_func_id;
		}
		string Get_Func() {
			return DB_func;
		}
		string Get_Func_Des() {
			return DB_func_des;
		}
		string Get_Func_Pw() {
			return DB_func_pw;
		}
		string Get_NSTI() {
			return DB_nsti;
		}

		bool Get_Is_Tree() {
			return Is_tree;
		}

		bool Get_Is_Cp() {
			return Is_cp;
		}

		bool Get_Is_Func() {
			return Is_func;
		}

		static int Load_config();

		static string Get_Args() {
			return DB_args;
		}

		static string Get_Func_Args() {
			return DB_func_args;
		}

	private:
		char DB_id;
		int DB_domain; //0: bacteria; 1: euk
		string DB_name;
		string DB_description;
		string DB_path;
		string DB_cp_number;
		string DB_taxa_anno;

		string DB_tree_id;
		string DB_tree_order;

		string DB_func_path;
		string DB_func_id;
		string DB_func;
		string DB_func_des;
		string DB_func_pw;
		string DB_nsti;

		bool Is_taxa;
		bool Is_tree;
		bool Is_cp;
		bool Is_func;

		void Make_base() {
			DB_path = Check_Env();
			DB_path += "/databases/";

			DB_func_path = DB_path + "KO/";

			DB_path += DB_name;
			DB_path += "/";
		}

		void Make_taxa() {
			DB_cp_number = DB_path + "copy_number.txt";
			DB_taxa_anno = DB_path + "taxonomy_annotation.txt";
		}

		void Make_tree() {
			DB_tree_id = DB_path + "tree/id.txt";
			DB_tree_order = DB_path + "tree/order.txt";
		};
		void Make_func() {
			DB_nsti = DB_path + "otu_nsti.tab";
			DB_func = DB_path + "KO/ko.tab";

			DB_func_id = DB_func_path + "ko_id.tab";
			DB_func_des = DB_func_path + "ko_des.tab";
			DB_func_pw = DB_func_path + "ko_pw.tab";

		}

		static map <char, vector <string> > DB_config;
		static string DB_args;
		static string DB_func_args;
		static int DB_config_count;
};

//public
unsigned int _PMDB::Read_Taxonomy(hash_map<string, string, std_string_hash> & table) {

	ifstream infile(DB_taxa_anno.c_str(), ifstream::in);
	if (!infile) {
		cerr << "Error: Open Taxonomy Annotation file error : " << DB_taxa_anno << endl;
		exit(0);
	}

	unsigned int count = 0;

	string buffer;
	getline(infile, buffer); //title

	while (getline(infile, buffer)) {

		if (buffer.size() == 0) continue;

		string id;
		string taxa;
		string taxon;
		/*
		int first_table =  buffer.find('\t', 0); //location of first '\t'
		int last_table = buffer.rfind('\t', buffer.size()-1);//location of last '\t'

		id = buffer.substr(0, first_table);
		taxa = buffer.substr(last_table + 1, buffer.size() - last_table -1 );
		*/

		stringstream strin(buffer);
		strin >> id;
		strin >> taxa;
		while(strin >> taxon) {
			taxa += "\t";
			taxa += taxon;
		}

		if (table.count(id) > 0) {

			cerr << "Error: Loading Taxonomy Annotation error : Duplicate ID : " << id << endl;
			exit(0);

		} else (table)[id] = taxa;

		count ++;

	}

	infile.close();
	infile.clear();

	return count;
}

unsigned int _PMDB::Read_Taxonomy_openMP(hash_map<string, string, std_string_hash> & table, int coren) {
	ifstream infile(DB_taxa_anno.c_str(), ifstream::in);
	if (!infile) {
		std::cerr << "Error: Open Taxonomy Annotation file error: " << DB_taxa_anno << std::endl;
		return 0;
	}

	string buffer;
	getline(infile, buffer); // Skip title line

	vector<string> lines;
	while (getline(infile, buffer)) {
		lines.push_back(buffer); // Collect non-empty lines
	}
	infile.close();

	omp_set_num_threads(coren); // Set the number of threads used
	#pragma omp parallel
	{
		// Create a local unordered_map for each thread
		hash_map<string, string, std_string_hash> local_table;

		#pragma omp for
		for (size_t i = 0; i < lines.size(); ++i) {
			istringstream strin(lines[i]);
			string id, taxa, taxon;

			strin >> id;
			strin >> taxa;
			while (strin >> taxon) {
				taxa += "\t"; // add tab
				taxa += taxon;
			}

			// 插入到局部表中
			if (local_table.count(id) == 0) {
				local_table[id] = taxa;
			}
		}

		// Enter critical section to merge the local table into the shared table
		#pragma omp critical
		{
			for (const auto &pair : local_table) {
				if (table.count(pair.first) > 0) {
					cerr << "Error: Loading Taxonomy Annotation error: Duplicate ID: " << pair.first << std::endl;
					exit(0);
				} else {
					table[pair.first] = pair.second;
				}
			}
		}
	}

	return table.size(); // return the number of table
}

int _PMDB::Load_Copy_Number(hash_map <string, float, std_string_hash> & cp_number) {

	ifstream in_cp_number(DB_cp_number.c_str(), ifstream::in);
	if (!in_cp_number) {
		cerr << "Cannot open copy number table : " << DB_cp_number << ", copy number normalization is disabled" << endl;
		exit(1);
	}

	int count = 0;
	string buffer;
	getline(in_cp_number, buffer);
	while(getline(in_cp_number, buffer)) {

		stringstream strin(buffer);
		string id;
		float cp_no;
		strin >> id >> cp_no;

		if (cp_number.count(id) == 0)
			cp_number[id] = cp_no;
		else cerr << "Warning, dup id : " << id << endl;

		count ++;
	}

	in_cp_number.close();
	in_cp_number.clear();

	return count;
}

int _PMDB::Load_Copy_Number_openMP(hash_map <string, float, std_string_hash> & cp_number, int coren) {

	ifstream in_cp_number(DB_cp_number.c_str(), ifstream::in);
	if (!in_cp_number) {
		cerr << "Cannot open copy number table: " << DB_cp_number << ", copy number normalization is disabled" << std::endl;
		exit(1);
	}

	vector<std::string> lines;
	string buffer;
	getline(in_cp_number, buffer); // Skip title line

	// Collect all lines from the file
	while (getline(in_cp_number, buffer)) {
		lines.push_back(buffer);
	}
	in_cp_number.close();

	omp_set_num_threads(coren); // Set the number of threads to use

	#pragma omp parallel
	{
		// Create a local unordered_map for each thread
		hash_map <string, float, std_string_hash> local_cp_number;

		#pragma omp for
		for (size_t i = 0; i < lines.size(); ++i) {
			istringstream strin(lines[i]);
			string id;
			float cp_no;

			strin >> id >> cp_no;

			// Insert into the local table
			if (local_cp_number.count(id) == 0) {
				local_cp_number[id] = cp_no;
			} else {
				#pragma omp critical
				cerr << "Warning, duplicate id: " << id << std::endl;
			}
		}

		// Enter critical section to merge the local table into the shared table
		#pragma omp critical
		{
			for (const auto &pair : local_cp_number) {
				if (cp_number.count(pair.first) == 0) {
					cp_number[pair.first] = pair.second;
				} else {
					cerr << "Warning, duplicate id: " << pair.first << std::endl;
				}
			}
		}
	}

	return cp_number.size();
}

int _PMDB::Load_Nsti(hash_map <string, double, std_string_hash> & nsti_table) {

	ifstream infile(DB_nsti.c_str(), ifstream::in);
	if (!infile) {
		cerr << "Error: Cannot open NSTI table file: " << DB_nsti << endl;
		exit(1);
	}
	int count;
	string buffer;
	while(getline(infile, buffer)) {
		if (buffer[0] == '#') continue;
		stringstream strin(buffer);
		string otu;
		double otu_nsti_value;
		strin >> otu >> otu_nsti_value;
		nsti_table[otu] = otu_nsti_value;
		count ++;
	}
	infile.close();
	infile.clear();
	return nsti_table.size();
}

//private
map <char, vector <string> > _PMDB::DB_config = map <char, vector <string> > ();
string _PMDB::DB_args = "Empty database";
string _PMDB::DB_func_args = "Empty database";
int _PMDB::DB_config_count = Load_config();

int _PMDB::Load_config() {

	//debug
	//cout << "Loading db config" << endl;

	string config_file = Check_Env();
	config_file += "/databases/db.config";

	ifstream in_config(config_file.c_str(), ios::in);
	if (!in_config) {
		cerr << "Error: Cannot open reference database configuration" << endl;
		return 0;
	}

	DB_args = "default is ";
	DB_func_args = "default is ";

	int db_count = 0;
	string buffer;
	while(getline(in_config, buffer)) {

		if (buffer.size() == 0) continue;
		if (buffer[0] == '#') continue;

		vector <string> db_config_entry;
		db_config_entry.push_back(buffer);

		for (int i = 0; i < DB_CONFIG_COUNT - 1; i ++) { // configs

			string a_config;
			while(getline(in_config, a_config)) {

				if (a_config.size() == 0) continue;
				if (a_config[0] == '#') continue;

				db_config_entry.push_back(a_config);

				break;
			}
		}

		if (db_config_entry.size() < DB_CONFIG_COUNT) break; //skip incomplete config

		if (DB_config.count(db_config_entry[0][0]) != 0) continue; //skip dup id;
		DB_config[db_config_entry[0][0]] = db_config_entry;

		if (DB_config.size() > 1) DB_args += ", or ";

		DB_args += db_config_entry[0][0];
		DB_args += " (";
		DB_args += db_config_entry[3];
		DB_args += ")";

		db_count ++;

		if (db_config_entry[7][0] == 'Y' || db_config_entry[7][0] == 'y') { //Is_func
			if (DB_config.size() > 1) DB_func_args += ", or ";
			DB_func_args += db_config_entry[0][0];
			DB_func_args += " (";
			DB_func_args += db_config_entry[3];
			DB_func_args += ")";
		}
	}

	in_config.close();
	in_config.clear();

	//debug
	//cout << "Loading db config complete: " << DB_config.size() << endl;

	return db_count;
}

#endif
