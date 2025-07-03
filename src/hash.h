// Updated at Feb. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0

#include <ext/hash_set>
#include <ext/hash_map>

#ifndef HASH_H
#define HASH_H

using namespace __gnu_cxx;

struct std_string_hash {
	size_t operator()( const std::string& x ) const {

		return  __gnu_cxx::hash< const char* >()( x.c_str() );
	}
};

#endif
