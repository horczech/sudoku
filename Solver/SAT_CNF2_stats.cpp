#include<iostream>
#include<vector>
#include<fstream>
#include<cstdlib>
#include<string>
#include<sstream>
#include<unordered_set>
#include<unordered_map>
#include<iterator>
#include<algorithm>
#include<cstdlib>


typedef std::unordered_set<int> partial_model;
typedef int indexx;
typedef int entry;
typedef int sys_size;
typedef int lit;
typedef std::vector<lit> clause;
typedef std::vector<clause> cnf;

constexpr int STATS=0;

[[gnu::always_inline]] [[gnu::hot]] 
inline lit decode_native(sys_size N, indexx i, indexx j, entry n) {
	return N*N * (i - 1) + N * (j - 1) + n;
}

inline void reconvert(sys_size N, lit c, indexx *i, indexx *j, entry *n) {
    c -= 1;
    int first = c % (N*N);
    *i = (c -first)/(N*N) + 1;
    int second = first % N;
    *j = (first - second)/N + 1;
    *n = second + 1;
}

void set_new_negatives(sys_size N, sys_size n, lit cc, partial_model *Neg);


std::vector<std::string> explode( const std::string &delimiter, const std::string &str)
{
    std::vector<std::string> arr;

    int strleng = str.length();
    int delleng = delimiter.length();
    if (delleng==0)
        return arr;//no change

    int i=0;
    int k=0;
    while( i<strleng )
    {
        int j=0;
        while (i+j<strleng && j<delleng && str[i+j]==delimiter[j])
            j++;
        if (j==delleng)//found delimiter
        {
            arr.push_back(  str.substr(k, i-k) );
            i+=delleng;
            k=i;
        }
        else
        {
            i++;
        }
    }
    arr.push_back(  str.substr(k, i-k) );
    return arr;
}

void initial_literals(sys_size N, sys_size n, partial_model *Pos, partial_model *Neg, std::string file_name) {
    std::vector<std::string> stringArr = explode(" 0\n", file_name);


    for(int i = 0; i < stringArr.size(); i++) {
        std::string number = stringArr[i];
//        std::cout << "PYCO" << "\n";
//        std::cout << number << "\n";
        if(! number.empty()){
            int c = std::stoi(number);

            set_new_negatives(N, n, c, Neg);
            Pos->insert(c);
        }
    }
}

int clauses_2(sys_size N, partial_model *Neg) {
	int count=0;
    lit c1, c2;
    partial_model::iterator end_ptr = Neg->end();
    for (entry z=1; z<=N; z++) {
		for (indexx y=1; y<=N; y++) {
			for (indexx x=1; x<=N-1; x++) {
                c1 = -decode_native(N, x, y, z);
                if (Neg->find(c1) == end_ptr)  {
				    for (indexx i=x+1; i<=N; i++) {
                        c2 = -decode_native(N, i, y, z);
                        if (Neg->find(c2) == end_ptr) {
                            if (STATS) count++;
                            std::cout << c1 << " " << c2 << " 0\n";
                        }
                    }
                }
                c1 = -decode_native(N, y, x, z);
                if (Neg->find(c1) == end_ptr) {
                    for (indexx i=x+1; i<=N;i++) {
                        c2 = -decode_native(N, y, i, z);
                        if (Neg->find(c2) == end_ptr) {
                            if (STATS) count++;
                            std::cout << c1 << " " << c2 << " 0\n";
                        }
                    }
                }
			}
		}
	}
    return count;
}

void set_new_negatives(sys_size N, sys_size n, lit l, partial_model *Neg) {
    int x, y, z;
    reconvert(N, l, &x, &y, &z);
    for (indexx i=1; i<=N; i++) {
        if (i!=z) Neg->insert(-decode_native(N, x, y, i));
        if (i!=y) Neg->insert(-decode_native(N, x, i, z));
        if (i!=x) Neg->insert(-decode_native(N, i, y, z));
    }
    int x_block = (x-1) / n;
    int y_block = (y-1) / n;
    int ind_x, ind_y;
    for (int i=1;i<=n;i++) {
        for (int j=1;j<=n;j++) {
            ind_x = n*x_block+i;
            ind_y = n*y_block+j;
            if ((ind_x != x) || (ind_y != y)) Neg->insert(-decode_native(N, ind_x, ind_y, z));
        }
    }
}

[[gnu::hot]] int analyse_positive_clause(sys_size N, sys_size n, clause *sst, partial_model *Pos, 
        partial_model *Neg, int *count) {
    clause::iterator b = sst->begin();
    clause::iterator e = sst->end();

    if (sst->size() > 1) 
        sst->erase(
            std::remove_if(b, e, [Neg](lit c){return (Neg->find(-c)!=Neg->end());}), e);

    int num = sst->size();
    if (num == 1) {
        lit l = sst->at(0);
        if (Pos->find(l) == Pos->end())  {
            Pos->insert(l);
            set_new_negatives(N, n, l, Neg);
            *count += 1;
        }
    } else {
        if (num == 0) { 
            std::cerr << "=> UNSAT\n";
            std::cout << "1 0\n -1 0\n" << std::endl;
            exit(20);
        }
    }
    return num;
}

int clauses_3(sys_size N, sys_size n, partial_model *Pos, partial_model *Neg) {
	partial_model::iterator end = Neg->end();
    lit c1, c2;
	int count=0;
    for (entry z=1; z<=N; z++) {
	for (indexx i=0; i<n; i++) {
	for (indexx j=0; j<n; j++) {
	for (indexx x=1; x<=n; x++) {
	for (indexx y=1; y<=n; y++) {
        c1 = -decode_native(N, (n*i + x), (n*j+y), z);
        if (Neg->find(c1) == end) {
            for (indexx k=y+1; k<=n; k++) {    
                c2 = -decode_native(N, (n*i + x), (n*j+k), z); 
                if (Neg->find(c2) == end) {
                    if (STATS) count++;
                    std::cout << c1 << " " << c2 << " 0\n";
                }
            }
            for (indexx k=x+1; k<=n; k++) {
	            for (indexx l=1; l<=n; l++) {
                    c2 = -decode_native(N, (n*i + k), (n*j+l), z); 
                    if (Neg->find(c2) == end) {
                        if (STATS) count++;
                        std::cout << c1 << " " << c2 << " 0\n";
                    }
            }}
        }
    }}}}}
    return count;
}

int clauses_5(sys_size N, partial_model *Neg) {
    partial_model::iterator end = Neg->end();
    lit c1, c2;
    int count=0;
	for (indexx x=1; x<=N; x++) {
	    for (indexx y=1; y<=N; y++) {
	        for (entry z=1; z<=N-1; z++) {
                c1 = -decode_native(N, x, y, z); 
                if (Neg->find(c1) == end) {
	                for (indexx i=z+1; i<=N; i++) {
		                c2 = -decode_native(N, x, y, i); 
                        if (Neg->find(c2) == end) {
                            if (STATS) count++;
                            std::cout << c1 << " " << c2 << " 0\n";
                        }
                    }
    }}}}
    return count;
}

int clauses_reduction(sys_size N, sys_size n, partial_model *Pos, partial_model *Neg,
        cnf *clauses) {
    int new_learned=0;
    for (auto& c: *clauses) {
        analyse_positive_clause(N, n, &c, Pos, Neg, &new_learned);
    }
    std::cerr << "\tNew learned singular var: " << new_learned << std::endl;
    return new_learned;
}

void make_positive_clauses(sys_size N, sys_size n, cnf *clauses) {
    for (indexx y=1; y<=N; y++){
        for (indexx z=1; z<=N; z++) {
            clause s1, s2, s3;
            for (indexx x=1; x<=N; x++){
                s1.push_back(decode_native(N, x, y, z)); 
                s2.push_back(decode_native(N, y, x, z));
                s3.push_back(decode_native(N, y, z, x));
            }
            clauses->push_back(s1);
            clauses->push_back(s2);
            clauses->push_back(s3);
        }
    }
	for (entry z=1; z<=N; z++) {
	for (indexx i=0; i<n; i++) {
	for (indexx j=0; j<n; j++) {
        clause c;
	    for (indexx x=1; x<=n; x++) {
	    for (indexx y=1; y<=n; y++) {
            c.push_back(decode_native(N, (n*i + x), (n*j+y), z));
        }}
        clauses->push_back(c);
    }}}
}

inline void print_positive_clauses(cnf *clauses) {
    for (auto& c: *clauses) {
        for (auto& l: c) std::cout << l << " ";
    std::cout << "0\n";
    }
} 

inline void print_initial_literals(sys_size N, partial_model *Pos, partial_model *Neg) {
    for (auto c: *Pos) std::cout << c << " 0\n";
    for (auto c: *Neg) std::cout << c << " 0\n";
}

inline void run_singular_clause_reduction(sys_size N, sys_size n, 
        partial_model *Pos, partial_model *Neg, cnf *clauses) {
    while (clauses_reduction(N, n, Pos, Neg, clauses)) {};
}

inline void remove_single_clauses(cnf *clauses) {
    cnf::iterator b = clauses->begin();
    cnf::iterator e = clauses->end();
    clauses->erase(
            std::remove_if(b, e, [](clause c){return (c.size()==1);}), e);
} 

void stats_of_positive_clauses(cnf *clauses) {
    int size;
    std::unordered_map<int, int> stats;
    std::unordered_map<int, int>::iterator ptr;
    for (auto& c: *clauses) {
        size = c.size();
        ptr = stats.find(size);
        if (ptr == stats.end()) {
            stats.insert(std::make_pair(size, 1));
        } else {
            (ptr->second)++;
        }
    }
    std::ofstream file("size.txt");
    for (auto& p: stats) {
        std::cerr << "\t\tClause size " << p.first << ": " << p.second << std::endl;
        file << p.first << ", " << p.second << std::endl;
    }
    file.close();
}

int main(int argc, char** argv) {
    std::cout << "p cnf 729 852" << "\n";

	sys_size n = atoi(argv[1]);
//    std::cout << "ARGUMENT 1: " << argv[1] <<'\n';
	sys_size N = n*n;
    std::cout.sync_with_stdio(false);
    partial_model Pos, Neg;
    cnf positive_clauses;
    
    int n_pos, n_neg;
    int n_total=N*N*N;


    std::cerr << "Run CNF-Preprocessing:\n";
    std::cerr << "=> Using:\n"
              << "\tOutput statistics: " << STATS << std::endl;
    
//    std::cout << "KOKOT" << '\n';
//    std::cout << "ARGUMENT 2: " << argv[2] << '\n';
//    std::cout << "CURAK" << '\n';

    std::string file_name = argv[2];


    initial_literals(N, n, &Pos, &Neg, file_name);
    n_pos = Pos.size();
    n_neg = Neg.size();
    std::cerr << "=> Reading initial values of model:\n"
              << "\tSystem size: " << N << "x" << N << "\n"
              << "\tp: " << (float)(N*N - n_pos)/((float) N*N) << "\n"
              << "\tTotal number of var: " << n_total << "\n"
              << "\tNumber of positive initial var: "<< n_pos << "\n"
              << "\tNumber of negative initial var: " << n_neg << "\n"
              << "\tFirst modell reduction: " << n_total - n_pos - n_neg << std::endl;
    std::cerr << "=> Singular clause reduction cycle: \n";
    make_positive_clauses(N, n, &positive_clauses);
    run_singular_clause_reduction(N, n, &Pos, &Neg, &positive_clauses);
    remove_single_clauses(&positive_clauses);
    n_pos = Pos.size();
    n_neg = Neg.size();
    int n1 = n_total - n_pos - n_neg;
    std::cerr << "\tSecond model reduction: " << n1 << std::endl;
    if (!n1) std::cerr << "=> SAT -> (effective 2SAT)" << std::endl;
     
   
    std::cerr << "=> Start CNF writing:" << std::endl;
    print_initial_literals(N, &Pos, &Neg);
    int nc_neg=0;
    if (n1) {
        print_positive_clauses(&positive_clauses);
        nc_neg += clauses_3(N, n, &Pos, &Neg);
        nc_neg += clauses_2(N, &Neg);
        nc_neg += clauses_5(N, &Neg);
    }
    
    if (STATS) {
        int n_pos = positive_clauses.size();
        int n_var = n_total - Pos.size() - Neg.size();
        //TODO: Some clauses occure multiple times -> needs to be filtered
        float f = (float) (n_pos + nc_neg) / (float) n_var;
        std::cerr << "\tNumber of positive clauses: " << n_pos << std::endl;
        stats_of_positive_clauses(&positive_clauses);
        std::cerr << "\tNumber of negative clauses: " << nc_neg << "\n"
                  << "\tNumber of all clauses: " << n_pos + nc_neg << "\n"
                  << "\tNumber of variables: " << n_var << "\n"
                  << "\tFrequency: " << f << "\n";
    }
    std::cerr << "=> Passing CNF to SAT solver:\n";
    
    
    return 0;
}
