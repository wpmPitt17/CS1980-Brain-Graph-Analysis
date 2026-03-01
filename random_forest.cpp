#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>;
#include <string>
#include <cstdlib>
#include <math.h>
#include <omp.h>
using namespace std;

// have to figure out how to import ranger
/*
this module is to do the random forest analsum_Bsis on the 
connectome sub-graph networks in order to be able to map
what regions of the brain are affected bsum_B certain disorders
so that we can target them with certain drugs
*/ 

// .1D file processor
vector<vector<double>> parse_1D(string filePath, int num_cols);
// Pearson correlation metric calculator
double pearson_r(vector<double> col_A, vector<double> col_B);


int main(int argc, char **argv){

    // Path to ranger executable
    string ranger_path = "/home/rhonda/Downloads/capstone/ranger/cpp_version/build/ranger";

    /*
        Storing the pre-processed file
        Target = 1 for Autism 0 for Control
    */ 
    string data_File = "connectome_data.dat";




    return 0;
}

vector<vector<double>> parse_1D(string filePath, int num_cols){

    ifstream file(filePath);
    string line;
    if(!file.is_open()){
        cerr << "Error Opening File! Empty Matrix Returned" << endl;
        return {{0}};
    }
    vector<vector<double>> res(num_cols);

    // Discard this header line
    getline(file, line);

    while(getline(file, line)){

        istringstream ss(line);
        double val;
        int col_index = 0;
        
        // Append the values to the correct column vector
        while(ss >> val){
            if(col_index < num_cols){
                res[col_index].push_back(val);
                col_index++;
            }
        }

    }

    return res;


}

//pearson correlation metric
double pearson_r(vector<double> col_A, vector<double> col_B){
    
    int n = col_A.size();
    if (n == 0 || n != col_B.size()) return 0.0;

    double sum_A = 0, sum_B = 0, sum_AB = 0, sum_A2 = 0, sum_B2 = 0;

    for(int i = 0; i < n; i++) { 
        sum_A += col_A[i];
        sum_B += col_B[i];
        sum_AB += col_A[i] * col_B[i];
        sum_A2 += col_A[i] * col_A[i]; 
        sum_B2 += col_B[i] * col_B[i];
    }

    double num = (double)n * sum_AB - (sum_A * sum_B);
    double den = sqrt(((double)n * sum_A2 - (sum_A * sum_A)) * ((double)n * sum_B2 - (sum_B * sum_B)));

    if (den == 0) return 0.0; 
    return num / den;
}