#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <math.h>
#include <omp.h>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

/*
this module is to do the random forest analysis on the 
connectome sub-graph networks in order to be able to map
what regions of the brain are affected by certain disorders
so that we can target them with certain drugs
*/ 

// .1D file processor
vector<vector<double>> parse_1D(string filePath, int COLS);
// Get number of colums
int get_cols(string filePath);
// Pearson correlation metric calculator
double pearson_r(vector<double> col_A, vector<double> col_B);
// Write each file from the ASD and Control
void write_row(ofstream &outFile, string file_Path, int target, int COLS);



int main(int argc, char **argv){

    // path to target folder
    string asd_path = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/1D/ASD/Outputs/cpac/filt_global/rois_ho";
    // path to control folder
    string con_path = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/1D/Control/Outputs/cpac/filt_global/rois_ho";
    string data_File = "connectome_data.dat";
    int COLS = -1; 

    vector<pair<string, int>> my_files; // {filepath, target}

    // Add ASD Folder to my_file vector Target = 1
    for(const auto &entry : fs::directory_iterator(asd_path)){
        if (entry.path().extension() == ".1D"){
            my_files.push_back({entry.path().string(), 1});
        }
    }

    // Add Control Folder to my_file vector Target = 0
    for (const auto& entry : fs::directory_iterator(con_path)){
        if (entry.path().extension() == ".1D"){
            my_files.push_back({entry.path().string(), 0});
        }
    }

    // Determine the columns
    if(!my_files.empty()){
        COLS = get_cols(my_files[0].first);
    }

    if(COLS <= 0){
        cerr << "Cannot Determine Number of Columns" << endl;
        return -1;
    }

    ofstream outFile(data_File);    

    // Create the Header for Ranger
    for (int i = 0; i < COLS; i++) {
        for (int j = i + 1; j < COLS; j++) {
            outFile << "ROI" << i << "_ROI" << j << ",";
        }
    }
    outFile << "Diagnosis\n" << endl;

    // Write all rows

    cout << "Processing" << endl;
    for(auto &[path, target] : my_files){
        write_row(outFile, path, target, COLS);
    }
    outFile.close();
    cout << "Successfully Created " << data_File << " For Ranger" << endl;


    string ranger_cmd = "/home/rhonda/Downloads/capstone/ranger/cpp_version/build/ranger " //path to ranger executable
                    "--file connectome_data.dat --depvarname Diagnosis "
                    "--treetype 1 --ntree 1000 --nthreads 4 --impmeasure 1 --verbose";

    cout << "Starting Ranger Random Forest..." << endl;
    int status = system(ranger_cmd.c_str());

    if (status == 0) {
        cout << "Ranger finished successfully. Check ranger_importance.out for results!" << endl;
    }
    return 0;
}

vector<vector<double>> parse_1D(string filePath, int COLS){

    ifstream file(filePath);
    string line;
    if(!file.is_open()){
        cerr << "Error Opening File! Empty Matrix Returned" << endl;
        return {};
    }
    
    // Discard the header line
    getline(file, line);

    vector<vector<double>> res(COLS);

    while(getline(file, line)){
        if(line.empty()) continue;

        istringstream ss(line);
        double val;
        int col_index = 0;
        
        // Append the values to the correct column vector
        while(ss >> val){
            if(col_index < COLS){
                res[col_index].push_back(val);
                col_index++;
            }
        }

    }

    return res;


}

int get_cols(string filePath){

    ifstream file(filePath);
    string line;
    if(!file.is_open() || !getline(file, line)) return -1;
    
    // Determine number of columns
    istringstream header(line);
    string token;
    int COLS = 0;
    while(header >> token){
        COLS++;
    }

    return COLS;
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

void write_row(ofstream &outFile, string file_Path, int target, int COLS){
    vector<vector<double>> matrix = parse_1D(file_Path, COLS);
    if (matrix.size() < COLS) return;

    for (int i = 0; i < COLS; i++) {
        for (int j = i + 1; j < COLS; j++) {
            outFile << pearson_r(matrix[i], matrix[j]) << ",";
        }
    }
    outFile << target << "\n";
}
