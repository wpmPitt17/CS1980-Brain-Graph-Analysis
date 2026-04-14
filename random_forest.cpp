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
Written by: Rhonda Ojongmboh
Module: Random Forest analysis on connectome sub-graph networks.
Goal: Map brain regions affected by disorders for drug targeting.
*/ 

// Function Prototypes
vector<vector<double>> parse_1D(string filePath, int COLS);
int get_cols(string filePath);
double pearson_r(const vector<double>& col_A, const vector<double>& col_B);
void write_row(ofstream &outFile, string file_Path, int target, int COLS);

int main(int argc, char **argv) {
    // Path Definition
    string asd_path_test = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/ML/Test/asd";
    string con_path_test = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/ML/Test/control";
    
    string asd_path_train = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/ML/Train/asd";
    string con_path_train = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/ML/Train/control";

    string data_File_test = "connectome_data_test.dat";
    string data_File_train = "connectome_data_train.dat";

    // Helper to process a specific set (Train or Test)
    auto process_set = [&](string output_filename, string asd_dir, string con_dir) {
        vector<pair<string, int>> files;
        
        cout << "\n--- Processing Folder: " << asd_dir << " ---" << endl;
        for (const auto &entry : fs::directory_iterator(asd_dir)) {
            if (entry.path().extension() == ".1D") {
                files.push_back({entry.path().string(), 1});
            }
        }

        cout << "--- Processing Folder: " << con_dir << " ---" << endl;
        for (const auto &entry : fs::directory_iterator(con_dir)) {
            if (entry.path().extension() == ".1D") {
                files.push_back({entry.path().string(), 0});
            }
        }

        if(files.empty()) return;

        // Determine columns from the first file found in this set
        int COLS = get_cols(files[0].first);
        if(COLS <= 0) return;

        ofstream outFile(output_filename);
        // Create Header
        for (int i = 0; i < COLS; i++) {
            for (int j = i + 1; j < COLS; j++) {
                outFile << "ROI" << i << "_ROI" << j << ",";
            }
        }
        outFile << "Diagnosis\n";

        // Write Rows
        cout << "Writing " << files.size() << " subjects to " << output_filename << "..." << endl;
        for (auto &[path, target] : files) {
            write_row(outFile, path, target, COLS);
        }
        outFile.close();
    };

    // Generate the Training and Testing data files
    process_set(data_File_train, asd_path_train, con_path_train);
    process_set(data_File_test, asd_path_test, con_path_test);

    // Run Ranger on the Training set 
    // still working on impriving model
    string ranger_cmd = "/home/rhonda/Downloads/capstone/ranger/cpp_version/build/ranger "
                "--file connectome_data_train.dat "
                "--depvarname Diagnosis "
                "--treetype 1 "
                "--ntree 3000 "        
                "--mtry 200 "          
                "--minbucket 5 "  
                "--impmeasure 1 "      
                "--nthreads 16 --verbose";

    cout << "\nStarting Ranger Training..." << endl;
    int status = system(ranger_cmd.c_str());

    if (status == 0) {
        cout << "Ranger finished successfully. Check ranger_importance.out for drug target mapping!" << endl;
    }
    
    return 0;
}

// --- Helper Functions ---

vector<vector<double>> parse_1D(string filePath, int COLS) {
    ifstream file(filePath);
    string line;
    if (!file.is_open()) return {};
    
    getline(file, line); // Skip header
    vector<vector<double>> res(COLS);

    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        double val;
        int col_index = 0;
        while (ss >> val && col_index < COLS) {
            res[col_index].push_back(val);
            col_index++;
        }
    }
    return res;
}

int get_cols(string filePath) {
    ifstream file(filePath);
    string line;
    if (!file.is_open() || !getline(file, line)) return -1;
    istringstream header(line);
    string token;
    int count = 0;
    while (header >> token) count++;
    return count;
}

double pearson_r(const vector<double>& col_A, const vector<double>& col_B) {
    int n = col_A.size();
    if(n == 0 || n != (int)col_B.size()) return 0.0;

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
    return (den == 0) ? 0.0 : num / den;
}

void write_row(ofstream &outFile, string file_Path, int target, int COLS){
    vector<vector<double>> matrix = parse_1D(file_Path, COLS);
    if ((int)matrix.size() < COLS) return;

    for (int i = 0; i < COLS; i++) {
        for (int j = i + 1; j < COLS; j++) {
            outFile << pearson_r(matrix[i], matrix[j]) << ",";
        }
    }
    outFile << target << "\n";
}