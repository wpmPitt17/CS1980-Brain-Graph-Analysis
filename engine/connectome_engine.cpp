// connectome_engine.cpp
// Author: Rhonda Ojongmboh
//
// pybind11 wrapper 
// Import this as: import connectome_engine
//
// Build instructions in CMakeLists.txt

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <map>
#include <filesystem>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace fs = std::filesystem;

// ------------------------------------------------------------
// Core functions (identical logic to random_forest.cpp)
// ------------------------------------------------------------

int get_cols(const std::string &filePath) {
    std::ifstream file(filePath);
    std::string line;
    if (!file.is_open() || !std::getline(file, line)) return -1;
    std::istringstream header(line);
    std::string token;
    int COLS = 0;
    while (header >> token) COLS++;
    return COLS;
}

std::vector<std::vector<double>> parse_1D(const std::string &filePath, int COLS) {
    std::ifstream file(filePath);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Error Opening File: " << filePath << std::endl;
        return {};
    }
    std::getline(file, line); // discard header
    std::vector<std::vector<double>> res(COLS);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double val;
        int col_index = 0;
        while (ss >> val) {
            if (col_index < COLS) {
                res[col_index].push_back(val);
                col_index++;
            }
        }
    }
    return res;
}

double pearson_r(const std::vector<double> &col_A, const std::vector<double> &col_B) {
    int n = col_A.size();
    if (n == 0 || n != (int)col_B.size()) return 0.0;
    double sum_A = 0, sum_B = 0, sum_AB = 0, sum_A2 = 0, sum_B2 = 0;
    for (int i = 0; i < n; i++) {
        sum_A  += col_A[i];
        sum_B  += col_B[i];
        sum_AB += col_A[i] * col_B[i];
        sum_A2 += col_A[i] * col_A[i];
        sum_B2 += col_B[i] * col_B[i];
    }
    double num = (double)n * sum_AB - (sum_A * sum_B);
    double den = std::sqrt(((double)n * sum_A2 - sum_A * sum_A) *
                           ((double)n * sum_B2 - sum_B * sum_B));
    if (den == 0) return 0.0;
    return num / den;
}

bool write_row(std::ofstream &outFile, const std::string &filePath, int target, int COLS) {
    std::vector<std::vector<double>> matrix = parse_1D(filePath, COLS);
    if ((int)matrix.size() < COLS) return false;
    for (int i = 0; i < COLS; i++) {
        for (int j = i + 1; j < COLS; j++) {
            outFile << pearson_r(matrix[i], matrix[j]) << ",";
        }
    }
    outFile << target << "\n";
    return true;
}

// ------------------------------------------------------------
// Result struct -- plain data, pybind11 exposes as Python object
// Avoids py::dict outside module scope which causes the errors
// ------------------------------------------------------------
struct RunResult {
    std::string data_file;
    int subjects_written;
    int ranger_status;
    bool success;
};

// ------------------------------------------------------------
// Python-callable run() function
// ------------------------------------------------------------
RunResult run(
    const std::string &asd_path,
    const std::string &con_path,
    const std::string &ranger_path,
    const std::string &data_file,
    int ntree,
    int nthreads,
    bool verbose
) {
    std::vector<std::pair<std::string, int>> my_files;

    if (!fs::exists(asd_path))
        throw std::runtime_error("ASD path does not exist: " + asd_path);
    for (const auto &entry : fs::directory_iterator(asd_path))
        if (entry.path().extension() == ".1D")
            my_files.push_back({entry.path().string(), 1});

    if (!fs::exists(con_path))
        throw std::runtime_error("Control path does not exist: " + con_path);
    for (const auto &entry : fs::directory_iterator(con_path))
        if (entry.path().extension() == ".1D")
            my_files.push_back({entry.path().string(), 0});

    if (my_files.empty())
        throw std::runtime_error("No .1D files found in provided paths.");

    int COLS = get_cols(my_files[0].first);
    if (COLS <= 0)
        throw std::runtime_error("Cannot determine number of columns from: " + my_files[0].first);

    if (verbose)
        std::cout << "COLS: " << COLS << " | Files: " << my_files.size() << std::endl;

    std::ofstream outFile(data_file);
    for (int i = 0; i < COLS; i++)
        for (int j = i + 1; j < COLS; j++)
            outFile << "ROI" << i << "_ROI" << j << ",";
    outFile << "Diagnosis\n" << std::endl;

    if (verbose) std::cout << "Processing subjects..." << std::endl;
    int written = 0;
    for (auto &[path, target] : my_files)
        if (write_row(outFile, path, target, COLS)) written++;
    outFile.close();

    if (verbose)
        std::cout << "Created " << data_file << " (" << written << " subjects)" << std::endl;

    std::string ranger_cmd = ranger_path
        + " --file " + data_file
        + " --depvarname Diagnosis"
        + " --treetype 1"
        + " --ntree "    + std::to_string(ntree)
        + " --nthreads " + std::to_string(nthreads)
        + " --impmeasure 1"
        + (verbose ? " --verbose" : "");

    if (verbose) std::cout << "Running Ranger..." << std::endl;
    int status = system(ranger_cmd.c_str());

    return RunResult{data_file, written, status, status == 0};
}

// ------------------------------------------------------------
// pybind11 module definition
// ------------------------------------------------------------
PYBIND11_MODULE(connectome_engine, m) {
    m.doc() = "C++ connectome engine: builds correlation feature matrix and runs Ranger.";

    // Expose RunResult as a Python class with readable attributes
    py::class_<RunResult>(m, "RunResult")
        .def_readonly("data_file",        &RunResult::data_file)
        .def_readonly("subjects_written", &RunResult::subjects_written)
        .def_readonly("ranger_status",    &RunResult::ranger_status)
        .def_readonly("success",          &RunResult::success)
        .def("__repr__", [](const RunResult &r) {
            return "<RunResult data_file='" + r.data_file
                + "' subjects_written=" + std::to_string(r.subjects_written)
                + " success=" + (r.success ? "True" : "False") + ">";
        });

    m.def("run", &run,
        py::arg("asd_path"),
        py::arg("con_path"),
        py::arg("ranger_path"),
        py::arg("data_file")  = "connectome_data.dat",
        py::arg("ntree")      = 1000,
        py::arg("nthreads")   = 4,
        py::arg("verbose")    = true,
        R"doc(
        Run the full connectome pipeline.

        Args:
            asd_path    : Path to folder of ASD .1D files
            con_path    : Path to folder of Control .1D files
            ranger_path : Path to the Ranger executable
            data_file   : Output feature matrix filename (default: connectome_data.dat)
            ntree       : Number of trees (default: 1000)
            nthreads    : Number of threads (default: 4)
            verbose     : Print progress (default: True)

        Returns:
            RunResult with fields: data_file, subjects_written, ranger_status, success
        )doc"
    );

    m.def("pearson_r", &pearson_r,
        py::arg("a"), py::arg("b"),
        "Compute Pearson correlation between two lists of doubles."
    );

    m.def("parse_1D", &parse_1D,
        py::arg("filepath"), py::arg("cols"),
        "Parse a .1D fMRI file into a list of column vectors."
    );
}