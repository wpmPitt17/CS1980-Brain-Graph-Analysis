#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace fs = std::filesystem;
namespace py = pybind11;

/*
Written by: Rhonda Ojongmboh
Module: Random Forest analysis on connectome sub-graph networks.
Goal: Map brain regions affected by disorders for drug targeting.
*/ 

// Function Prototypes
std::vector<std::vector<double>> parse_1D(std::string filePath, int COLS);
int get_cols(std::string filePath);
double pearson_r(const std::vector<double>& col_A, const std::vector<double>& col_B);
void write_row(std::ofstream &outFile, std::string file_Path, int target, int COLS);
std::string shell_quote(const std::string& value);
void process_set(const std::string& output_filename, const std::string& asd_dir, const std::string& con_dir);
std::vector<double> read_diagnosis_labels(const std::string& data_file);
std::vector<double> read_prediction_values(const std::string& prediction_file);

struct RFRunResult {
    std::string train_data_file;
    std::string test_data_file;
    std::string forest_file;
    std::string prediction_file;
    std::string ranger_command;
    std::string ranger_predict_command;
    int ranger_exit_code;
    int ranger_predict_exit_code;
    bool ranger_succeeded;
    bool ranger_predict_succeeded;
    double test_accuracy;
    int test_sample_count;
    int test_misclassifications;
    int true_positive;
    int true_negative;
    int false_positive;
    int false_negative;
    double asd_precision;
    double asd_recall;
    double asd_f1;
    double control_precision;
    double control_recall;
    double control_f1;
};

// --- Helper Functions ---

std::vector<std::vector<double>> parse_1D(std::string filePath, int COLS) {
    std::ifstream file(filePath);
    std::string line;
    if (!file.is_open()) return {};
    
    std::getline(file, line); // Skip header
    std::vector<std::vector<double>> res(COLS);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double val;
        int col_index = 0;
        while (ss >> val && col_index < COLS) {
            res[col_index].push_back(val);
            col_index++;
        }
    }
    return res;
}

int get_cols(std::string filePath) {
    std::ifstream file(filePath);
    std::string line;
    if (!file.is_open() || !std::getline(file, line)) return -1;
    std::istringstream header(line);
    std::string token;
    int count = 0;
    while (header >> token) count++;
    return count;
}

double pearson_r(const std::vector<double>& col_A, const std::vector<double>& col_B) {
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

void write_row(std::ofstream &outFile, std::string file_Path, int target, int COLS){
    std::vector<std::vector<double>> matrix = parse_1D(file_Path, COLS);
    if ((int)matrix.size() < COLS) return;

    for (int i = 0; i < COLS; i++) {
        for (int j = i + 1; j < COLS; j++) {
            outFile << pearson_r(matrix[i], matrix[j]) << ",";
        }
    }
    outFile << target << "\n";
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

void process_set(const std::string& output_filename, const std::string& asd_dir, const std::string& con_dir) {
    if (!fs::exists(asd_dir)) {
        throw std::runtime_error("ASD directory does not exist: " + asd_dir);
    }
    if (!fs::exists(con_dir)) {
        throw std::runtime_error("Control directory does not exist: " + con_dir);
    }

    std::vector<std::pair<std::string, int>> files;

    std::cout << "\n--- Processing Folder: " << asd_dir << " ---" << std::endl;
    for (const auto &entry : fs::directory_iterator(asd_dir)) {
        if (entry.path().extension() == ".1D") {
            files.push_back({entry.path().string(), 1});
        }
    }

    std::cout << "--- Processing Folder: " << con_dir << " ---" << std::endl;
    for (const auto &entry : fs::directory_iterator(con_dir)) {
        if (entry.path().extension() == ".1D") {
            files.push_back({entry.path().string(), 0});
        }
    }

    if (files.empty()) {
        throw std::runtime_error("No .1D files found for output file: " + output_filename);
    }

    int COLS = get_cols(files[0].first);
    if (COLS <= 0) {
        throw std::runtime_error("Could not determine ROI column count from: " + files[0].first);
    }

    std::ofstream outFile(output_filename);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open output file for writing: " + output_filename);
    }

    for (int i = 0; i < COLS; i++) {
        for (int j = i + 1; j < COLS; j++) {
            outFile << "ROI" << i << "_ROI" << j << ",";
        }
    }
    outFile << "Diagnosis\n";

    std::cout << "Writing " << files.size() << " subjects to " << output_filename << "..." << std::endl;
    for (auto &[path, target] : files) {
        write_row(outFile, path, target, COLS);
    }
}

std::vector<double> read_diagnosis_labels(const std::string& data_file) {
    std::ifstream input(data_file);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open dataset for label parsing: " + data_file);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Dataset is empty: " + data_file);
    }

    std::vector<double> labels;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        size_t last_comma = line.rfind(',');
        if (last_comma == std::string::npos) {
            throw std::runtime_error("Malformed data row in dataset: " + data_file);
        }

        labels.push_back(std::stod(line.substr(last_comma + 1)));
    }

    return labels;
}

std::vector<double> read_prediction_values(const std::string& prediction_file) {
    std::ifstream input(prediction_file);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open prediction file: " + prediction_file);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Prediction file is empty: " + prediction_file);
    }

    std::vector<double> predictions;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        predictions.push_back(std::stod(line));
    }

    return predictions;
}

RFRunResult run_random_forest(
    const std::string& asd_path_train,
    const std::string& con_path_train,
    const std::string& asd_path_test,
    const std::string& con_path_test,
    const std::string& output_train_file,
    const std::string& output_test_file,
    const std::string& ranger_binary,
    int ntree,
    int mtry,
    int minbucket,
    int splitrule,
    int nthreads,
    bool verbose
) {
    const std::string outprefix = "ranger_out";
    const std::string forest_file = outprefix + ".forest";
    const std::string prediction_file = outprefix + ".prediction";

    process_set(output_train_file, asd_path_train, con_path_train);
    process_set(output_test_file, asd_path_test, con_path_test);

    std::string ranger_cmd = shell_quote(ranger_binary) +
        " --file " + shell_quote(output_train_file) +
        " --depvarname Diagnosis" +
        " --treetype 1" +
        " --ntree " + std::to_string(ntree) +
        " --mtry " + std::to_string(mtry) +
        " --minbucket " + std::to_string(minbucket) +
        " --splitrule " + std::to_string(splitrule) +
        " --impmeasure 1" +
        " --nthreads " + std::to_string(nthreads) +
        " --write" +
        " --outprefix " + shell_quote(outprefix);

    if (verbose) {
        ranger_cmd += " --verbose";
    }

    std::cout << "\nStarting Ranger Training..." << std::endl;
    int train_status = system(ranger_cmd.c_str());

    std::string ranger_predict_cmd = shell_quote(ranger_binary) +
        " --file " + shell_quote(output_test_file) +
        " --predict " + shell_quote(forest_file) +
        " --outprefix " + shell_quote(outprefix) +
        " --nthreads " + std::to_string(nthreads);

    if (verbose) {
        ranger_predict_cmd += " --verbose";
    }

    int predict_status = -1;
    double test_accuracy = 0.0;
    int test_sample_count = 0;
    int test_misclassifications = 0;
    int true_positive = 0;
    int true_negative = 0;
    int false_positive = 0;
    int false_negative = 0;
    double asd_precision = 0.0;
    double asd_recall = 0.0;
    double asd_f1 = 0.0;
    double control_precision = 0.0;
    double control_recall = 0.0;
    double control_f1 = 0.0;

    if (train_status == 0) {
        std::cout << "Starting Ranger Prediction on held-out test set..." << std::endl;
        predict_status = system(ranger_predict_cmd.c_str());

        if (predict_status == 0) {
            std::vector<double> true_labels = read_diagnosis_labels(output_test_file);
            std::vector<double> predicted_labels = read_prediction_values(prediction_file);

            if (true_labels.size() != predicted_labels.size()) {
                throw std::runtime_error(
                    "Prediction count does not match test label count."
                );
            }

            test_sample_count = static_cast<int>(true_labels.size());
            for (size_t i = 0; i < true_labels.size(); ++i) {
                const double actual = true_labels[i];
                const double predicted = predicted_labels[i];

                if (actual != predicted) {
                    ++test_misclassifications;
                }

                if (actual == 1.0 && predicted == 1.0) {
                    ++true_positive;
                } else if (actual == 0.0 && predicted == 0.0) {
                    ++true_negative;
                } else if (actual == 0.0 && predicted == 1.0) {
                    ++false_positive;
                } else if (actual == 1.0 && predicted == 0.0) {
                    ++false_negative;
                }
            }

            if (test_sample_count > 0) {
                test_accuracy = 1.0 - (
                    static_cast<double>(test_misclassifications) /
                    static_cast<double>(test_sample_count)
                );
            }

            const double asd_precision_den = static_cast<double>(true_positive + false_positive);
            const double asd_recall_den = static_cast<double>(true_positive + false_negative);
            const double control_precision_den = static_cast<double>(true_negative + false_negative);
            const double control_recall_den = static_cast<double>(true_negative + false_positive);

            if (asd_precision_den > 0.0) {
                asd_precision = static_cast<double>(true_positive) / asd_precision_den;
            }
            if (asd_recall_den > 0.0) {
                asd_recall = static_cast<double>(true_positive) / asd_recall_den;
            }
            if (asd_precision + asd_recall > 0.0) {
                asd_f1 = 2.0 * asd_precision * asd_recall / (asd_precision + asd_recall);
            }

            if (control_precision_den > 0.0) {
                control_precision = static_cast<double>(true_negative) / control_precision_den;
            }
            if (control_recall_den > 0.0) {
                control_recall = static_cast<double>(true_negative) / control_recall_den;
            }
            if (control_precision + control_recall > 0.0) {
                control_f1 = 2.0 * control_precision * control_recall / (control_precision + control_recall);
            }
        }
    }

    return RFRunResult{
        output_train_file,
        output_test_file,
        forest_file,
        prediction_file,
        ranger_cmd,
        ranger_predict_cmd,
        train_status,
        predict_status,
        train_status == 0,
        predict_status == 0,
        test_accuracy,
        test_sample_count,
        test_misclassifications,
        true_positive,
        true_negative,
        false_positive,
        false_negative,
        asd_precision,
        asd_recall,
        asd_f1,
        control_precision,
        control_recall,
        control_f1
    };
}

int main(int argc, char **argv) {
    RFRunResult result = run_random_forest(
        "./data/ML/Train/asd",
        "./data/ML/Train/control",
        "./data/ML/Test/asd",
        "./data/ML/Test/control",
        "connectome_data_train.dat",
        "connectome_data_test.dat",
        "../ranger/cpp_version/build/ranger",
        3000,
        200,
        5,
        1,
        16,
        true
    );
    if (result.ranger_succeeded) {
        std::cout << "Ranger finished successfully. Check ranger_importance.out for drug target mapping!" << std::endl;
        if (result.ranger_predict_succeeded) {
            std::cout << "Held-out test accuracy: " << result.test_accuracy << std::endl;
            std::cout << "Held-out test ASD precision/recall/F1: "
                      << result.asd_precision << " / "
                      << result.asd_recall << " / "
                      << result.asd_f1 << std::endl;
            std::cout << "Held-out test Control precision/recall/F1: "
                      << result.control_precision << " / "
                      << result.control_recall << " / "
                      << result.control_f1 << std::endl;
        }
        return 0;
    }

    std::cerr << "Ranger failed with exit code " << result.ranger_exit_code << std::endl;
    return result.ranger_exit_code;
}

PYBIND11_MODULE(random_forest_cpp, m) {
    py::class_<RFRunResult>(m, "RFRunResult")
        .def_readonly("train_data_file", &RFRunResult::train_data_file)
        .def_readonly("test_data_file", &RFRunResult::test_data_file)
        .def_readonly("forest_file", &RFRunResult::forest_file)
        .def_readonly("prediction_file", &RFRunResult::prediction_file)
        .def_readonly("ranger_command", &RFRunResult::ranger_command)
        .def_readonly("ranger_predict_command", &RFRunResult::ranger_predict_command)
        .def_readonly("ranger_exit_code", &RFRunResult::ranger_exit_code)
        .def_readonly("ranger_predict_exit_code", &RFRunResult::ranger_predict_exit_code)
        .def_readonly("ranger_succeeded", &RFRunResult::ranger_succeeded)
        .def_readonly("ranger_predict_succeeded", &RFRunResult::ranger_predict_succeeded)
        .def_readonly("test_accuracy", &RFRunResult::test_accuracy)
        .def_readonly("test_sample_count", &RFRunResult::test_sample_count)
        .def_readonly("test_misclassifications", &RFRunResult::test_misclassifications)
        .def_readonly("true_positive", &RFRunResult::true_positive)
        .def_readonly("true_negative", &RFRunResult::true_negative)
        .def_readonly("false_positive", &RFRunResult::false_positive)
        .def_readonly("false_negative", &RFRunResult::false_negative)
        .def_readonly("asd_precision", &RFRunResult::asd_precision)
        .def_readonly("asd_recall", &RFRunResult::asd_recall)
        .def_readonly("asd_f1", &RFRunResult::asd_f1)
        .def_readonly("control_precision", &RFRunResult::control_precision)
        .def_readonly("control_recall", &RFRunResult::control_recall)
        .def_readonly("control_f1", &RFRunResult::control_f1);

    m.def(
        "run_random_forest",
        &run_random_forest,
        py::arg("asd_path_train") = "./data/ML/Train/asd",
        py::arg("con_path_train") = "./data/ML/Train/control",
        py::arg("asd_path_test") = "./data/ML/Test/asd",
        py::arg("con_path_test") = "./data/ML/Test/control",
        py::arg("output_train_file") = "connectome_data_train.dat",
        py::arg("output_test_file") = "connectome_data_test.dat",
        py::arg("ranger_binary") = "../ranger/cpp_version/build/ranger",
        py::arg("ntree") = 3000,
        py::arg("mtry") = 200,
        py::arg("minbucket") = 5,
        py::arg("splitrule") = 1,
        py::arg("nthreads") = 16,
        py::arg("verbose") = true
    );
}
