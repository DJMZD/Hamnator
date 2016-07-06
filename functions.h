#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <unistd.h>
#include "head.h"
#include "decision-tree.h"
#include "node.h"

namespace functions {
    // parse $data_file_stream into vectors of vectors of strings
    void parseFile(std::ifstream& data_file_stream, StrVecVec& data, StrVecVec& feature_info) {
        std::string buffer1, buffer2;
        StrVec features;
        const char sep = ',';
        // read the first line
        if (getline(data_file_stream, buffer1)) {
            std::stringstream ss(buffer1);
            // split the line by $sep
            while (getline(ss, buffer2, sep)) {
                features.push_back(buffer2);
                feature_info.push_back(features);
                features.clear();
            }
        }
        /* data: [["y","y","y","n","y","y","y","n","大谷　翔平"],
         *        ["n","y","n","y","y","n","n","n","石井　裕也"], ...]
         */
        while (getline(data_file_stream, buffer1)) {
            std::stringstream ss(buffer1);
            // split the line by $sep
            while (getline(ss, buffer2, sep)) {
                features.push_back(buffer2);
            }
            data.push_back(features);
            features.clear();
        }

        /* feature_info: [["25歳以下", "y", "n"],
         *                ["背番号が30より小さい", "y", "n"], ...]
         */
        for (int i = 0; i < data[0].size(); i++) {
            for (int j = 0; j < data.size(); j++) {
                std::string feature = data[j][i];
                if (std::find(feature_info[i].begin(), feature_info[i].end(), feature)
                        == feature_info[i].end()) {
                    feature_info[i].push_back(feature);
                }
            }
        }
    }

    float calculateEntropy(StrVecVec& data, SIMap& feature_table, int column) {
        SIMap feature_count;
        float entropy;
        for (auto p: feature_table) {
            for (auto example: data) {
                if (example[column] == p.first) {
                    std::string ex_class = example[example.size() - 1];
                    feature_count[ex_class]++;
                }
            }
            float tmp_entropy = 0.0;
            for (auto co: feature_count) {
                float pr = float(co.second) / p.second;
                tmp_entropy -= pr * std::log2(pr);
            }
            entropy += float(p.second) / data.size() * tmp_entropy;
            feature_count.clear();
        }
        return entropy;
    }

    // decide the feature(column) to classify by calculate the entropy
    int decideFeatureToClass(StrVecVec& data) {
        float min_entropy = 0x3f3f3f;
        int decided_column = -1;
        // calculate the entropy for each column
        for (int i = 0; i < data[0].size() - 1; i++) {
            // feature_table: {{"n": 19}, {"y": 14}}
            // feature_info: [[""25歳以下"", "n", "y"], ["背番号が30より小さい", "y", "n"], ...]
            // feature_count: {{"大谷　翔平": 1}, {"加藤　貴之": 1}, ...}
            // entropy: 19 / 33 * (- 1 / 19 * log(1 / 19) - 1 / 19 * log(1 / 19) - ...)
            //        + 14 / 33 * (- 1 / 14 * log(1 / 14) - 1 / 14 * log(1 / 14) - ...)
            SIMap feature_table;
            for (int j = 0; j < data.size(); j++) {
                std::string feature = data[j][i];
                if (feature_table.count(feature) == 0) {
                    feature_table[feature] = 1;
                }
                else { feature_table[feature]++; }
            }
            // calculate the entropy
            // TO-IMPROVE
            float entropy = calculateEntropy(data, feature_table, i);
            if (entropy < min_entropy) { min_entropy = entropy; decided_column = i; }
        }
        return decided_column;
    }

    // remove all rows with the label in the column
    // and remove the corresponding column
    StrVecVec pruneData(StrVecVec& data, int column, std::string label) {
        StrVecVec pruned_data;
        for (auto example: data) {
            if (example[column] != label) continue;
            StrVec features = example;
            features.erase(features.begin() + column);
            pruned_data.push_back(features);
        }
        return pruned_data;
    }

    // remove all rows with the label in the column
    // and remove the corresponding column
    StrVecVec pruneInfo(StrVecVec& data, StrVecVec& feature_info, int column, std::string label) {
        StrVecVec pruned_info;

        for (int i = 0; i < data[0].size(); i++) {
            if (i == column) continue;
            StrVec features;
            features.push_back(feature_info[i][0]);
            for (int j = 0; j < data.size(); j++) {
                if (data[j][column] != label) continue;
                std::string feature = data[j][i];
                if (std::find(features.begin(), features.end(), feature)
                        == features.end()) {
                    features.push_back(feature);
                }
            }
            pruned_info.push_back(features);
        }
        return pruned_info;
    }

    void buildDecisionTree(StrVecVec& data, StrVecVec& feature_info, Node* root) {
        // return if the remained data belong to the same class
        if (feature_info[feature_info.size() - 1].size() == 2) {
            root->classified_feature_ = feature_info[feature_info.size() - 1][1];
            return;
        }
        // decide the feature to classify
        int column_to_classify= decideFeatureToClass(data);
        root->classified_feature_ = feature_info[column_to_classify][0];
        // build the decision tree recursively
        for (auto it = feature_info[column_to_classify].begin() + 1;
                  it != feature_info[column_to_classify].end();
                  ++it) {
            std::string label = *it;
            std::string cf = "";
            Node* child_node = new Node(label, cf, root);
            root->answers_.push_back(child_node);
            StrVecVec pruned_data = pruneData(data, column_to_classify, label);
            StrVecVec pruned_info = pruneInfo(data, feature_info, column_to_classify, label);
            buildDecisionTree(pruned_data, pruned_info, child_node);
        }
    }

    void deleteSource(Node* root) {
        if (root->answers_.empty()) { delete root; return; }
        for (auto child_node: root->answers_) {
            deleteSource(child_node);
        }
    }

    // TO-DO
    // view the tree
    void printTree(Node* root) {
        std::cout << "--------------------" << std::endl;
        //if (root->father_)
        { std::cout << "father: " << root->father_ << std::endl; }
        std::cout << "question: " << root->classified_feature_ << std::endl;
        //if (root->label_)
        { std::cout << "label: " << root->label_ << std::endl; }
        if (!root->answers_.empty()) {
            std::cout << "child: " << std::endl;
            for (auto child_node: root->answers_) {
                printTree(child_node);
            }
        }
    }

    std::string produceQuestion(Node* node) {
        std::string q = node->classified_feature_ + "?";
        return q;
    }

    void gameStart(StrVecVec& feature_info, Node* root) {
        auto root_bak = root;
        while (true) {
            root = root_bak;
            std::cout << "Hamnatorへようこそ" << std::endl;
            sleep(2);
            std::cout << "あなたが思い浮かべている日ハムのピッチャーを当てます" << std::endl;
            sleep(4);
            std::cout << "それでは、はじめます" << std::endl;
            sleep(4);
            // produce questions through the decision tree
            while (!root->answers_.empty()) {
                std::cout << produceQuestion(root) << " (y/n)" << std::endl;
                std::string read_str;
                std::cin >> read_str;
                //read_str = "y";
                if (read_str == root->answers_[0]->label_) { root = root->answers_[0]; }
                else if(read_str == root->answers_[1]->label_) { root = root->answers_[1]; }
                     else { std::cout << "yかnを入力してください" << std::endl; }
            }
            std::cout << "あなたが思い浮かべていたのは" << std::endl;
            sleep(2);
            std::cout << root->classified_feature_ << std::endl;
            sleep(2);
            std::cout << "当たりましたか？ (y/n)" << std::endl;
            char c;
            std::cin >> c;
            while (c != 'y' && c != 'n') {
                std::cout << "yかnを入力してください" << std::endl;
                std::cin >> c;
            }
            // learn the failed test data
            switch (c) {
                case 'y':{
                    std::cout << "Hamnatorの勝ち！" << std::endl;
                    break;
                }
                case 'n':{
                    std::cout << "あなたの勝ち！" << std::endl;
                    sleep(2);
                    std::cout << "Hamnatorをよくするために、データの提供をお願いします" << std::endl;
                    sleep(4);
                    std::cout << "以下の問題をyかnで答えてください" << std::endl;
                    sleep(4);
                    std::string new_data = "", new_feature;
                    for (auto features: feature_info) {
                        std::cout << features[0] << "?" << std::endl;
                        std::cin >> new_feature;
                        while (new_feature != "y" && new_feature != "n" && features[0] != "名前") {
                            std::cout << "yかnを入力してください" << std::endl;
                            std::cin >> new_feature;
                        }
                        new_data += new_feature + ",";
                    }
                    // update the data file
                    new_data = new_data.substr(0, new_data.size() - 1);
                    std::string data_file_name = "/Users/yong-zu/OJ/data.txt";
                    std::ofstream data_file_stream;
                    data_file_stream.open(data_file_name, std::ios::out | std::ios::app);
                    data_file_stream << new_data << std::endl;
                    std::cout << "ご協力ありがとうございました" << std::endl;
                    break;
                }
                default: std::cout << "yかnを入力してください" << std::endl;
            }
            sleep(2);
            std::cout << "もう一回遊びますか？ (y/n)" << std::endl;
            std::cin >> c;
            switch (c) {
                case 'y': continue; break;
                case 'n':
                    std::cout << "またね" << std::endl;
                    sleep(2);
                    return;
                default: std::cout << "yかnを入力してください" << std::endl;
            }
        }
    }
}

#endif // FUNCTIONS_H
