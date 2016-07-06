#include <iostream>
#include <fstream>
#include <sstream>
#include "head.h"
#include "node.h"
#include "functions.h"
#include "decision-tree.h"

using namespace std;
using namespace functions;

int main(int argc, const char* argv[]) {
    Node* root = new Node();
    DecisionTree dt(root);

    string data_file_name = "data.txt";
    ifstream data_file_stream;
    data_file_stream.open(data_file_name, ios::in);

    if (!data_file_stream) {
        cerr << "Data file not found!" << endl;
        exit(1);
    }

    /* data: [["y","y","y","n","y","y","y","n","大谷　翔平"],
     *        ["n","y","n","y","y","n","n","n","石井　裕也"], ...]
     */
    /* feature_info: [["25歳以下", "y", "n"],
     *                ["背番号が30より小さい", "y", "n"], ...]
     */
    StrVecVec data, feature_info;
    // parse $data_file_stream into $data and $feature_info
    parseFile(data_file_stream, data, feature_info);
    buildDecisionTree(data, feature_info, dt.getRoot());
    //printTree(root);
    gameStart(feature_info, dt.getRoot());
    deleteSource(dt.getRoot());
}
