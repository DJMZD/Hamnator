#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <iostream>
#include <string>
#include <unordered_map>
#include "node.h"

class DecisionTree {
private:
    Node* root_;
public:
    Node* getRoot() { return root_; }
    DecisionTree(Node* r): root_(r) { }
};

#endif // DECISION_TREE_H
