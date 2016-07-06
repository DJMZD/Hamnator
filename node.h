#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>

struct Node
{
    std::string label_;
    std::string classified_feature_;
    Node* father_;
    std::vector<Node*> answers_;

    Node() { }
    Node(std::string label, std::string cf, Node* father):
        label_(label),
        classified_feature_(cf),
        father_(father) { }
};
#endif // NODE_H
