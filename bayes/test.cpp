#include <vector>
#include <iostream>

int main(int argc, char** argv)
{
    std::vector<int> test({1, 2, 3, 4, 5, 6, 7,8});

    for (size_t i = 0; i < test.size(); ++i) std::cout << test[i] << " ";
    std::cout << std::endl;

    for (const int& a: test) std::cout << a << " ";
    std::cout << std::endl;

    for (const auto& a = test.begin(); a != test.end(); ++a) std::cout << a << " ";
        std::cout << std::endl;

}