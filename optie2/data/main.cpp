#include <iostream>
#include <string>
#include <vector>

#include <fstream>
#include <map>

int main(int argc, char* argv[]) {
    std::string filename = argv[1];
    std::fstream file(filename);
    std::string line;
    std::map<std::string, int> word_count;
    std::map<std::string, int> new_word_count;
    // one word per line, count occurences of words
    while (std::getline(file, line)) {
        std::string new_word;
        bool cijfer = true;
        std::string getal;
        for (char& c : line) {
            if (cijfer) {
                if (c == '\t') {
                    cijfer = false;
                    continue;
                }
                getal += c;
            } else {
            // A, G, P, V, L, I, M : 1
            // S, T, C, N, Q : 2
            // K, R, H : 3
            // D, E : 4
            // F, Y, W : 5
            // C : 6 
            if (c == 'A' || c == 'G' || c == 'P' || c == 'V' || c == 'L' || c == 'I' || c == 'M') {
                new_word += '1';
            } else if (c == 'S' || c == 'T' || c == 'C' || c == 'N' || c == 'Q') {
                new_word += '2';
            } else if (c == 'K' || c == 'R' || c == 'H') {
                new_word += '3';
            } else if (c == 'D' || c == 'E') {
                new_word += '4';
            } else if (c == 'F' || c == 'Y' || c == 'W') {
                new_word += '5';
            } else if (c == 'C') {
                new_word += '6';
            } else {
                new_word += c;
            }}
        }

        if (new_word_count.find(new_word) == new_word_count.end()) {
            new_word_count[new_word] = std::stoi(getal);
        } else {
            new_word_count[new_word] = new_word_count[new_word] + std::stoi(getal);
        }
    }
    std::cout << "new word count: " << new_word_count.size() << std::endl;
    // write new counts to file
    std::ofstream new_file("new_" + filename);
    for (auto& word : new_word_count) {
        new_file << word.second << "\t" << word.first << std::endl;
    }
    new_file.close();
    file.close();
    return 0;
}



