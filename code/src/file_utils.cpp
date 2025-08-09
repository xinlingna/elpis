#include "file_utils.h"
#include <iostream>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

std::string getFileNameWithoutBin(const std::string& path) {
    size_t last_slash = path.find_last_of('/');
    size_t last_dot = path.find_last_of('.');
    if (last_slash == std::string::npos) last_slash = -1;
    if (last_dot == std::string::npos || last_dot < last_slash) last_dot = path.length();
    return path.substr(last_slash + 1, last_dot - last_slash - 1);
}

bool deleteDirectory(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "Failed to open directory: " << path << "\n";
        return false;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string fullPath = path + "/" + name;
        struct stat statbuf;
        if (stat(fullPath.c_str(), &statbuf) != 0) {
            std::cerr << "Failed to stat file: " << fullPath << "\n";
            continue;
        }

        if (S_ISDIR(statbuf.st_mode)) {
            // Recursive delete for subdirectory
            if (!deleteDirectory(fullPath)) {
                closedir(dir);
                return false;
            }
        } else {
            // Delete file
            if (unlink(fullPath.c_str()) != 0) {
                std::cerr << "Failed to delete file: " << fullPath << "\n";
                closedir(dir);
                return false;
            }
        }
    }

    closedir(dir);

    // Delete the empty directory
    if (rmdir(path.c_str()) != 0) {
        std::cerr << "Failed to remove directory: " << path << "\n";
        return false;
    }
    return true;
}

