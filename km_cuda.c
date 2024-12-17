#include <stdio.h>
#include <stdlib.h>

int main() {
    if (system("make run") == -1) {
        perror("Error executing 'make run'");
        return 1;
    }
    
    return 0;
}
