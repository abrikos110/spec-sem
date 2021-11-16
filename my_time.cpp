#ifndef MY_TIME_CPP
#define MY_TIME_CPP

#include "my_time.h"

double time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        / 1e9;
}

void sleep_ms(unsigned long long t) {
    std::this_thread::sleep_for(std::chrono::milliseconds(t));
}


#endif
