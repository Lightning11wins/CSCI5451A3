#ifndef CLUSTER_H_
#define CLUSTER_H_
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// OSX timer includes
#ifdef __MACH__
    #include <mach/mach.h>
    #include <mach/mach_time.h>
#endif

#define duration(start, end) ((end) - (start))
#define parse_int(str) ((int) strtol((str), (char**) NULL, 10))
#define start_timer() const double start = monotonic_seconds()
#define stop_timer() const double end = monotonic_seconds()
#define print_timer() print_time(duration(start, end))

#define convergence_threshold 0.000001
#define cluster_output_file_name "clusters.txt"
#define medoid_output_file_name "medoids.txt"

/**
* @brief Return the number of seconds since an unspecified time (e.g., Unix
*        epoch). This is accomplished with a high-resolution monotonic timer,
*        suitable for performance timing.
*
* @return The number of seconds.
*/
static inline double monotonic_seconds() {
  #ifdef __MACH__
    // OSX
    const static mach_timebase_info_data_t info;
    static double seconds_per_unit;
    if(seconds_per_unit == 0) {
        mach_timebase_info(&info);
        seconds_per_unit = (info.numer / info.denom) / 1e9;
    }
    return seconds_per_unit * mach_absolute_time();
  #else
    // Linux systems
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  #endif
}

/**
* @brief Output the seconds elapsed while clustering.
*
* @param seconds Seconds spent on k-medoids clustering, excluding IO.
*/
static inline void print_time(double const seconds) {
    printf("k-medoids clustering time: %0.04fs\n", seconds);
}

// Calculate Euclidean distance between two points.
static inline double euclidean_distance(const double point1[], const double point2[], int const dimensions) {
    double sum = 0.0;
    for (int i = dimensions - 1; i >= 0; i--) {
        const double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static inline void check(const int result, const char* functionName) {
    if (result <= -1) {
        char errorBuffer[BUFSIZ];
        sprintf(errorBuffer, "cluster.c: Fail - %s", functionName);
        perror(errorBuffer);
        while (1) exit(-1);
    }
}
#endif