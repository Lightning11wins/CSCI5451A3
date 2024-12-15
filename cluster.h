#ifndef CLUSTER_H_
#define CLUSTER_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// OSX timer includes
#ifdef __MACH__
    #include <mach/mach.h>
    #include <mach/mach_time.h>
#endif

// Really cursed preprocessor directives.
#define index(point, dim) ((point) * num_dims + (dim))
#define parse_int(str) ((int) strtol((str), (char**) NULL, 10))
#define duration(start, end) ((end) - (start))
#define start_timer() const double start = monotonic_seconds()
#define stop_timer() const double end = monotonic_seconds()
#define print_timer() print_time(duration(start, end))

#define convergence_threshold 0.000001
#define CLUSTER_OUTPUT_PATH "clusters.txt"
#define MEDOID_OUTPUT_PATH "medoids.txt"

/**
 * @brief Gets the name of the executible file executing the this program.
 * 
 * @param buffer A buffer into which to save the name.
 * @param buffer_size The size of the buffer. If this is smaller than the
 *                    length of the name, extra characters are chopped off.
 *
 * It is recomended to ensure buffer_size is at least PATH_MAX to avoid
 * characters being chopped off of the file name.
 */
static inline void get_exe_name(char* buffer, size_t buffer_size) {
    char exePath[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);

    if (count == -1) {
        perror("readlink");
        exit(1);
    }
    
    // TODO: Make these statements easier to read (if I have time).
    exePath[count] = '\0'; // Null-terminate the path
    char* exeName = strrchr(exePath, '/'); // Find the last '/'
    if (exeName) {
        exeName++; // Move past the last '/'
    } else {
        exeName = exePath; // If no '/' found, the whole path is the name
    }
    strncpy(buffer, exeName, buffer_size - 1); // Copy the name into the buffer
    buffer[buffer_size - 1] = '\0'; // Ensure null-termination
}

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

// Calculate squared euclidean distance between two points.
// Intended for comparison, not accurate results, since the
// sqrt call has been removed to increase reliability.
__device__ __host__
static inline double distance(const double* point1, const double* point2, int const dims) {
    double sum = 0.0;
    for (int i = 0; i < dims; i++) {
        const double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

static inline void check(const int result, const char* f_name) {
    if (result <= -1) {
        char errorBuffer[BUFSIZ];
        sprintf(errorBuffer, "cluster.c: Fail - %s", f_name);
        perror(errorBuffer);
        while (1) exit(-1);
    }
}
static inline void check_cuda(const cudaError_t result, const char* f_name) {
    if (result != cudaSuccess) {
        fprintf(stderr, "cluster.c - %s failed: %s\n", f_name, cudaGetErrorString(err));
        while (1) exit(-1);
    }
}
#endif