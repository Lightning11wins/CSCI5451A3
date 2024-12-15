#include "cluster.h"

// Expects params to be initialized.
static inline double* read_points(char* filename, int* num_points, int* num_dims) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    check(fscanf(file, "%d %d", num_points, num_dims), "fscanf()");

    double* points = (double*) malloc(*num_points * *num_dims * sizeof(double));
    for (int point = 0; point < *num_points; point++) {
        for (int dim = 0; dim < *num_dims; dim++) {
            check(fscanf(file, "%lf", &(points[index(point, dim)])), "fscanf()");
        }
    }

    check(fclose(file), "fclose()");

    return points;
}

static inline void write_clusters(int* point_medoid_ids, int num_points) {
    FILE *file = fopen(CLUSTER_OUTPUT_PATH, "w");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    for (int point_id = 0; point_id < num_points; point_id++) {
        fprintf(file, "%d\n", point_medoid_ids[point_id]);
    }

    check(fclose(file), "fclose()");
}

static inline void write_medoids(double* points, int* medoids, int num_medoids, int num_dims) {
    FILE *file = fopen(MEDOID_OUTPUT_PATH, "w");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    fprintf(file, "%d %d", num_medoids, num_dims);
    for (int id = num_medoids - 1; id >= 0; id--) {
        fprintf(file, "\n%.4f", points[index(0, dim)]);
        for (int dim = 1; dim < num_dims; dim++) {
            fprintf(file, " %.4f", points[index(medoids[id], dim)]);
        }
    }

    check(fclose(file), "fclose()");
}

static inline void print_points(double* points, int num_points, int num_dims) {
    printf("%dx%d\n", num_points, num_dims);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_dims; j++) {
            printf("%.1f ", points[index(i, j)]);
        }
        printf("\n");
    }
}

static inline void free_points(double** points, int num_points) {
    for (int i = 0; i < num_points; i++) {
        free(points[i]);
    }
    free(points);
}

static inline double get_cluster_size(const int medoid_id, const int* point_medoid_ids, const double* points, const int num_points, int num_dims) {
    double total_distance = 0.0;
    int point_count = 0;
    int cluster_id = point_medoid_ids[medoid_id];
    for (int point_id = 0; point_id < num_points; point_id++) {
        if (point_medoid_ids[point_id] == cluster_id) {
            total_distance += distance(points + point_id, points + medoid_id, num_dims);
            point_count++;
        }
    }
    return sqrt(total_distance) / point_count;
}

__device__ __host__
static inline void get_chunk(int p, int i, int n, int* start, int* end) {
    // Slower version with branching, easier to understand.
    int chunk_size = n / p, extra = n % p;
    *start = i * chunk_size + ((i < extra) ? i : extra);
    *end = *start + chunk_size + ((i < extra) ? 1 : 0);
    
    // Faster version, less readable.
    // int chunk_size = n / p, extra = n % p;
    // int offset = i * (chunk_size + 1) - (i >= extra) * (i - extra);
    // *start = offset;
    // *end = offset + chunk_size + (i < extra);
}

__global__
void assign_points_to_clusters(
    int* point_medoid_ids,
    double* points,
    int* num_points_ptr,
    int* medoids,
    int* num_medoids_ptr,
    int* num_dims_ptr,
) {
    // Calculate my point allocation.
    int p = gridDim.x * blockDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int n = *num_points_ptr;
    int start, end;
    get_chunk(p, i, n, &start, &end);

    // Itterate through my points.
    for (int point_id = start; point_id < end; point_id++) {
        double min_distance = INFINITY;
        int closest_medoid_id = -1;

        // Itterate through medoids to find the closest one.
        for (int id = 0; id < *num_medoids_ptr; id++) {
            const double* point = points + point_id;
            const double* medoid = points + medoids[id];
            double distance = distance(point, medoid, *num_dims_ptr);
            if (distance < min_distance) {
                min_distance = distance;
                medoid_cluster_id = id;
            }
        }

        // Update this point to be part of the closest cluster.
        point_medoid_ids[point_id] = medoid_cluster_id;
    }
}

__global__
void get_cluster_sizes(
    double* point_cluster_sizes,
    double* points,
    int* point_medoid_ids,
    int* num_points_ptr,
    int* num_dims_ptr,
) {
    int p = gridDim.x * blockDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int n = *num_points_ptr;
    int start, end;
    get_chunk(p, i, n, &start, &end);
    
    int num_points = *num_points_ptr;
    int num_dims = *num_dims_ptr;

    // Calculate the cluster size for each point if that point were the mediod
    for (int point_id = 0; point_id < num_points; point_id++) {
        double total_distance = 0.0;
        int point_count = 0;
        for (int other_point_id = 0; other_point_id < num_points; other_point_id++) {
            if (point_medoid_ids[other_point_id] == point_medoid_ids[point_id]) {
                total_distance += distance(points + other_point_id, points + point_id, num_dims);
                point_count++;
            }
        }
        point_cluster_sizes[point_id] = total_distance / point_count;
    }
}

int main(int argc, char* argv[]) {
    if (argc <= 4) {
        char exe_name[PATH_MAX] = {0};
        get_exe_name(exe_name, PATH_MAX);

        fprintf(stderr, "Usage: ./%s <input_file> <num_medoids> <num_blocks> <num_threads_per_block>\n", exe_name);
        exit(1);
    }

    // Read the file.
    int pts, dms;
    double* points = read_points(argv[1], &pts, &dms);
    const uint num_points = pts,
        num_dims = dms,
        num_medoids = parse_int(argv[2]),
        num_blocks = parse_int(argv[3]),
        num_threads_per_block = parse_int(argv[4]);

    // Error checking.
    if (num_points < num_medoids) {
        fprintf(stderr, "FAIL: Cannot cluster %d points with only %d medoids\n", num_points, num_medoids);
        exit(1);
    }

    printf("Clustering:\n  num_points: %d\n  num_dims: %d\n  num_medoids: %d\n  num_blocks: %d\n  num_threads_per_block: %d\n\n",
        num_points, num_dims, num_medoids, num_blocks, num_threads_per_block);

    // Start the clock.
    start_timer();

    // Select initial medoids.
    int medoids[num_medoids];
    for (int medoid_id = num_medoids - 1; medoid_id >= 0; medoid_id--) {
        medoids[medoid_id] = medoid_id; // Assign an id to each medoid.
    }

    // Define initial data.
    double average_cluster_size = INFINITY;
    int point_medoid_ids[num_points];

    for (int iteration = 0; iteration < 20; iteration++) {
        {   // Assign points to medoids.
            int* gpu_point_medoid_ids;
            double* gpu_points;
            int* gpu_num_points_ptr;
            int* gpu_medoids;
            int* gpu_num_clusters_ptr;
            int* gpu_num_dims_ptr;

            check_cuda(cudaMalloc((void**) &gpu_point_medoid_ids, num_points * sizeof(int)),               "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_points,           num_points * num_dims * sizeof(double)), "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_points_ptr,   sizeof(int)),                            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_medoids,          num_medoids * sizeof(int)),              "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_clusters_ptr, sizeof(int)),                            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_dims_ptr,     sizeof(int)),                            "cudaMalloc");

            check_cuda(cudaMemcpy(gpu_point_medoid_ids, point_medoid_ids,  num_points * sizeof(int),               TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_points,           points,            num_points * num_dims * sizeof(double), TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_points_ptr,   &num_points,       sizeof(int),                            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_medoids,          medoids,           num_medoids * sizeof(int),              TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_clusters_ptr, &num_medoids,      sizeof(int),                            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_dims_ptr,     &num_dims,         sizeof(int),                            TO_GPU), "cudaMemcpyTo");

            assign_points_to_clusters<<<num_blocks, num_threads_per_block>>>(
                gpu_point_medoid_ids,
                gpu_points,
                gpu_num_points_ptr,
                gpu_medoids,
                gpu_num_clusters_ptr,
                gpu_num_dims_ptr,
            );
            
            check_cuda(cudaMemcpy(point_medoid_ids, gpu_point_medoid_ids, num_points * sizeof(int), FROM_GPU), "cudaMemcpyFrom");
            
            check_cuda(cudaFree(gpu_point_medoid_ids), "cudaFree");
            check_cuda(cudaFree(gpu_points),           "cudaFree");
            check_cuda(cudaFree(gpu_num_points_ptr),   "cudaFree");
            check_cuda(cudaFree(gpu_medoids),          "cudaFree");
            check_cuda(cudaFree(gpu_num_clusters_ptr), "cudaFree");
            check_cuda(cudaFree(gpu_num_dims_ptr),     "cudaFree");
        }

         // point_cluster_sizes: The size of this cluster if this the point with this id were the medoid.
        double point_cluster_sizes[num_points];
        for (int i = 0; i < num_points; i++) point_cluster_sizes[i] = INFINITY; // Remove later
        
        {
            double* gpu_point_cluster_sizes;
            double* gpu_points;
            int* gpu_point_medoid_ids;
            int* gpu_num_points_ptr;
            int* gpu_num_dims_ptr;
            
            check_cuda(cudaMalloc((void**) &gpu_point_cluster_sizes, num_points * sizeof(double)),            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_points,              num_points * num_dims * sizeof(double)), "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_point_medoid_ids,    num_points * sizeof(int)),               "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_points_ptr,      sizeof(int)),                            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_dims_ptr,        sizeof(int)),                            "cudaMalloc");

            check_cuda(cudaMemcpy(gpu_point_cluster_sizes, point_cluster_sizes, num_points * sizeof(double),            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_points,              points,              num_points * num_dims * sizeof(double), TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_point_medoid_ids,    point_medoid_ids,    num_points * sizeof(int),               TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_points_ptr,      &num_points,         sizeof(int),                            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_dims_ptr,        &num_dims,           sizeof(int),                            TO_GPU), "cudaMemcpyTo");
            
            get_cluster_sizes<<<num_blocks, num_threads_per_block>>>(
                gpu_point_cluster_sizes,
                gpu_points,
                gpu_medoids,
                gpu_point_medoid_ids,
                gpu_num_points_ptr,
                gpu_num_dims_ptr
            );
            
            check_cuda(cudaMemcpy(point_cluster_sizes, gpu_point_cluster_sizes, num_medoids * sizeof(double), FROM_GPU), "cudaMemcpyFrom");
            
            check_cuda(cudaFree(gpu_point_cluster_sizes), "cudaFree");
            check_cuda(cudaFree(gpu_points),              "cudaFree");
            check_cuda(cudaFree(gpu_point_medoid_ids),    "cudaFree");
            check_cuda(cudaFree(gpu_num_points_ptr),      "cudaFree");
            check_cuda(cudaFree(gpu_num_dims_ptr),        "cudaFree");
        }

        double medoid_sizes[num_points];
        for (int point_id = 0; point_id < num_points; point_id++) {
            double size = point_cluster_sizes[point_id];
            if (size < medoid_sizes[medoid_id]) {
                medoid_sizes[medoid_id] = size;
                medoids[medoid_id] = point_id;
            }
        }

        double total_size = 0.0;
        for (int i = 0; i < num_medoids; i++) total_size += medoid_sizes[i];
        double new_average_cluster_size = total_size / num_medoids;
        double dif = average_cluster_size - new_average_cluster_size;
        printf("Cluster size was %f, but is now %f (dif: %f). %d iterations remaining\n", average_cluster_size, new_average_cluster_size, dif, 20 - iterations);

        if (dif < 0) {
            // This should not be possible.
            printf("EXISTANCE BRINGS PAIN, PLEASE FREE ME!!\n");
            fflush(stdout);
        }

        if (dif < convergence_threshold) break;
        average_cluster_size = new_average_cluster_size;
    }

    // End the clock and print time.
    stop_timer();
    print_timer();

    // Write output data.
    write_clusters(point_medoid_ids, num_points);
    write_medoids(points, medoids, num_medoids, num_dims);

    // Cleanup main memory.
    free_points(points, num_points);

    return 0; // Success
}
