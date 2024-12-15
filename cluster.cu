#include "cluster.h"

// Expects params to be initialized.
static inline double* read_points(char* filename, uint* num_points, uint* num_dims_ptr) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    check(fscanf(file, "%d %d", num_points, num_dims_ptr), "fscanf()");

    uint num_dims = *num_dims_ptr;
    double* points = (double*) malloc(*num_points * num_dims * sizeof(double));
    for (uint point = 0; point < *num_points; point++) {
        for (uint dim = 0; dim < num_dims; dim++) {
            check(fscanf(file, "%lf", &(points[index(point, dim)])), "fscanf()");
        }
    }

    check(fclose(file), "fclose()");

    return points;
}

static inline void write_clusters(uint* point_medoid_ids, uint num_points) {
    FILE *file = fopen(CLUSTER_OUTPUT_PATH, "w");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    for (uint point_id = 0; point_id < num_points; point_id++) {
        fprintf(file, "%d\n", point_medoid_ids[point_id]);
    }

    check(fclose(file), "fclose()");
}

static inline void write_medoids(double* points, uint* medoids, uint num_medoids, uint num_dims) {
    FILE *file = fopen(MEDOID_OUTPUT_PATH, "w");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    fprintf(file, "%d %d", num_medoids, num_dims);
    for (uint id = 0; id < num_medoids; id++) {
        fprintf(file, "\n%.4f", points[index(medoids[id], 0)]);
        for (uint dim = 1; dim < num_dims; dim++) {
            fprintf(file, " %.4f", points[index(medoids[id], dim)]);
        }
    }

    check(fclose(file), "fclose()");
}

static inline void print_points(double* points, uint num_points, uint num_dims) {
    printf("%dx%d\n", num_points, num_dims);
    for (uint i = 0; i < num_points; i++) {
        for (uint j = 0; j < num_dims; j++) {
            printf("%.1f ", points[index(i, j)]);
        }
        printf("\n");
    }
}

__device__ __host__
static inline void get_chunk(uint p, uint i, uint n, uint* start, uint* end) {
    // Slower version with branching, easier to understand.
    uint chunk_size = n / p, extra = n % p;
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
    uint* point_medoid_ids,
    double* points,
    uint* num_points_ptr,
    uint* medoids,
    uint* num_medoids_ptr,
    uint* num_dims_ptr
) {
    // Calculate my point allocation.
    uint p = gridDim.x * blockDim.x;
    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint n = *num_points_ptr;
    uint start, end;
    get_chunk(p, i, n, &start, &end);

    // Itterate through my points.
    for (uint point_id = start; point_id < end; point_id++) {
        double min_distance = INFINITY;
        uint closest_medoid_id = 1234567890;

        // Itterate through medoids to find the closest one.
        for (uint id = 0; id < *num_medoids_ptr; id++) {
            const double* point = points + point_id;
            const double* medoid = points + medoids[id];
            double distance = get_distance(point, medoid, *num_dims_ptr);
            if (distance < min_distance) {
                min_distance = distance;
                closest_medoid_id = id;
            }
        }

        // Update this point to be part of the closest cluster.
        point_medoid_ids[point_id] = closest_medoid_id;
    }
}

__global__
void get_cluster_sizes(
    double* point_cluster_sizes,
    double* points,
    uint* point_medoid_ids,
    uint* num_points_ptr,
    uint* num_dims_ptr
) {
    uint p = gridDim.x * blockDim.x;
    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint n = *num_points_ptr;
    uint start, end;
    get_chunk(p, i, n, &start, &end);
    
    uint num_points = *num_points_ptr;
    uint num_dims = *num_dims_ptr;

    // Calculate the cluster size for each point if that point were the mediod
    for (uint point_id = 0; point_id < num_points; point_id++) {
        double total_distance = 0.0;
        uint point_count = 0;
        for (uint other_point_id = 0; other_point_id < num_points; other_point_id++) {
            if (point_medoid_ids[other_point_id] == point_medoid_ids[point_id]) {
                total_distance += get_distance(points + other_point_id, points + point_id, num_dims);
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
    uint pts, dms;
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
    uint medoids[num_medoids];
    for (uint medoid_id = 0; medoid_id < num_medoids; medoid_id++) {
        medoids[medoid_id] = medoid_id; // Assign an id to each medoid.
    }

    // Define initial data.
    double average_cluster_size = INFINITY;
    uint point_medoid_ids[num_points];

    for (uint iteration = 0; iteration < 20; iteration++) {
        {   // Assign points to medoids.
            uint* gpu_point_medoid_ids;
            double* gpu_points;
            uint* gpu_num_points_ptr;
            uint* gpu_medoids;
            uint* gpu_num_clusters_ptr;
            uint* gpu_num_dims_ptr;

            check_cuda(cudaMalloc((void**) &gpu_point_medoid_ids, num_points * sizeof(uint)),               "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_points,           num_points * num_dims * sizeof(double)), "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_points_ptr,   sizeof(uint)),                            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_medoids,          num_medoids * sizeof(uint)),              "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_clusters_ptr, sizeof(uint)),                            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_dims_ptr,     sizeof(uint)),                            "cudaMalloc");

            check_cuda(cudaMemcpy(gpu_point_medoid_ids, point_medoid_ids,  num_points * sizeof(uint),               TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_points,           points,            num_points * num_dims * sizeof(double), TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_points_ptr,   &num_points,       sizeof(uint),                            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_medoids,          medoids,           num_medoids * sizeof(uint),              TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_clusters_ptr, &num_medoids,      sizeof(uint),                            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_dims_ptr,     &num_dims,         sizeof(uint),                            TO_GPU), "cudaMemcpyTo");

            assign_points_to_clusters<<<num_blocks, num_threads_per_block>>>(
                gpu_point_medoid_ids,
                gpu_points,
                gpu_num_points_ptr,
                gpu_medoids,
                gpu_num_clusters_ptr,
                gpu_num_dims_ptr
            );
            
            check_cuda(cudaMemcpy(point_medoid_ids, gpu_point_medoid_ids, num_points * sizeof(uint), FROM_GPU), "cudaMemcpyFrom");
            
            check_cuda(cudaFree(gpu_point_medoid_ids), "cudaFree");
            check_cuda(cudaFree(gpu_points),           "cudaFree");
            check_cuda(cudaFree(gpu_num_points_ptr),   "cudaFree");
            check_cuda(cudaFree(gpu_medoids),          "cudaFree");
            check_cuda(cudaFree(gpu_num_clusters_ptr), "cudaFree");
            check_cuda(cudaFree(gpu_num_dims_ptr),     "cudaFree");
        }

         // point_cluster_sizes: The size of this cluster if this the point with this id were the medoid.
        double point_cluster_sizes[num_points];
        for (uint i = 0; i < num_points; i++) point_cluster_sizes[i] = INFINITY; // Remove later
        
        {
            double* gpu_point_cluster_sizes;
            double* gpu_points;
            uint* gpu_point_medoid_ids;
            uint* gpu_num_points_ptr;
            uint* gpu_num_dims_ptr;
            
            check_cuda(cudaMalloc((void**) &gpu_point_cluster_sizes, num_points * sizeof(double)),            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_points,              num_points * num_dims * sizeof(double)), "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_point_medoid_ids,    num_points * sizeof(uint)),               "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_points_ptr,      sizeof(uint)),                            "cudaMalloc");
            check_cuda(cudaMalloc((void**) &gpu_num_dims_ptr,        sizeof(uint)),                            "cudaMalloc");

            check_cuda(cudaMemcpy(gpu_point_cluster_sizes, point_cluster_sizes, num_points * sizeof(double),            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_points,              points,              num_points * num_dims * sizeof(double), TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_point_medoid_ids,    point_medoid_ids,    num_points * sizeof(uint),               TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_points_ptr,      &num_points,         sizeof(uint),                            TO_GPU), "cudaMemcpyTo");
            check_cuda(cudaMemcpy(gpu_num_dims_ptr,        &num_dims,           sizeof(uint),                            TO_GPU), "cudaMemcpyTo");
            
            get_cluster_sizes<<<num_blocks, num_threads_per_block>>>(
                gpu_point_cluster_sizes,
                gpu_points,
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
        for (uint point_id = 0; point_id < num_points; point_id++) {
            const uint medoid_id = point_medoid_ids[point_id];
            double size = point_cluster_sizes[point_id];
            if (size < medoid_sizes[medoid_id]) {
                medoid_sizes[medoid_id] = size;
                medoids[medoid_id] = point_id;
            }
        }

        double total_size = 0.0;
        for (uint i = 0; i < num_medoids; i++) total_size += medoid_sizes[i];
        double new_average_cluster_size = total_size / num_medoids;
        double dif = average_cluster_size - new_average_cluster_size;
        printf("Cluster size was %f, but is now %f (dif: %f). %d iterations remaining\n", average_cluster_size, new_average_cluster_size, dif, 20 - iteration);

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
    free(points);

    return 0; // Success
}
