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

static inline void write_clusters(int* point_cluster_ids, int num_points) {
    FILE *file = fopen(CLUSTER_OUTPUT_PATH, "w");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    for (int point_id = 0; point_id < num_points; point_id++) {
        fprintf(file, "%d\n", point_cluster_ids[point_id]);
    }

    check(fclose(file), "fclose()");
}

static inline void write_medoids(double* points, int* medoids, int num_clusters, int num_dims) {
    FILE *file = fopen(MEDOID_OUTPUT_PATH, "w");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    fprintf(file, "%d %d", num_clusters, num_dims);
    for (int id = num_clusters - 1; id >= 0; id--) {
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

static inline double get_cluster_size(const int medoid_id, const int* point_cluster_ids, const double* points, const int num_points, int num_dims) {
    double total_distance = 0.0;
    int point_count = 0;
    int cluster_id = point_cluster_ids[medoid_id];
    for (int point_id = 0; point_id < num_points; point_id++) {
        if (point_cluster_ids[point_id] == cluster_id) {
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
    int* point_cluster_ids,
    double* points,
    int num_points,
    int* medoids,
    int num_clusters,
    int num_dims,
) {
    // Calculate my point allocation.
    int p = gridDim.x * blockDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int n = num_points;
    int start, end;
    get_chunk(p, i, n, &start, &end);

    // Itterate through my points.
    for (int point_id = start; point_id < end; point_id++) {
        double min_distance = INFINITY;
        int closest_cluster_id = -1;

        // Itterate through medoids to find the closest one.
        for (int id = 0; id < num_clusters; id++) {
            const double* point = points + point_id;
            const double* medoid = points + medoids[id];
            double distance = distance(point, medoid, num_dims);
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster_id = id;
            }
        }

        // Update this point to be part of the closest cluster.
        point_cluster_ids[point_id] = closest_cluster_id;
    }
}

int main(int argc, char* argv[]) {
    if (argc <= 4) {
        char exe_name[PATH_MAX] = {0};
        get_exe_name(exe_name, PATH_MAX);

        fprintf(stderr, "Usage: ./%s <input_file> <num_clusters> <num_blocks> <num_threads_per_block>\n", exe_name);
        exit(1);
    }

    // Read the file.
    int pts, dms;
    double* points = read_points(argv[1], &pts, &dms);
    const uint num_points = pts,
        num_dims = dms,
        num_clusters = parse_int(argv[2]),
        num_blocks = parse_int(argv[3]),
        num_threads_per_block = parse_int(argv[4]);

    // Error checking.
    if (num_points < num_clusters) {
        fprintf(stderr, "FAIL: Cannot cluster %d points into %d clusters\n", num_points, num_clusters);
        exit(1);
    }

    printf("Clustering:\n  num_points: %d\n  num_dims: %d\n  num_clusters: %d\n  num_blocks: %d\n  num_threads_per_block: %d\n\n",
        num_points, num_dims, num_clusters, num_blocks, num_threads_per_block);

    // Start the clock.
    start_timer();

    // Select initial medoids.
    int medoids[num_clusters];
    for (int cluster_id = num_clusters - 1; cluster_id >= 0; cluster_id--) {
        medoids[cluster_id] = cluster_id; // Assign an id to each medoid.
    }

    // Define initial data.
    double average_cluster_size = INFINITY;
    int point_cluster_ids[num_points];

    for (int iteration = 0; iteration < 20; iteration++) {
        // Assign points to medoids.
        int* gpu_point_cluster_ids;
        double* gpu_points;
        int gpu_num_points;
        int* gpu_medoids;
        int gum_clusters;
        int gpu_num_dims;

        check_cuda(cudaMalloc((void**) &gpu_point_cluster_ids, num_points * sizeof(int)),            "cudaMalloc");
        check_cuda(cudaMalloc((void**) &gpu_points,            num_points * num_dims * sizeof(int)), "cudaMalloc");
        check_cuda(cudaMalloc((void**) &gpu_num_points,        sizeof(int)),                         "cudaMalloc");
        check_cuda(cudaMalloc((void**) &gpu_medoids,           num_clusters * sizeof(int)),          "cudaMalloc");
        check_cuda(cudaMalloc((void**) &gpu_num_clusters,      sizeof(int)),                         "cudaMalloc");
        check_cuda(cudaMalloc((void**) &gpu_num_dims,          sizeof(int)),                         "cudaMalloc");

        check_cuda(cudaMemcpy(gpu_point_cluster_ids, point_cluster_ids, num_points * sizeof(int),   cudaMemcpyHostToDevice), "cudaMemcpyTo");
        check_cuda(cudaMemcpy(gpu_points,            points, num_points * num_dims * sizeof(int),   cudaMemcpyHostToDevice), "cudaMemcpyTo");
        check_cuda(cudaMemcpy(gpu_num_points,        num_points,        sizeof(int),                cudaMemcpyHostToDevice), "cudaMemcpyTo");
        check_cuda(cudaMemcpy(gpu_medoids,           medoids,           num_clusters * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpyTo");
        check_cuda(cudaMemcpy(gpu_num_clusters,      num_clusters,      sizeof(int),                cudaMemcpyHostToDevice), "cudaMemcpyTo");
        check_cuda(cudaMemcpy(gpu_num_dims,          num_dims,          sizeof(int),                cudaMemcpyHostToDevice), "cudaMemcpyTo");

        assign_points_to_clusters<<<num_blocks, num_threads_per_block>>>(
            gpu_point_cluster_ids,
            gpu_points,
            gpu_num_points,
            gpu_medoids,
            gpu_num_clusters,
            gpu_num_dims,
        );
        
        check_cuda(cudaMemcpy(point_cluster_ids, gpu_point_cluster_ids, num_points * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpyFrom");
        
        check_cuda(cudaFree(gpu_point_cluster_ids), "cudaFree");
        check_cuda(cudaFree(gpu_points),            "cudaFree");
        check_cuda(cudaFree(gpu_num_points),        "cudaFree");
        check_cuda(cudaFree(gpu_medoids),           "cudaFree");
        check_cuda(cudaFree(gum_clusters),          "cudaFree");
        check_cuda(cudaFree(gpu_num_dims),          "cudaFree");

        // Assign medoids to clusters of points.
        double medoid_sizes[num_clusters];
        for (int i = num_clusters - 1; i >= 0; i--) medoid_sizes[i] = INFINITY;
        #pragma omp parallel for schedule(dynamic) default(shared)
        for (int point_id = num_points - 1; point_id >= 0; point_id--) { // Iterate through each point
            const int cluster_id = point_cluster_ids[point_id];
            double size = get_cluster_size(point_id, point_cluster_ids, (const double**) points, num_points, num_dims);
            if (size < medoid_sizes[cluster_id]) { // Check if the point should be the new medoid
                medoid_sizes[cluster_id] = size;
                medoids[cluster_id] = point_id;
            }
        }

        double total_size = 0.0;
        for (int i = num_clusters - 1; i >= 0; i--) total_size += medoid_sizes[i];
        double new_average_cluster_size = total_size / num_clusters;
        double dif = average_cluster_size - new_average_cluster_size;
        printf("Cluster size was %f, but is now %f (dif: %f). %d iterations remaining\n", average_cluster_size, new_average_cluster_size, dif, iterations);

        if (dif < convergence_threshold) break;
        average_cluster_size = new_average_cluster_size;
    }

    // End the clock and print time.
    stop_timer();
    print_timer();

    // Write output data.
    write_clusters(point_cluster_ids, num_points);
    write_medoids(points, medoids, num_clusters, num_dims);

    // Cleanup main memory.
    free_points(points, num_points);

    return 0; // Success
}
