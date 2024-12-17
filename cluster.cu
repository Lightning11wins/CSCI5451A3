#include "cluster.h"

// Expects params to be initialized.
static inline double* read_points(char* filename, uint* num_points_ptr, uint* num_dims_ptr) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        char buff[1024];
        sprintf(buff, "cluster.c: Fail - fopen(%s, \"r\")", filename);
        perror(buff);
        exit(1);
    }

    check(fscanf(file, "%d %d", num_points_ptr, num_dims_ptr), "fscanf()");

    uint num_points = *num_points_ptr, num_dims = *num_dims_ptr;
    double* points = (double*) malloc(num_points * num_dims * sizeof(double));
    for (uint point = 0; point < num_points; point++) {
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
        char buff[1024];
        sprintf(buff, "cluster.c: Fail - fopen(%s, \"r\")", CLUSTER_OUTPUT_PATH);
        perror(buff);
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
        char buff[1024];
        sprintf(buff, "cluster.c: Fail - fopen(%s, \"r\")", MEDOID_OUTPUT_PATH);
        perror(buff);
        exit(1);
    }

    for (uint id = 0; id < num_medoids; id++) {
        fprintf(file, "%.8f", points[index(medoids[id], 0)]);
        for (uint dim = 1; dim < num_dims; dim++) {
            fprintf(file, " %.8f", points[index(medoids[id], dim)]);
        }
        fprintf(file, "\n");
    }

    check(fclose(file), "fclose()");
}

__device__ __host__
static inline void get_chunk(uint p, uint i, uint n, uint* start, uint* end) {
    // Slower version with branching, easier to understand.
    // uint chunk_size = n / p, extra = n % p;
    // *start = i * chunk_size + ((i < extra) ? i : extra);
    // *end = *start + chunk_size + ((i < extra) ? 1 : 0);
    
    // Faster version, less readable.
    uint chunk_size = n / p, extra = n % p;
    uint offset = i * (chunk_size + 1) - (i >= extra) * (i - extra);
    *start = offset;
    *end = offset + chunk_size + (i < extra);
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

    uint num_medoids = *num_medoids_ptr;
    uint num_dims = *num_dims_ptr;

    // Itterate through my points.
    for (uint cur_point_id = start; cur_point_id < end; cur_point_id++) {
        double min_distance = INFINITY;
        uint closest_medoid_id = 1234567890; // Obviously incorrect value, since it should never apear anywhere.

        // Itterate through medoids to find the closest one.
        for (uint medoid_id = 0; medoid_id < num_medoids; medoid_id++) {
            uint mediod_point_id = medoids[medoid_id];

            // If this mediod IS the current point, it's closest by definition.
            // This ensures that all clusters have at least one point at all times.
            // This bug took over an hour to find. Please end my suffering.
            if (mediod_point_id == cur_point_id) {
                closest_medoid_id = medoid_id;
                break;
            }

            double distance = get_distance(get_point(cur_point_id), get_point(mediod_point_id), num_dims);

            if (distance < min_distance) {
                min_distance = distance;
                closest_medoid_id = medoid_id;
            }
        }

        // Update this point to be part of the closest cluster.
        point_medoid_ids[cur_point_id] = closest_medoid_id;
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
    for (uint point_id = start; point_id < end; point_id++) {
        point_cluster_sizes[point_id] = 0.0;
        uint point_count = 0;
        uint medoid_id = point_medoid_ids[point_id];
        for (uint other_point_id = 0; other_point_id < num_points; other_point_id++) {
            if (medoid_id == point_medoid_ids[other_point_id]) {
                point_cluster_sizes[point_id] += get_distance(get_point(point_id), get_point(other_point_id), num_dims);
                point_count++;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc <= 4) {
        fprintf(stderr, "Usage: ./%s <input_file> <num_medoids> <num_blocks> <num_threads_per_block>\n", argv[0]);
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

    // Print stats.
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
    double old_total_size = INFINITY;
    uint point_medoid_ids[num_points];

    // Declare GPU Variables
    uint* gpu_point_medoid_ids;
    double* gpu_points;
    uint* gpu_num_points_ptr;
    uint* gpu_medoids;
    uint* gpu_num_clusters_ptr;
    uint* gpu_num_dims_ptr;
    double* gpu_point_cluster_sizes;
    
    // Malloc GPU Variables
    check_cuda(cudaMalloc((void**) &gpu_point_medoid_ids, num_points * sizeof(uint)),              "cudaMalloc");
    check_cuda(cudaMalloc((void**) &gpu_points,           num_points * num_dims * sizeof(double)), "cudaMalloc");
    check_cuda(cudaMalloc((void**) &gpu_num_points_ptr,   sizeof(uint)),                           "cudaMalloc");
    check_cuda(cudaMalloc((void**) &gpu_medoids,          num_medoids * sizeof(uint)),             "cudaMalloc");
    check_cuda(cudaMalloc((void**) &gpu_num_clusters_ptr, sizeof(uint)),                           "cudaMalloc");
    check_cuda(cudaMalloc((void**) &gpu_num_dims_ptr,     sizeof(uint)),                           "cudaMalloc");
    check_cuda(cudaMalloc((void**) &gpu_point_cluster_sizes, num_points * sizeof(double)),            "cudaMalloc");

    // Update GPU variables that don't change.
    check_cuda(cudaMemcpy(gpu_points,           points,            num_points * num_dims * sizeof(double), TO_GPU), "cudaMemcpyTo");
    check_cuda(cudaMemcpy(gpu_num_points_ptr,   &num_points,       sizeof(uint),                           TO_GPU), "cudaMemcpyTo");
    check_cuda(cudaMemcpy(gpu_num_clusters_ptr, &num_medoids,      sizeof(uint),                           TO_GPU), "cudaMemcpyTo");
    check_cuda(cudaMemcpy(gpu_num_dims_ptr,     &num_dims,         sizeof(uint),                           TO_GPU), "cudaMemcpyTo");

    for (uint iteration = 0; iteration < 20; iteration++) {
        // Update GPU variables that do change.
        check_cuda(cudaMemcpy(gpu_medoids,          medoids,           num_medoids * sizeof(uint),             TO_GPU), "cudaMemcpyTo");

        // Call into GPU
        assign_points_to_clusters<<<num_blocks, num_threads_per_block>>>(
            gpu_point_medoid_ids,
            gpu_points,
            gpu_num_points_ptr,
            gpu_medoids,
            gpu_num_clusters_ptr,
            gpu_num_dims_ptr
        );
        
        // Get GPU Result
        check_cuda(cudaMemcpy(point_medoid_ids, gpu_point_medoid_ids, num_points * sizeof(uint), FROM_GPU), "cudaMemcpyFrom");

        // point_cluster_sizes[i] is the size of point i's cluster if it were the medoid.
        double point_cluster_sizes[num_points];
            
        // Call into GPU
        get_cluster_sizes<<<num_blocks, num_threads_per_block>>>(
            gpu_point_cluster_sizes,
            gpu_points,
            gpu_point_medoid_ids,
            gpu_num_points_ptr,
            gpu_num_dims_ptr
        );
        
        // Get GPU result
        check_cuda(cudaMemcpy(point_cluster_sizes, gpu_point_cluster_sizes, num_points * sizeof(double), FROM_GPU), "cudaMemcpyFrom");

        double cluster_sizes[num_points];
        for (uint point_id = 0; point_id < num_points; point_id++) cluster_sizes[point_id] = 0.0 / 0.0;
        for (uint point_id = 0; point_id < num_points; point_id++) {
            const uint medoid_id = point_medoid_ids[point_id];
            double size = point_cluster_sizes[point_id];
            if (isnan(cluster_sizes[medoid_id]) || size < cluster_sizes[medoid_id]) {
                cluster_sizes[medoid_id] = size;
                medoids[medoid_id] = point_id;
            }
        }

        // Calculate the new size.
        double new_total_size = 0.0;
        for (uint medoid_id = 0; medoid_id < num_medoids; medoid_id++) {
            if (isnan(cluster_sizes[medoid_id])) {
                printf("Medoid %d has no points.\n", medoid_id);
            }
            new_total_size += cluster_sizes[medoid_id];
        }
        double dif = old_total_size - new_total_size;
        printf("Iteration %d - Size: %lf->%lf (dif: %lf).\n", iteration, old_total_size, new_total_size, dif);

        // This should not be possible.
        if (dif < 0) {
            printf("EXISTANCE BRINGS PAIN, PLEASE FREE ME!!\n");
            fflush(stdout);
        }

        // End if we reach the threashold
        if (dif < THRESHOLD) break;

        // Update the size for tracking and logging purposes.
        old_total_size = new_total_size;
    }

    // Free GPU variables    
    check_cuda(cudaFree(gpu_point_medoid_ids), "cudaFree");
    check_cuda(cudaFree(gpu_points),           "cudaFree");
    check_cuda(cudaFree(gpu_num_points_ptr),   "cudaFree");
    check_cuda(cudaFree(gpu_medoids),          "cudaFree");
    check_cuda(cudaFree(gpu_num_clusters_ptr), "cudaFree");
    check_cuda(cudaFree(gpu_num_dims_ptr),     "cudaFree");
    check_cuda(cudaFree(gpu_point_cluster_sizes), "cudaFree");

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
