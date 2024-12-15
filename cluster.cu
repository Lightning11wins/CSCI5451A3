#include "cluster.h"

// Expects params to be initialized.
static inline double** read_points(char* filename, int* num_points, int* num_dims) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    check(fscanf(file, "%d %d", num_points, num_dims), "fscanf()");

    double** points = (double**) malloc(*num_points * sizeof(double*));
    for (int point = 0; point < *num_points; point++) {
        points[point] = (double*) malloc(*num_dims * sizeof(double));
        for (int dim = 0; dim < *num_dims; dim++) {
            check(fscanf(file, "%lf", &(points[point][dim])), "fscanf()");
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

static inline void write_medoids(double** points, int* medoids, int num_clusters, int num_dims) {
    FILE *file = fopen(MEDOID_OUTPUT_PATH, "w");
    if (file == NULL) {
        perror("cluster.c: Fail - fopen()");
        exit(1);
    }

    fprintf(file, "%d %d", num_clusters, num_dims);
    for (int id = num_clusters - 1; id >= 0; id--) {
        double* medoid = points[medoids[id]];
        fprintf(file, "\n%.4f", medoid[0]);
        for (int dim = 1; dim < num_dims; dim++) {
            fprintf(file, " %.4f", medoid[dim]);
        }
    }

    check(fclose(file), "fclose()");
}

static inline void print_points(double** points, int num_points, int num_dims) {
    printf("%dx%d\n", num_points, num_dims);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_dims; j++) {
            printf("%.1f ", points[i][j]);
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

static inline double get_cluster_size(const int medoid_id, const int* point_cluster_ids, const double** points, const int num_points, int num_dims) {
    double total_distance = 0.0;
    int point_count = 0;
    int cluster_id = point_cluster_ids[medoid_id];
    for (int point_id = 0; point_id < num_points; point_id++) {
        if (point_cluster_ids[point_id] == cluster_id) {
            total_distance += euclidean_distance(points[point_id], points[medoid_id], num_dims);
            point_count++;
        }
    }
    return total_distance / point_count;
}

int main(int argc, char* argv[]) {
    if (argc <= 3) {
        char exe_name[PATH_MAX] = {0};
        get_exe_name(exe_name, PATH_MAX);

        fprintf(stderr, "Usage: ./%s <input_file> <num_clusters> <num_blocks> <num_threads_per_block>\n", exe_name);
        exit(1);
    }

    // Read the file.
    int pts, dms;
    double** points = read_points(argv[1], &pts, &dms);
    const uint num_points = pts,
        num_dimensions = dms,
        num_clusters = parse_int(argv[2]),
        num_blocks = parse_int(argv[3]),
        num_threads_per_block = parse_int(argv[4]);

    // Error checking.
    if (num_points < num_clusters) {
        fprintf(stderr, "FAIL: Cannot cluster %d points into %d clusters\n", num_points, num_clusters);
        exit(1);
    }

    printf("Clustering:\n  num_points: %d\n  num_dimensions: %d\n  num_clusters: %d\n  num_blocks: %d\n  num_threads_per_block: %d\n\n",
        num_points, num_dimensions, num_clusters, num_blocks, num_threads_per_block);

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

    for (int iterations = 20; iterations > 0; iterations--) {
        // Assign points to medoids.
        #pragma omp parallel for schedule(dynamic) default(shared)
        for (int point_id = num_points - 1; point_id >= 0; point_id--) { // Itterate through points.
            const double* point = points[point_id];
            double min_distance = INFINITY;
            int point_cluster_id = -1;
            for (int id = num_clusters - 1; id >= 0; id--) { // Itterate through medoids to find the closest one.
                const double* medoid = points[medoids[id]];
                double distance = euclidean_distance(point, medoid, num_dimensions);
                if (distance < min_distance) {
                    min_distance = distance;
                    point_cluster_id = id;
                }
            }
            point_cluster_ids[point_id] = point_cluster_id;
        }

        // Assign medoids to clusters of points.
        double medoid_sizes[num_clusters];
        for (int i = num_clusters - 1; i >= 0; i--) medoid_sizes[i] = INFINITY;
        #pragma omp parallel for schedule(dynamic) default(shared)
        for (int point_id = num_points - 1; point_id >= 0; point_id--) { // Iterate through each point
            const int cluster_id = point_cluster_ids[point_id];
            double size = get_cluster_size(point_id, point_cluster_ids, (const double**) points, num_points, num_dimensions);
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
    write_medoids(points, medoids, num_clusters, num_dimensions);

    // Cleanup main memory.
    free_points(points, num_points);

    return 0; // Success
}
