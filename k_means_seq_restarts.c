
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_POINTS 1000000
#define FEATURES 8
#define MAX_K 10
#define MAX_ITER 2000
#define DEFAULT_RESTARTS 100

// Global arrays (avoid stack issues)
static float data[MAX_POINTS][FEATURES];
static int labels[MAX_POINTS];

static float centroids[MAX_K][FEATURES];
static float new_centroids[MAX_K][FEATURES];
static int counts[MAX_K];

static inline double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// -------------------------------------------
// Load CSV/TSV/whitespace-separated file
// Accepts delimiters: comma ',', space ' ', tab '\t'
// -------------------------------------------
static int load_csv_flexible(const char *filename, int max_points)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("Error opening file");
        exit(1);
    }

    char line[4096];
    int idx = 0;

    while (fgets(line, sizeof(line), fp) && idx < max_points)
    {
        char *ptr = line;

        for (int j = 0; j < FEATURES; j++)
        {
            // Skip leading delimiters
            while (*ptr == ',' || *ptr == ' ' || *ptr == '\t')
                ptr++;

            // Parse float
            char *endptr = NULL;
            data[idx][j] = strtof(ptr, &endptr);

            // If parsing failed, stop with a helpful message
            if (endptr == ptr)
            {
                fprintf(stderr,
                        "Parse error at row %d, feature %d. Offending text starts with: '%.20s'\n",
                        idx, j, ptr);
                fclose(fp);
                exit(1);
            }

            ptr = endptr;

            // Skip trailing delimiters after the number
            while (*ptr == ',' || *ptr == ' ' || *ptr == '\t')
                ptr++;
        }

        idx++;
    }

    fclose(fp);
    return idx;
}

// -------------------------------------------
// Randomly pick initial centroids from dataset
// -------------------------------------------
static void init_centroids(int k, int n_points)
{
    for (int c = 0; c < k; c++)
    {
        int rand_idx = rand() % n_points;
        for (int f = 0; f < FEATURES; f++)
        {
            centroids[c][f] = data[rand_idx][f];
        }
    }
}

// -------------------------------------------
// Euclidean distance
// -------------------------------------------
static inline float distance_pt(const float *a, const float *b)
{
    float sum = 0.0f;
    for (int i = 0; i < FEATURES; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// -------------------------------------------
// Assignment step (sequential)
// -------------------------------------------
static int assign_points_seq(int n_points, int k)
{
    int changes = 0;

    for (int i = 0; i < n_points; i++)
    {
        float best_dist = FLT_MAX;
        int best_cluster = 0;

        for (int c = 0; c < k; c++)
        {
            float d = distance_pt(data[i], centroids[c]);
            if (d < best_dist)
            {
                best_dist = d;
                best_cluster = c;
            }
        }

        if (labels[i] != best_cluster)
        {
            changes++;
            labels[i] = best_cluster;
        }
    }

    return changes;
}

// -------------------------------------------
// Update step (sequential)
// -------------------------------------------
static void update_centroids_seq(int n_points, int k)
{
    // reset accumulators
    for (int c = 0; c < k; c++)
    {
        counts[c] = 0;
        for (int f = 0; f < FEATURES; f++)
            new_centroids[c][f] = 0.0f;
    }

    // accumulate sums
    for (int i = 0; i < n_points; i++)
    {
        int c = labels[i];
        counts[c]++;
        for (int f = 0; f < FEATURES; f++)
        {
            new_centroids[c][f] += data[i][f];
        }
    }

    // compute means
    for (int c = 0; c < k; c++)
    {
        if (counts[c] == 0)
            continue; // leave centroid as-is if empty
        for (int f = 0; f < FEATURES; f++)
        {
            centroids[c][f] = new_centroids[c][f] / counts[c];
        }
    }
}

// -------------------------------------------
// Accuracy = average distance to assigned centroid
// -------------------------------------------
static float compute_accuracy_seq(int n_points)
{
    double total = 0.0;
    for (int i = 0; i < n_points; i++)
    {
        int c = labels[i];
        total += (double)distance_pt(data[i], centroids[c]);
    }
    return (float)(total / (double)n_points);
}

// -------------------------------------------
// Copy centroids
// -------------------------------------------
static void copy_centroids(float dst[MAX_K][FEATURES], float src[MAX_K][FEATURES], int k)
{
    for (int c = 0; c < k; c++)
    {
        for (int f = 0; f < FEATURES; f++)
            dst[c][f] = src[c][f];
    }
}

// -------------------------------------------
// Main
// -------------------------------------------
#ifndef TEST_MODE
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: ./k_means_seq dataset.csv 10 [restarts]\n");
        return 1;
    }

    const char *filename = argv[1];
    int k = atoi(argv[2]);
    int restarts = (argc >= 4) ? atoi(argv[3]) : DEFAULT_RESTARTS;

    if (k <= 0 || k > MAX_K)
    {
        printf("Error: number_of_clusters must be in [1, %d]\n", MAX_K);
        return 1;
    }
    if (restarts <= 0)
    {
        printf("Error: restarts must be positive\n");
        return 1;
    }

    int n_points = load_csv_flexible(filename, MAX_POINTS);
    printf("Loaded %d points.\n", n_points);
    printf("K=%d, Restarts=%d, MaxIter=%d\n", k, restarts, MAX_ITER);

    // Seed RNG once (different initializations each restart)
    srand((unsigned)time(NULL));

    // Track best result across restarts
    float best_centroids[MAX_K][FEATURES];
    float best_accuracy = FLT_MAX;
    int best_restart = -1;

    double total_A = 0.0;
    double total_U = 0.0;
    long long total_iters = 0;

    double t0_total = now_seconds();

    for (int r = 0; r < restarts; r++)
    {
        // reset labels
        for (int i = 0; i < n_points; i++)
            labels[i] = -1;

        // random init centroids
        init_centroids(k, n_points);

        for (int iter = 0; iter < MAX_ITER; iter++)
        {
            double tA0 = now_seconds();
            int changes = assign_points_seq(n_points, k);
            double tA1 = now_seconds();

            double tU0 = now_seconds();
            update_centroids_seq(n_points, k);
            double tU1 = now_seconds();

            total_A += (tA1 - tA0);
            total_U += (tU1 - tU0);
            total_iters++;

            if (changes == 0)
                break; // converged
        }

        float acc = compute_accuracy_seq(n_points);

        if (acc < best_accuracy)
        {
            best_accuracy = acc;
            best_restart = r;
            copy_centroids(best_centroids, centroids, k);
        }
    }

    double t1_total = now_seconds();

    // Print best centroids
    printf("\nBest restart = %d\n", best_restart);
    printf("Best accuracy = %f\n", best_accuracy);
    printf("Best centroids:\n");
    for (int c = 0; c < k; c++)
    {
        printf("C%d: ", c);
        for (int f = 0; f < FEATURES; f++)
        {
            printf("%f", best_centroids[c][f]);
            if (f < FEATURES - 1)
                printf(", ");
        }
        printf("\n");
    }

    // Timing summary
    double total_elapsed = (t1_total - t0_total);
    printf("\nTotal elapsed time (s) = %.6f\n", total_elapsed);

    if (total_iters > 0)
    {
        printf("Avg A-step time per iteration (s) = %.9f\n", total_A / (double)total_iters);
        printf("Avg U-step time per iteration (s) = %.9f\n", total_U / (double)total_iters);
        printf("Total iterations executed (across all restarts) = %lld\n", total_iters);
    }

    return 0;
}
#endif
