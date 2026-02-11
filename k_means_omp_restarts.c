
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#define MAX_POINTS 1000000
#define FEATURES 8
#define MAX_K 10
#define MAX_RESTARTS 1000
#define MAX_ITER 2000
#define DEFAULT_RESTARTS 100
#define MAX_CLUSTERS 10

// Set to 1 if you want periodic prints (will slow timing experiments)
#define VERBOSE 0

static float data[MAX_POINTS][FEATURES];
static int labels[MAX_POINTS];

static float centroids[MAX_CLUSTERS][FEATURES];
static float new_centroids[MAX_CLUSTERS][FEATURES];
static int counts[MAX_CLUSTERS];

static float best_centroids[MAX_CLUSTERS][FEATURES];

static int load_csv(const char *filename, int max_points)
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
            data[idx][j] = strtof(ptr, &ptr);
        }
        idx++;
    }

    fclose(fp);
    return idx;
}

// srand() should be called once in main
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

static int assign_points_omp(int n_points, int k)
{
    int changes = 0;

#pragma omp parallel for reduction(+ : changes) schedule(static)
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
            changes += 1;
            labels[i] = best_cluster;
        }
    }

    return changes;
}

static void update_centroids_omp(int n_points, int k)
{
    for (int c = 0; c < k; c++)
    {
        counts[c] = 0;
        for (int f = 0; f < FEATURES; f++)
            new_centroids[c][f] = 0.0f;
    }

#pragma omp parallel
    {
        int local_counts[MAX_CLUSTERS] = {0};
        float local_sums[MAX_CLUSTERS][FEATURES];

        for (int c = 0; c < k; c++)
            for (int f = 0; f < FEATURES; f++)
                local_sums[c][f] = 0.0f;

#pragma omp for schedule(static)
        for (int i = 0; i < n_points; i++)
        {
            int c = labels[i];
            local_counts[c]++;
            for (int f = 0; f < FEATURES; f++)
            {
                local_sums[c][f] += data[i][f];
            }
        }

#pragma omp critical
        {
            for (int c = 0; c < k; c++)
            {
                counts[c] += local_counts[c];
                for (int f = 0; f < FEATURES; f++)
                {
                    new_centroids[c][f] += local_sums[c][f];
                }
            }
        }
    }

    for (int c = 0; c < k; c++)
    {
        if (counts[c] == 0)
            continue;
        for (int f = 0; f < FEATURES; f++)
        {
            centroids[c][f] = new_centroids[c][f] / counts[c];
        }
    }
}

static float compute_accuracy_omp(int n_points)
{
    double total = 0.0;

#pragma omp parallel for reduction(+ : total) schedule(static)
    for (int i = 0; i < n_points; i++)
    {
        int c = labels[i];
        total += (double)distance_pt(data[i], centroids[c]);
    }

    return (float)(total / (double)n_points);
}

static void copy_centroids(float dst[MAX_CLUSTERS][FEATURES], float src[MAX_CLUSTERS][FEATURES], int k)
{
    for (int c = 0; c < k; c++)
        for (int f = 0; f < FEATURES; f++)
            dst[c][f] = src[c][f];
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: ./k_means_omp dataset.csv 10 [restarts]\n");
        return 1;
    }

    const char *filename = argv[1];
    int k = atoi(argv[2]);
    if (k <= 0 || k > MAX_CLUSTERS)
    {
        printf("Error: k must be in [1, %d]\n", MAX_CLUSTERS);
        return 1;
    }

    int restarts = (argc >= 4) ? atoi(argv[3]) : DEFAULT_RESTARTS;

    if (k <= 0 || k > MAX_K)
    {
        printf("Error: k must be in [1, %d]\n", MAX_K);
        return 1;
    }

    if (restarts <= 0)
        restarts = DEFAULT_RESTARTS;
    else if (restarts > MAX_RESTARTS)
    {
        printf("Error: restarts must be <= %d\n", MAX_RESTARTS);
        return 1;
    }

    int n_points = load_csv(filename, MAX_POINTS);
    printf("Loaded %d points.\n", n_points);
    printf("k = %d, restarts = %d, max_iter = %d\n", k, restarts, MAX_ITER);

    srand((unsigned)time(NULL));

    double total_A_time = 0.0, total_U_time = 0.0;
    long long A_steps = 0, U_steps = 0;

    float best_acc = FLT_MAX;
    int best_restart = -1;
    int best_iters = -1;

    double t_all0 = omp_get_wtime();

    for (int r = 0; r < restarts; r++)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_points; i++)
            labels[i] = -1;

        init_centroids(k, n_points);

        int iters_done = 0;

        for (int iter = 0; iter < MAX_ITER; iter++)
        {
            double tA0 = omp_get_wtime();
            int changes = assign_points_omp(n_points, k);
            double tA1 = omp_get_wtime();

            double tU0 = omp_get_wtime();
            update_centroids_omp(n_points, k);
            double tU1 = omp_get_wtime();

            total_A_time += (tA1 - tA0);
            total_U_time += (tU1 - tU0);
            A_steps++;
            U_steps++;

            iters_done = iter + 1;

#if VERBOSE
            if (iter % 50 == 0)
            {
                float acc_iter = compute_accuracy_omp(n_points);
                printf("[restart %d] iter %d  acc=%f  changes=%d\n", r, iter, acc_iter, changes);
            }
#endif

            if (changes == 0)
                break;
        }

        float acc = compute_accuracy_omp(n_points);

        if (acc < best_acc)
        {
            best_acc = acc;
            best_restart = r;
            best_iters = iters_done;
            copy_centroids(best_centroids, centroids, k);
        }
    }

    double t_all1 = omp_get_wtime();

    printf("\nBest restart = %d, iterations in best run = %d\n", best_restart, best_iters);
    printf("Best (lowest) accuracy = %f\n", best_acc);

    printf("\nBest centroids:\n");
    for (int c = 0; c < k; c++)
    {
        printf("C%d: ", c);
        for (int f = 0; f < FEATURES; f++)
        {
            printf("%f%s", best_centroids[c][f], (f == FEATURES - 1) ? "" : ", ");
        }
        printf("\n");
    }

    double avg_A = (A_steps > 0) ? (total_A_time / (double)A_steps) : 0.0;
    double avg_U = (U_steps > 0) ? (total_U_time / (double)U_steps) : 0.0;

    printf("\nTiming:\n");
    printf("Total elapsed time (s) = %.6f\n", (t_all1 - t_all0));
    printf("Avg A-step time per iteration (s) = %.9f\n", avg_A);
    printf("Avg U-step time per iteration (s) = %.9f\n", avg_U);

    return 0;
}
