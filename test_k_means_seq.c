#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define ENABLE_TESTING
#include "k_means_seq_restarts.c"

// Helper macros for testing
#define ASSERT_FLOAT_EQ(a, b, epsilon) \
    do { \
        if (fabs((a) - (b)) > (epsilon)) { \
            fprintf(stderr, "Assertion failed: %f != %f (epsilon %f) at line %d\n", (float)(a), (float)(b), (float)(epsilon), __LINE__); \
            exit(1); \
        } \
    } while (0)

#define ASSERT_INT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            fprintf(stderr, "Assertion failed: %d != %d at line %d\n", (int)(a), (int)(b), __LINE__); \
            exit(1); \
        } \
    } while (0)

void test_distance_pt() {
    printf("Running test_distance_pt...\n");
    float p1[FEATURES];
    float p2[FEATURES];

    // Zero out
    for(int i=0; i<FEATURES; i++) {
        p1[i] = 0.0f;
        p2[i] = 0.0f;
    }

    // p2 = (1, 0, ...)
    p2[0] = 1.0f;

    float dist = distance_pt(p1, p2);
    ASSERT_FLOAT_EQ(dist, 1.0f, 1e-6);

    // p3 = (1, 1, ...)
    float p3[FEATURES];
    for(int i=0; i<FEATURES; i++) p3[i] = 1.0f;

    // sqrt(8)
    float dist2 = distance_pt(p1, p3);
    ASSERT_FLOAT_EQ(dist2, sqrtf((float)FEATURES), 1e-6);
    printf("test_distance_pt passed.\n");
}

void test_assign_points_seq() {
    printf("Running test_assign_points_seq...\n");

    // Setup data
    // Point 0: near origin (0,0,...)
    for(int f=0; f<FEATURES; f++) data[0][f] = 0.0f;
    // Point 1: far away (10,10,...)
    for(int f=0; f<FEATURES; f++) data[1][f] = 10.0f;

    // Setup centroids
    // Centroid 0: (1,1,...)
    for(int f=0; f<FEATURES; f++) centroids[0][f] = 1.0f;
    // Centroid 1: (9,9,...)
    for(int f=0; f<FEATURES; f++) centroids[1][f] = 9.0f;

    // Initially all labels are -1
    labels[0] = -1;
    labels[1] = -1;

    int changes = assign_points_seq(2, 2);

    // Point 0 should be assigned to Centroid 0
    ASSERT_INT_EQ(labels[0], 0);
    // Point 1 should be assigned to Centroid 1
    ASSERT_INT_EQ(labels[1], 1);

    ASSERT_INT_EQ(changes, 2);

    // Run again, should be 0 changes
    changes = assign_points_seq(2, 2);
    ASSERT_INT_EQ(changes, 0);

    printf("test_assign_points_seq passed.\n");
}

void test_update_centroids_seq() {
    printf("Running test_update_centroids_seq...\n");

    // Setup data
    // P0: (2, 2, ...)
    for(int f=0; f<FEATURES; f++) data[0][f] = 2.0f;
    // P1: (4, 4, ...)
    for(int f=0; f<FEATURES; f++) data[1][f] = 4.0f;
    // P2: (10, 10, ...)
    for(int f=0; f<FEATURES; f++) data[2][f] = 10.0f;

    // Assign P0 and P1 to C0, P2 to C1
    labels[0] = 0;
    labels[1] = 0;
    labels[2] = 1;

    update_centroids_seq(3, 2);

    // Check C0: average of P0 and P1 -> (3, 3, ...)
    for(int f=0; f<FEATURES; f++) {
        ASSERT_FLOAT_EQ(centroids[0][f], 3.0f, 1e-6);
    }
    // Check C1: P2 -> (10, 10, ...)
    for(int f=0; f<FEATURES; f++) {
        ASSERT_FLOAT_EQ(centroids[1][f], 10.0f, 1e-6);
    }

    printf("test_update_centroids_seq passed.\n");
}

void test_compute_accuracy_seq() {
    printf("Running test_compute_accuracy_seq...\n");

    // Setup data
    for(int f=0; f<FEATURES; f++) data[0][f] = 0.0f;

    // Setup centroid (1,1,...)
    for(int f=0; f<FEATURES; f++) centroids[0][f] = 1.0f; // dist is sqrt(8)

    labels[0] = 0;

    float acc = compute_accuracy_seq(1);
    ASSERT_FLOAT_EQ(acc, sqrtf((float)FEATURES), 1e-6);

    printf("test_compute_accuracy_seq passed.\n");
}

void test_load_csv_flexible() {
    printf("Running test_load_csv_flexible...\n");

    const char *filename = "test_data_temp.csv";
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to create temp file");
        exit(1);
    }

    // Write 2 rows with FEATURES features each
    // Row 0: 1.0, 2.0, ...
    for(int i=0; i<FEATURES; i++) {
        fprintf(fp, "%.1f%c", (float)(i+1), (i==FEATURES-1) ? '\n' : ',');
    }
    // Row 1: 10.0 20.0 ... (space separated)
    for(int i=0; i<FEATURES; i++) {
        fprintf(fp, "%.1f%c", (float)((i+1)*10), (i==FEATURES-1) ? '\n' : ' ');
    }
    fclose(fp);

    int n = load_csv_flexible(filename, 10);
    ASSERT_INT_EQ(n, 2);

    // Check data
    for(int i=0; i<FEATURES; i++) {
        ASSERT_FLOAT_EQ(data[0][i], (float)(i+1), 1e-6);
        ASSERT_FLOAT_EQ(data[1][i], (float)((i+1)*10), 1e-6);
    }

    remove(filename);
    printf("test_load_csv_flexible passed.\n");
}

int main() {
    test_distance_pt();
    test_assign_points_seq();
    test_update_centroids_seq();
    test_compute_accuracy_seq();
    test_load_csv_flexible();

    printf("All tests passed!\n");
    return 0;
}
