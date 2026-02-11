#define TEST_MODE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

// Include the source file to test static functions
#include "../k_means_omp_restarts.c"

// Helper functions for assertions
void assert_float_eq(float expected, float actual, float epsilon, const char *msg) {
    if (fabs(expected - actual) > epsilon) {
        printf("FAIL: %s. Expected %f, got %f\n", msg, expected, actual);
        exit(1);
    }
}

// Test function declarations
void test_update_centroids_basic();
void test_update_centroids_empty_cluster();
void test_update_centroids_single_point();
void test_update_centroids_parallel_correctness();

int main() {
    printf("Running tests...\n");

    test_update_centroids_basic();
    printf("test_update_centroids_basic passed\n");

    test_update_centroids_empty_cluster();
    printf("test_update_centroids_empty_cluster passed\n");

    test_update_centroids_single_point();
    printf("test_update_centroids_single_point passed\n");

    test_update_centroids_parallel_correctness();
    printf("test_update_centroids_parallel_correctness passed\n");

    printf("All tests passed!\n");
    return 0;
}

// Implementations of test cases
void test_update_centroids_basic() {
    // Setup
    int n_points = 4;
    int k = 2;

    // Clear data and labels
    memset(data, 0, sizeof(data));
    memset(labels, 0, sizeof(labels));
    memset(centroids, 0, sizeof(centroids));

    // Cluster 0: (1,1,...), (2,2,...) -> Centroid (1.5, 1.5, ...)
    for (int f = 0; f < FEATURES; f++) {
        data[0][f] = 1.0f;
        data[1][f] = 2.0f;
    }
    labels[0] = 0;
    labels[1] = 0;

    // Cluster 1: (10,10,...), (12,12,...) -> Centroid (11, 11, ...)
    for (int f = 0; f < FEATURES; f++) {
        data[2][f] = 10.0f;
        data[3][f] = 12.0f;
    }
    labels[2] = 1;
    labels[3] = 1;

    update_centroids_omp(n_points, k);

    for (int f = 0; f < FEATURES; f++) {
        assert_float_eq(1.5f, centroids[0][f], 0.0001f, "Cluster 0 centroid mismatch");
        assert_float_eq(11.0f, centroids[1][f], 0.0001f, "Cluster 1 centroid mismatch");
    }
}

void test_update_centroids_empty_cluster() {
    // Setup
    int n_points = 2;
    int k = 2;

    memset(data, 0, sizeof(data));
    memset(labels, 0, sizeof(labels));
    memset(centroids, 0, sizeof(centroids));

    // Preset centroid 1 to some value to ensure it doesn't change
    for(int f=0; f<FEATURES; f++) centroids[1][f] = 999.0f;

    // All points to cluster 0
    for (int f = 0; f < FEATURES; f++) {
        data[0][f] = 5.0f;
        data[1][f] = 7.0f;
    }
    labels[0] = 0;
    labels[1] = 0;

    update_centroids_omp(n_points, k);

    for (int f = 0; f < FEATURES; f++) {
        assert_float_eq(6.0f, centroids[0][f], 0.0001f, "Cluster 0 centroid mismatch");
        assert_float_eq(999.0f, centroids[1][f], 0.0001f, "Cluster 1 should remain unchanged");
    }
}

void test_update_centroids_single_point() {
    int n_points = 1;
    int k = 1;

    memset(data, 0, sizeof(data));
    memset(labels, 0, sizeof(labels));
    memset(centroids, 0, sizeof(centroids));

    for(int f=0; f<FEATURES; f++) data[0][f] = 3.14f;
    labels[0] = 0;

    update_centroids_omp(n_points, k);

    for(int f=0; f<FEATURES; f++) {
        assert_float_eq(3.14f, centroids[0][f], 0.0001f, "Centroid should match single point");
    }
}

void test_update_centroids_parallel_correctness() {
    // This test uses enough points to hopefully trigger parallel chunks
    int n_points = 1000;
    int k = 2;

    memset(data, 0, sizeof(data));
    memset(labels, 0, sizeof(labels));

    // Cluster 0: points with value 2.0
    // Cluster 1: points with value 4.0
    for(int i=0; i<n_points; i++) {
        if (i % 2 == 0) {
            labels[i] = 0;
            for(int f=0; f<FEATURES; f++) data[i][f] = 2.0f;
        } else {
            labels[i] = 1;
            for(int f=0; f<FEATURES; f++) data[i][f] = 4.0f;
        }
    }

    update_centroids_omp(n_points, k);

    for(int f=0; f<FEATURES; f++) {
        assert_float_eq(2.0f, centroids[0][f], 0.0001f, "Cluster 0 parallel check");
        assert_float_eq(4.0f, centroids[1][f], 0.0001f, "Cluster 1 parallel check");
    }
}
