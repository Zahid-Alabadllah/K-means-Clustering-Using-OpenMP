#define TEST_MODE
#include "../k_means_seq_restarts.c"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Helper to check if two floats are close enough
int float_equal(float a, float b, float epsilon) {
    return fabsf(a - b) < epsilon;
}

int main() {
    printf("Running tests for distance_pt...\n");

    // Test Case 1: Zero Distance (Identical Points)
    {
        float p1[FEATURES] = {0};
        float p2[FEATURES] = {0};

        // Initialize with some values
        for(int i=0; i<FEATURES; i++) {
            p1[i] = (float)i;
            p2[i] = (float)i;
        }

        float d = distance_pt(p1, p2);
        assert(d == 0.0f);
        printf("PASS: Zero Distance\n");
    }

    // Test Case 2: Known Distance (3-4-5 Triangle)
    {
        float p1[FEATURES] = {0};
        float p2[FEATURES] = {0};

        // Let's use first two dimensions for 3-4-5
        p1[0] = 0.0f; p1[1] = 0.0f;
        p2[0] = 3.0f; p2[1] = 4.0f;

        // Ensure others are 0
        for(int i=2; i<FEATURES; i++) {
            p1[i] = 0.0f;
            p2[i] = 0.0f;
        }

        float d = distance_pt(p1, p2);
        assert(float_equal(d, 5.0f, 1e-5f));
        printf("PASS: Known Distance (3-4-5)\n");
    }

    // Test Case 3: Negative Coordinates
    {
        float p1[FEATURES] = {0};
        float p2[FEATURES] = {0};

        p1[0] = -1.0f;
        p2[0] = 1.0f;
        // distance is 2.0 along dimension 0

        float d = distance_pt(p1, p2);
        assert(float_equal(d, 2.0f, 1e-5f));
        printf("PASS: Negative Coordinates\n");
    }

    // Test Case 4: Symmetry
    {
        float p1[FEATURES];
        float p2[FEATURES];

        // Use fixed seed for reproducibility
        srand(42);

        for(int i=0; i<FEATURES; i++) {
            p1[i] = (float)rand() / RAND_MAX;
            p2[i] = (float)rand() / RAND_MAX;
        }

        float d1 = distance_pt(p1, p2);
        float d2 = distance_pt(p2, p1);

        assert(float_equal(d1, d2, 1e-6f));
        printf("PASS: Symmetry\n");
    }

    // Test Case 5: All dimensions contribution
    {
         float p1[FEATURES] = {0};
         float p2[FEATURES] = {0};

         // Set each dimension difference to 1.0
         // Sum of squares = FEATURES * 1^2 = FEATURES
         // Distance = sqrt(FEATURES)
         for(int i=0; i<FEATURES; i++) {
             p2[i] = 1.0f;
         }

         float d = distance_pt(p1, p2);
         assert(float_equal(d, sqrtf((float)FEATURES), 1e-5f));
         printf("PASS: All dimensions\n");
    }

    printf("All tests passed!\n");
    return 0;
}
