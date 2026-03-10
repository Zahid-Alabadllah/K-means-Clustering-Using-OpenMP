#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define UNIT_TEST
#include "../k_means_omp_restarts.c"

// Helper function to create a temporary file with content
void create_temp_file(const char *filename, const char *content) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error creating temp file");
        exit(1);
    }
    fprintf(fp, "%s", content);
    fclose(fp);
}

// Helper function to remove a temporary file
void remove_temp_file(const char *filename) {
    remove(filename);
}

void test_load_csv_valid() {
    printf("Running test_load_csv_valid...\n");
    const char *filename = "test_valid.csv";
    // 8 features
    const char *content =
        "1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0\n"
        "1.1 2.1 3.1 4.1 5.1 6.1 7.1 8.1\n"
        "1.2 2.2 3.2 4.2 5.2 6.2 7.2 8.2\n";

    create_temp_file(filename, content);

    // Clear data to ensure no residue
    memset(data, 0, sizeof(data));

    int n = load_csv(filename, MAX_POINTS);

    assert(n == 3);
    assert(fabs(data[0][0] - 1.0) < 1e-5);
    assert(fabs(data[0][7] - 8.0) < 1e-5);
    assert(fabs(data[2][0] - 1.2) < 1e-5);
    assert(fabs(data[2][7] - 8.2) < 1e-5);

    remove_temp_file(filename);
    printf("test_load_csv_valid passed.\n");
}

void test_load_csv_file_not_found() {
    printf("Running test_load_csv_file_not_found...\n");
    // Ensure we don't accidentally exit(1) but return -1
    int n = load_csv("non_existent_file_12345.csv", MAX_POINTS);
    assert(n == -1);
    printf("test_load_csv_file_not_found passed.\n");
}

void test_load_csv_empty() {
    printf("Running test_load_csv_empty...\n");
    const char *filename = "test_empty.csv";
    create_temp_file(filename, "");

    int n = load_csv(filename, MAX_POINTS);

    assert(n == 0);

    remove_temp_file(filename);
    printf("test_load_csv_empty passed.\n");
}

int main() {
    test_load_csv_valid();
    test_load_csv_file_not_found();
    test_load_csv_empty();

    printf("All tests passed!\n");
    return 0;
}
