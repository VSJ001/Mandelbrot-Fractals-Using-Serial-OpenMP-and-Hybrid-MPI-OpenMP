#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

int WIDTH = 1920;
int HEIGHT = 1080;
int MAX_ITER = 1000;

int mandelbrot(double real, double imag, int max_iter) {
    double z_real = 0.0, z_imag = 0.0;
    int iter = 0;
    while (z_real * z_real + z_imag * z_imag <= 4.0 && iter < max_iter) {
        double temp = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp;
        iter++;
    }
    return iter;
}

void write_ppm(const char *filename, unsigned char *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(image, 1, 3 * width * height, fp);
    fclose(fp);
}

void log_result(const char *mode, const char *filename, double duration, int threads, int ranks) {
    FILE *log = fopen("results_log.txt", "a");
    fprintf(log, "[%s] File: %s | Size: %dx%d | Iter: %d | Time: %.3f sec | Threads: %d | MPI Ranks: %d\n",
            mode, filename, WIDTH, HEIGHT, MAX_ITER, duration, threads, ranks);
    fclose(log);
}

void generate_serial() {
    double start = omp_get_wtime();
    unsigned char *image = malloc(3 * WIDTH * HEIGHT);
    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = -2.0 + i * (3.0 / WIDTH);
            double y = -1.0 + j * (2.0 / HEIGHT);
            int iter = mandelbrot(x, y, MAX_ITER);
            unsigned char color = iter % 256;
            int idx = (j * WIDTH + i) * 3;
            image[idx] = color;
            image[idx + 1] = color;
            image[idx + 2] = color;
        }
    }
    double end = omp_get_wtime();
    char fname[100];
    sprintf(fname, "mandelbrot_serial_%dx%d_%diter.ppm", WIDTH, HEIGHT, MAX_ITER);
    write_ppm(fname, image, WIDTH, HEIGHT);
    free(image);
    log_result("Serial", fname, end - start, 1, 1);
}

void generate_openmp() {
    double start = omp_get_wtime();
    unsigned char *image = malloc(3 * WIDTH * HEIGHT);
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = -2.0 + i * (3.0 / WIDTH);
            double y = -1.0 + j * (2.0 / HEIGHT);
            int iter = mandelbrot(x, y, MAX_ITER);
            unsigned char color = iter % 256;
            int idx = (j * WIDTH + i) * 3;
            image[idx] = color;
            image[idx + 1] = color;
            image[idx + 2] = color;
        }
    }
    double end = omp_get_wtime();
    char fname[100];
    sprintf(fname, "mandelbrot_openmp_%dx%d_%diter.ppm", WIDTH, HEIGHT, MAX_ITER);
    write_ppm(fname, image, WIDTH, HEIGHT);
    free(image);
    log_result("OpenMP", fname, end - start, omp_get_max_threads(), 1);
}

void generate_hybrid(int rank, int size) {
    double global_start = MPI_Wtime();
    int rows_per_proc = HEIGHT / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? HEIGHT : start_row + rows_per_proc;
    int local_height = end_row - start_row;

    unsigned char *local_image = malloc(3 * WIDTH * local_height);
    double local_start = MPI_Wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int j = start_row; j < end_row; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = -2.0 + i * (3.0 / WIDTH);
            double y = -1.0 + j * (2.0 / HEIGHT);
            int iter = mandelbrot(x, y, MAX_ITER);
            int idx = ((j - start_row) * WIDTH + i) * 3;
            unsigned char color = iter % 256;
            local_image[idx] = color;
            local_image[idx + 1] = color;
            local_image[idx + 2] = color;
        }
    }

    double local_end = MPI_Wtime();

    if (rank == 0) {
        unsigned char *full_image = malloc(3 * WIDTH * HEIGHT);
        MPI_Gather(local_image, 3 * WIDTH * local_height, MPI_UNSIGNED_CHAR,
                   full_image, 3 * WIDTH * local_height, MPI_UNSIGNED_CHAR,
                   0, MPI_COMM_WORLD);
        double global_end = MPI_Wtime();
        char fname[100];
        sprintf(fname, "mandelbrot_hybrid_%dx%d_%diter.ppm", WIDTH, HEIGHT, MAX_ITER);
        write_ppm(fname, full_image, WIDTH, HEIGHT);
        free(full_image);
        log_result("Hybrid", fname, global_end - global_start, omp_get_max_threads(), size);
    } else {
        MPI_Gather(local_image, 3 * WIDTH * local_height, MPI_UNSIGNED_CHAR,
                   NULL, 0, MPI_UNSIGNED_CHAR,
                   0, MPI_COMM_WORLD);
    }

    free(local_image);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s [serial|openmp|hybrid] [width height max_iter]\n", argv[0]);
        return 1;
    }

    if (argc >= 5) {
        WIDTH = atoi(argv[2]);
        HEIGHT = atoi(argv[3]);
        MAX_ITER = atoi(argv[4]);
    }

    if (strcmp(argv[1], "serial") == 0) {
        generate_serial();
    }
    else if (strcmp(argv[1], "openmp") == 0) {
        generate_openmp();
    }
    else if (strcmp(argv[1], "hybrid") == 0) {
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        generate_hybrid(rank, size);
        MPI_Finalize();
    }
    else {
        printf("Invalid mode. Use: serial | openmp | hybrid\n");
        return 1;
    }

    return 0;
}
