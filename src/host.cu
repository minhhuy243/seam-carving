#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);                                                                 
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
    FILE * f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);
    
    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);
    
    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName)
{
    FILE * f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }   

    fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * originalWidth + c;
            fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
        }
    }
    
    fclose(f);
}

int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

void convertRgb2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

char *concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n\n");

}

uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height, int originalWidth)
{
    if (r < 0) {
        r = 0;
    } else if (r >= height) {
        r = height - 1;
    }

    if (c < 0) {
        c = 0;
    } else if (c >= width) {
        c = width - 1;
    }

    return pixels[r * originalWidth + c];
}

int computePixelPriority(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth) {
    int x = 0, y = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            uint8_t closest = getClosest(grayPixels, row - 1 + i, col - 1 + j, width, height, originalWidth);
            x += closest * xSobel[i][j];
            y += closest * ySobel[i][j];
        }
    }
    return abs(x) + abs(y);
}

void computeSeamScoreTable(int *priority, int *score, int width, int height, int originalWidth) {
    for (int c = 0; c < width; ++c) {
        score[c] = priority[c];
    }
    for (int r = 1; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            int aboveIdx = (r - 1) * originalWidth + c;

            int min = score[aboveIdx];
            if (c > 0 && score[aboveIdx - 1] < min) {
                min = score[aboveIdx - 1];
            }
            if (c < width - 1 && score[aboveIdx + 1] < min) {
                min = score[aboveIdx + 1];
            }

            score[idx] = min + priority[idx];
        }
    }
}

void seamCarvingByHostNaive(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    const int originalWidth = width;

    // allocate memory
    int *priority = (int *)malloc(width * height * sizeof(int));
    int *score = (int *)malloc(width * height * sizeof(int));
    uint8_t *grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));
    
    // turn input image to grayscale
    convertRgb2Gray(inPixels, width, height, grayPixels);

    // compute pixel priority
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            priority[r * originalWidth + c] = computePixelPriority(grayPixels, r, c, width, height, width);
        }
    }

    while (width > targetWidth) {
        // compute min seam table
        computeSeamScoreTable(priority, score, width, height, originalWidth);

        // find min index of last row
        int minCol = 0, r = height - 1;
        for (int c = 1; c < width; ++c) {
            if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
                minCol = c;
        }

        // trace and remove seam from last to first row
        for (; r >= 0; --r) {
            // remove seam pixel on row r
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * originalWidth + i] = outPixels[r * originalWidth + i + 1];
                grayPixels[r * originalWidth + i] = grayPixels[r * originalWidth + i + 1];
                priority[r * originalWidth + i] = priority[r * originalWidth + i + 1];
            }

            // update priority
            if (r < height - 1) {
                for (int affectedCol = 0; affectedCol < width - 1; ++affectedCol) {
                    priority[(r + 1) * originalWidth + affectedCol] = computePixelPriority(grayPixels, r + 1, affectedCol, width - 1, height, originalWidth);
                }
            }

            // trace up
            if (r > 0) {
                int aboveIdx = (r - 1) * originalWidth + minCol;
                int min = score[aboveIdx], minColCpy = minCol;
                if (minColCpy > 0 && score[aboveIdx - 1] < min) {
                    min = score[aboveIdx - 1];
                    minCol = minColCpy - 1;
                }
                if (minColCpy < width - 1 && score[aboveIdx + 1] < min) {
                    minCol = minColCpy + 1;
                }
            }
        }

        for (int affectedCol = 0; affectedCol < width - 1; ++affectedCol) {
            priority[affectedCol] = computePixelPriority(grayPixels, 0, affectedCol, width - 1, height, originalWidth);
        }

        --width;
    }
    
    free(grayPixels);
    free(score);
    free(priority);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use host): %f ms\n\n", time);
}

void seamCarvingByHostOptimized(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    const int originalWidth = width;

    // allocate memory
    int *priority = (int *)malloc(width * height * sizeof(int));
    int *score = (int *)malloc(width * height * sizeof(int));
    uint8_t *grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));
    
    // turn input image to grayscale
    convertRgb2Gray(inPixels, width, height, grayPixels);

    // compute pixel priority
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            priority[r * originalWidth + c] = computePixelPriority(grayPixels, r, c, width, height, width);
        }
    }

    while (width > targetWidth) {
        // compute min seam table
        computeSeamScoreTable(priority, score, width, height, originalWidth);

        // find min index of last row
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c) {
            if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
                minCol = c;
        }

        // trace and remove seam from last to first row
        for (; r >= 0; --r) {
            // remove seam pixel on row r
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * originalWidth + i] = outPixels[r * originalWidth + i + 1];
                grayPixels[r * originalWidth + i] = grayPixels[r * originalWidth + i + 1];
                priority[r * originalWidth + i] = priority[r * originalWidth + i + 1];
            }

            // update priority
            if (r < height - 1) {
                for (int affectedCol = max(0, prevMinCol - 2); affectedCol <= prevMinCol + 2 && affectedCol < width - 1; ++affectedCol) {
                    priority[(r + 1) * originalWidth + affectedCol] = computePixelPriority(grayPixels, r + 1, affectedCol, width - 1, height, originalWidth);
                }
            }

            // trace up
            if (r > 0) {
                prevMinCol = minCol;

                int aboveIdx = (r - 1) * originalWidth + minCol;
                int min = score[aboveIdx], minColCpy = minCol;
                if (minColCpy > 0 && score[aboveIdx - 1] < min) {
                    min = score[aboveIdx - 1];
                    minCol = minColCpy - 1;
                }
                if (minColCpy < width - 1 && score[aboveIdx + 1] < min) {
                    minCol = minColCpy + 1;
                }
            }
        }

        for (int affectedCol = max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol) {
            priority[affectedCol] = computePixelPriority(grayPixels, 0, affectedCol, width - 1, height, originalWidth);
        }

        --width;
    }
    
    free(grayPixels);
    free(score);
    free(priority);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use host): %f ms\n\n", time);
}

int main(int argc, char ** argv)
{   
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    // Read input RGB image file
    int width, height;
    uchar3 *inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    int numSeamRemoved = stoi(argv[3]);
    if (numSeamRemoved <= 0 || numSeamRemoved >= width)
        return EXIT_FAILURE; // invalid ratio
    printf("Number of seam removed: %d\n\n", numSeamRemoved);

    int targetWidth = width - numSeamRemoved;

    // seam carving using host
    uchar3 * correctOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarvingByHostNaive(inPixels, width, height, targetWidth, correctOutPixels);

    // seam carving using device
    uchar3 * outPixels= (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarvingByHostOptimized(inPixels, width, height, targetWidth, outPixels);

    // Compute mean absolute error between host result and device result
    float err = computeError(outPixels, correctOutPixels, width * height);
    printf("Error between device result and host result: %f\n", err);

    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(correctOutPixels, targetWidth, height, width, concatStr(outFileNameBase, "_host_naive.pnm"));
    writePnm(outPixels, targetWidth, height, width, concatStr(outFileNameBase, "_host_optimized.pnm"));

    // Free memories
    free(inPixels);
    free(correctOutPixels);
    free(outPixels);
}