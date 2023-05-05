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

__constant__ int d_xSobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ int d_ySobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__device__ int d_originalWidth;

__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = row * width + col;
    if (row < height && col < width) {
        outPixels[idx] = 0.299f * inPixels[idx].x
                    + 0.587f * inPixels[idx].y
                    + 0.114f * inPixels[idx].z;
    }
}

__global__ void computePriorityKernel(uint8_t * inPixels, int width, int height, int * priority) {
    int filterWidth = 3;

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t s_width = blockDim.x + filterWidth - 1;
    size_t s_height = blockDim.y + filterWidth - 1;

    // Each block loads data from GMEM to SMEM
    extern __shared__ uint8_t s_inPixels[];

    int readRow = row - (filterWidth >> 1), readCol, tmpRow, tmpCol;
    int firstReadCol = col - (filterWidth >> 1);
    size_t virtualRow = threadIdx.y, virtualCol;
    for (; virtualRow < s_height; readRow += blockDim.y, virtualRow += blockDim.y) {
        tmpRow = readRow;
        if (readRow < 0) {
            readRow = 0;
        } else if (readRow >= height) {
            readRow = height - 1;
        }
        readCol = firstReadCol;
        virtualCol = threadIdx.x;
        for (; virtualCol < s_width; readCol += blockDim.x, virtualCol += blockDim.x) {
            tmpCol = readCol;
            if (readCol < 0) {
                readCol = 0;
            } else if (readCol >= width) {
                readCol = width - 1;
            }
            s_inPixels[virtualRow * s_width + virtualCol] = inPixels[readRow * d_originalWidth + readCol];
            readCol = tmpCol;
        }
        readRow = tmpRow;
    } 
    __syncthreads();
    // ---------------------------------------

    // Each thread compute priority on SMEM
    int x = 0, y = 0;
    for (int i = 0; i < filterWidth; ++i) {
        for (int j = 0; j < filterWidth; ++j) {
            uint8_t closest = s_inPixels[(threadIdx.y + i) * s_width + threadIdx.x + j];
            size_t filterIdx = i * filterWidth + j;
            x += closest * d_xSobel[filterIdx];
            y += closest * d_ySobel[filterIdx];
        }
    }

    // Each thread writes result from SMEM to GMEM
    if (col < width && row < height) {
        size_t idx = row * d_originalWidth + col;
        priority[idx] = abs(x) + abs(y);
    }
}

__global__ void carvingKernel(int *leastSignificantPixel, uchar3 *outPixels, uint8_t *grayPixels, int * priority, int width) {
    int row = blockIdx.x;
    int baseIdx = row * d_originalWidth;
    for (int i = leastSignificantPixel[row]; i < width - 1; ++i) {
        outPixels[baseIdx + i] = outPixels[baseIdx + i + 1];
        grayPixels[baseIdx + i] = grayPixels[baseIdx + i + 1];
        priority[baseIdx + i] = priority[baseIdx + i + 1];
    }
}

void trace(int *score, int *leastSignificantPixel, int width, int height, int originalWidth) {
    int minCol = 0, r = height - 1;
    for (int c = 1; c < width; ++c) {
        if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
            minCol = c;
    }
    for (; r >= 0; --r) {
        leastSignificantPixel[r] = minCol;
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
}

__global__ void computeSeamScoreTableKernel(int *priority, int *score, int width, int height, int fromRow) {
    size_t halfBlock = blockDim.x >> 1;

    int col = blockIdx.x * halfBlock - halfBlock + threadIdx.x;

    if (fromRow == 0 && col >= 0 && col < width) {
        score[col] = priority[col];
    }
    __syncthreads();

    for (int stride = fromRow != 0 ? 0 : 1; stride < halfBlock && fromRow + stride < height; ++stride) {
        if (threadIdx.x < blockDim.x - (stride << 1)) {
            int curRow = fromRow + stride;
            int curCol = col + stride;

            if (curCol >= 0 && curCol < width) {
                int idx = curRow * d_originalWidth + curCol;
                int aboveIdx = (curRow - 1) * d_originalWidth + curCol;

                int min = score[aboveIdx];
                if (curCol > 0 && score[aboveIdx - 1] < min) {
                    min = score[aboveIdx - 1];
                }
                if (curCol < width - 1 && score[aboveIdx + 1] < min) {
                    min = score[aboveIdx + 1];
                }

                score[idx] = min + priority[idx];
            }
        }
        __syncthreads();
    }
}

void seamCarvingByDevice(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels, dim3 blockSize) {
    GpuTimer timer;
    timer.Start();

    // allocate kernel memory
    uchar3 *d_inPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    uint8_t * d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    int * d_priority;
    CHECK(cudaMalloc(&d_priority, width * height * sizeof(int)));
    int * d_leastSignificantPixel;
    CHECK(cudaMalloc(&d_leastSignificantPixel, height * sizeof(int)));
    int * d_score;
    CHECK(cudaMalloc(&d_score, width * height * sizeof(int)));

    // allocate host memory
    int * priority = (int *)malloc(width * height * sizeof(int));
    int * leastSignificantPixel = (int *)malloc(height * sizeof(int));
    int * score = (int *)malloc(width * height * sizeof(int));

    // dynamically sized smem used to compute priority
    size_t smemSize = ((blockSize.x + 3 - 1) * (blockSize.y + 3 - 1)) * sizeof(uint8_t);
    
    // block size use to calculate seam score table
    int blockSizeDp = 256;
    int gridSizeDp = (((width - 1) / blockSizeDp + 1) << 1) + 1;
    int stripHeight = (blockSizeDp >> 1) + 1;

    // cache original width
    CHECK(cudaMemcpyToSymbol(d_originalWidth, &width, sizeof(int)));
    const int originalWidth = width;

    // copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // turn input image to grayscale
    dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
    convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // carve until reach desired width
    while (width > targetWidth) {
        // update priority
        computePriorityKernel<<<gridSize, blockSize, smemSize>>>(d_grayPixels, width, height, d_priority);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // compute min seam table
        for (int i = 0; i < height; i += (stripHeight >> 1)) {
            computeSeamScoreTableKernel<<<gridSizeDp, blockSizeDp>>>(d_priority, d_score, width, height, i);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
        }

        // find least significant pixel index of each row and store in d_leastSignificantPixel (SEQUENTIAL, in kernel or host)
        CHECK(cudaMemcpy(score, d_score, originalWidth * height * sizeof(int), cudaMemcpyDeviceToHost));
        trace(score, leastSignificantPixel, width, height, originalWidth);

        // carve
        CHECK(cudaMemcpy(d_leastSignificantPixel, leastSignificantPixel, height * sizeof(int), cudaMemcpyHostToDevice));
        carvingKernel<<<height, 1>>>(d_leastSignificantPixel, d_inPixels, d_grayPixels, d_priority, width);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        
        --width;
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, originalWidth * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_priority));
    CHECK(cudaFree(d_leastSignificantPixel));
    CHECK(cudaFree(d_score));

    free(score);
    free(leastSignificantPixel);
    free(priority);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use device): %f ms\n\n", time);    
}