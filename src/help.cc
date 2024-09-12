#include "help.h"

#ifdef __cplusplus
extern "C" {
#endif

static uint as_uint(const float x) {
    return *(uint*)&x;
}
static float as_float(const uint x) {
    return *(float*)&x;
}

int str_ends_with(const char *str, const char *suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) {
        return 0;
    }
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

const char* find_name_in_list_with_shuffix(const char *suffix, const char **list, int size) {
    for (int i = 0; i < size; i++) {
        if (str_ends_with(list[i], suffix)) {
            return (char*)list[i];
        }
    }
    return NULL;
}

void print_fp16_data(void* data, int size) {
    half* src = (half*)data;
    loop(i, size){
        printf("%f ", half_to_float(src[i]));
    }
    printf("\n");
}

void data_fp16_fp32(void* fp16_data, void* fp32_data, int size) {
    half* src = (half*)fp16_data;
    float* dst = (float*)fp32_data;
    loop(i, size){
        dst[i] = half_to_float(src[i]);
    }
}

float half_to_float(half x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint e = (x&0x7C00)>>10; // exponent
    const uint m = (x&0x03FF)<<13; // mantissa
    const uint v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}

void read_file(const char *filename, int size) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("open file %s failed\n", filename);
        return;
    }
    char *buf = (char *)malloc(size);
    if (buf == NULL) {
        printf("malloc failed\n");
        fclose(fp);
        return;
    }
    int read_size = 0;
    clock_t start = clock();
    while (read_size < size) {
        int ret = fread(buf + read_size, 1, size - read_size, fp);
        if (ret <= 0) {
            break;
        }
        read_size += ret;
    }
    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    // speeds GB/s
    double speed = (double)size / 1024 / 1024 / 1024 / duration;
    printf("read file %s, size %5d, duration %5f, speed %5f GB/s\n", filename, size, duration, speed);
    free(buf);
    fclose(fp);
}

void read_buffer_from_file(const char *filename, int size, int start, void* buffer) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("open file %s failed\n", filename);
        return;
    }
    // need check start + size < file size
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    if (start + size > file_size) {
        printf("start + size > file size. start: %d, size: %d, file size: %d\n", start, size, file_size);
        fclose(fp);
        return;
    }
    fseek(fp, start, SEEK_SET);
    int read_size = 0;
    while (read_size < size) {
        int ret = fread( (char*)buffer + read_size, 1, size - read_size, fp);
        if (ret <= 0) {
            break;
        }
        read_size += ret;
    }
    fclose(fp);
}

float fast_inverse_sqrt(float x)
{
    float half_x = 0.5 * x;
    int i = *((int *)&x); // 以整数方式读取X
    i = 0x5f3759df - (i>>1); // 神奇的步骤
    x = *((float *)&i); // 再以浮点方式读取i
    x = x*(1.5 - (half_x * x * x)); // 牛顿迭代一遍提高精度
    return x;
} 

#ifdef __cplusplus
}
#endif