#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>

#define SHOW_TIME_DEBUG 1
#define loop(x, n) for (int x = 0; x < n; x++) 
#define loops(x, n, s) for (int x = 0; x <= n-s; x+=s)
#define FUNC_TIME_START \
    struct timeval start__, end__; gettimeofday(&start__, NULL);
#define FUNC_TIME_END \
    if(SHOW_TIME_DEBUG){ \
        gettimeofday(&end__, NULL); \
        printf("funcname: %s\t", __func__); \
        printf("Time tokens: %ld ms\n", (end__.tv_sec - start__.tv_sec) * 1000 + (end__.tv_usec - start__.tv_usec) / 1000); \
        fflush(stdout);} 
#define FILE_OPEN_CHECK(fp, filename) \
    do { \
        if (fp == NULL) { \
            printf("open file %s failed\n", filename); \
            return; \
        } \
    } while (0)

#define DOMAIN_TIME_START \
    struct timeval domain_start__, domain_end__; gettimeofday(&domain_start__, NULL);

#define DOMAIN_TIME_END(domain) \
    if(SHOW_TIME_DEBUG){ \
        gettimeofday(&domain_end__, NULL); \
        printf("domain: %s\t", domain); \
        printf("Time tokens: %ld ms\n", (domain_end__.tv_sec - domain_start__.tv_sec) * 1000 + (domain_end__.tv_usec - domain_start__.tv_usec) / 1000); \
        fflush(stdout);}

typedef unsigned short half;
typedef unsigned short ushort;
typedef unsigned int uint;

#ifdef __cplusplus
extern "C" {
#endif

void  data_fp16_fp32(void* fp16_data, void* fp32_data, int size);
float half_to_float(half x);
void  read_file(const char *filename, int size);
void  print_fp16_data(void* data, int size);
float fast_inverse_sqrt(float x);
void  read_buffer_from_file(const char *filename, int size, int start, void* buffer);
int str_ends_with(const char *str, const char *suffix);
const char* find_name_in_list_with_shuffix(const char *suffix, const char **list, int size);

#ifdef __cplusplus
}
#endif