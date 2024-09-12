#pragma once
#include <stdint.h>
#include "stdio.h"
#include "stdlib.h"
#include "ops.h"
#include "device.h"
#include "help.h"

#define SD3_ASSERT(x) DEVICE_ASSERT(x)
#define SD3_UNUSED(x) (void)(x)
// for debug 
#define READ_FILE_SPEED_SHOW 1
#define SHOW_DEVICE_DATA 1