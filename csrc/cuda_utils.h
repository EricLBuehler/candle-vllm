#include <torch/extension.h>

extern "C" {
int get_device_attribute(
    int attribute,
    int device_id);
}
