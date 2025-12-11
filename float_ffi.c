/* Float FFI helper for Lean - provides IEEE 754 bit manipulation */
#include <stdint.h>
#include <string.h>

/* Get the IEEE 754 binary64 (double) bit representation of a Lean Float */
uint64_t lean_float_to_bits(double f) {
    uint64_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

/* Create a Lean Float from IEEE 754 binary64 bits */
double lean_float_from_bits(uint64_t bits) {
    double f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Convert double to float32 bits (for sending to GPU) */
uint32_t lean_float_to_float32_bits(double f) {
    float f32 = (float)f;
    uint32_t bits;
    memcpy(&bits, &f32, sizeof(bits));
    return bits;
}

/* Create a double from float32 bits (for receiving from GPU) */
double lean_float_from_float32_bits(uint32_t bits) {
    float f32;
    memcpy(&f32, &bits, sizeof(f32));
    return (double)f32;
}
