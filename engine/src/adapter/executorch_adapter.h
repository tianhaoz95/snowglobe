#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ExecuTorchModule ExecuTorchModule;

ExecuTorchModule* executorch_module_load(const char* pte_path);
void executorch_module_destroy(ExecuTorchModule* module);
const char* executorch_module_get_name(ExecuTorchModule* module);
size_t executorch_module_get_vocab_size(ExecuTorchModule* module);

// For simplicity, we assume a fixed input size of (1, 128) for tokens as per current lib.rs logic
// and output is (1, 128, vocab_size)
int32_t executorch_module_forward(
    ExecuTorchModule* module,
    const void* input_tokens,
    size_t input_len,
    int32_t use_int32,
    float* output_logits,
    size_t* output_vocab_size,
    size_t start_pos,
    size_t num_positions
);

#ifdef __cplusplus
}
#endif
