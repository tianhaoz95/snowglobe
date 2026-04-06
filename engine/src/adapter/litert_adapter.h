#ifndef SNOWGLOBE_LITERT_ADAPTER_H
#define SNOWGLOBE_LITERT_ADAPTER_H

#include <stdint.h>
#include <stddef.h>

extern "C" {
    /**
     * Loads a LiteRT (.tflite) model from the given path.
     * Returns an opaque pointer to the internal model state.
     */
    void* litert_model_load(const char* model_path);

    /**
     * Destroys the model and frees associated resources.
     */
    void litert_model_destroy(void* model);
    
    /**
     * Executes the 'prefill' signature for parallel prompt processing.
     * @param input_ids Pointer to the array of input token IDs.
     * @param length Number of tokens in input_ids.
     * @param output_logits Pointer to buffer for the resulting logits (size: vocab_size).
     * Returns 0 on success, non-zero on error.
     */
    int litert_model_prefill(
        void* model,
        const int32_t* input_ids,
        size_t length,
        float* output_logits
    );
    
    /**
     * Executes the 'decode' signature for single token generation.
     * @param input_id The latest token ID generated.
     * @param pos The current position in the sequence (for KV cache).
     * @param output_logits Pointer to buffer for the resulting logits (size: vocab_size).
     * Returns 0 on success, non-zero on error.
     */
    int litert_model_decode(
        void* model,
        int32_t input_id,
        int32_t pos,
        float* output_logits
    );

    /**
     * Resets the internal KV cache state of the model.
     */
    int litert_model_reset_state(void* model);
}

#endif // SNOWGLOBE_LITERT_ADAPTER_H
