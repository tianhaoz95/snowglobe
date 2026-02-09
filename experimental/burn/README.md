# Burn Qwen1.5-0.5B-Instruct Inference Example

This project provides an example of how to use the [Burn](https://github.com/tracel-ai/burn) framework to run inference for the `Qwen/Qwen1.5-0.5B-Instruct` model from the Hugging Face Hub.

## Running the Example

1.  **Navigate to the project directory:**

    ```bash
    cd experimental/burn
    ```

2.  **Run the inference command:**

    Use `cargo run` with the `--model-dir` and `--prompt` arguments. The `--model-dir` argument specifies the model repository on the Hugging Face Hub, and the `--prompt` argument provides the text to generate from.

    ```bash
    cargo run -- --model-dir Qwen/Qwen1.5-0.5B-Instruct --prompt "Hello, how are you?"
    ```

### Notes

*   The first time you run the command, it will download the model weights and tokenizer from the Hugging Face Hub and compile all the necessary dependencies. This process may take a significant amount of time, depending on your internet connection and system performance.
*   Subsequent runs will be much faster as the model and dependencies will be cached.

## Customization

You can change the prompt by modifying the value of the `--prompt` argument:

```bash
cargo run -- --model-dir Qwen/Qwen1.5-0.5B-Instruct --prompt "What is the capital of France?"
```
