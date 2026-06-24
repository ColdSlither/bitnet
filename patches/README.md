# llama.cpp I2_S Loader Patches

Apply these two changes to `src/llama.cpp` in the BitNet fork of llama.cpp
(https://github.com/Eddie-Wang1120/llama.cpp or ColdSlither fork).

## Patch 1: check_tensor_dims — skip shape check for I2_S tensors

I2_S tensors store packed ternary values with different shapes than the
logical weight dimensions. The shape check rejects them.

In `llama_model_loader::check_tensor_dims()`, after the `cur == NULL` check
and before the shape verification block, add:

```cpp
// Skip shape check for I2_S packed tensors — stored shape differs from logical shape
if (cur->type == GGML_TYPE_I2_S) {
    return cur;
}
```

## Patch 2: done_getting_tensors — allow fewer created tensors

After I2_S quantization, scale tensors are merged inline with weight data.
The quantizer keeps them in the GGUF info table but the loader creates fewer
tensors (optional scales are skipped).

In `llama_model_loader::done_getting_tensors()`, change:

```cpp
if (n_created != n_tensors) {
```
to:
```cpp
if (n_created > n_tensors) {
```
