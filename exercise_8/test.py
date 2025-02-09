import tensorflow as tf

print("Is GPU available:", tf.test.is_gpu_available())
print("Built with CUDA:", tf.test.is_built_with_cuda())
