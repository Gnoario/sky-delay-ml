import tensorflow as tf

print("Versão:", tf.__version__)
print("GPU disponível?", tf.config.list_physical_devices('GPU'))
print("Build CUDA:", tf.sysconfig.get_build_info()["cuda_version"])

