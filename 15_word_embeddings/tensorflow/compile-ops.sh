#!/usr/bin/env bash

git clone https://github.com/tensorflow/models.git tf_models;
mv tf_models/tutorials/embedding/*.cc .;

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# On Mac, add -undefined dynamic_lookup to the g++ command. The flag -D_GLIBCXX_USE_CXX11_ABI=0 is included to support newer versions of gcc. However, if you compiled TensorFlow from source using gcc 5 or later, you may need to exclude the flag. Specifically, if you get an error similar to the following: word2vec_ops.so: undefined symbol: _ZN10tensorflow7strings6StrCatERKNS0_8AlphaNumES3_S3_S3_ then you likely need to exclude the flag.

rm -rf tf_models;
rm *.cc