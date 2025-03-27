# textorient-train

This is for training the neural network that https://github.com/bmharper/textorient uses

## Training

```bash
# Since we rely on the doctrain package, we must have NCNN available
git clone https://github.com/Tencent/ncnn.git
mkdir -p ncnn/build
cd ncnn/build
cmake -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF ..
make -j8
cd ../..

# Use this to generate synthetic training data
CGO_CPPFLAGS="-I$(pwd)/ncnn/src -I$(pwd)/ncnn/build/src" CGO_LDFLAGS=-L$(pwd)/ncnn/build/src go run cmd/generate/generate.go

# Train the neural network
cd nn
python train.py

# Export to NCNN
pnnx text_angle_classifier.pt inputshape=[1,1,32,32]
```

The NN output files that the textorient package uses are:

1. `text_angle_classifier.ncnn.bin`
2. `text_angle_classifier.ncnn.param`
