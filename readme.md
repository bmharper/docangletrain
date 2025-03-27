# textorient-train

This is for training the neural network that https://github.com/bmharper/textorient uses.

## Training

### Python environment

- Install the latest `PyTorch`.
- Install `pnnx` for exporting from Pytorch to NCNN.

You'll also need native build tools necessary for building ncnn (e.g. cmake, g++, etc)

### Download the training images

Download training images from the latest release of this project on github, and unzip them into the root of this repo.
That should give you an `images` directory with this structure:

- textorient-train
  - images
    - real
      - train
        - 00001.jpg
        - ...
      - val
    - synth
      - train
      - val

### Train

```bash
# Train the neural network
cd nn
python train.py

# Export to NCNN
pnnx text_angle_classifier.pt inputshape=[1,1,32,32]
```

The NN output files that the textorient package uses are:

1. `text_angle_classifier.ncnn.bin`
2. `text_angle_classifier.ncnn.param`

These are the only two outputs that you need for the textorient package.

## Generating synthetic data

Use the following steps to generate the synthetic training data inside `images/synth`. The `images.zip` file that is published as part of the release already contains these generated images, so you only need to run this command if you're generating more synthetic images (or you've changed the image generation parameters, etc).

### Build NCNN

```bash
# Since we rely on the doctrain Go package, we must have NCNN available.
# We don't actually use ncnn for anything here. This is just an unfortunate
# side effect of importing the doctrain Go package.
git clone https://github.com/Tencent/ncnn.git
mkdir -p ncnn/build
cd ncnn/build
cmake -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF ..
make -j8
cd ../..
```

### Generate

```bash
# Use this to generate synthetic training data
CGO_CPPFLAGS="-I$(pwd)/ncnn/src -I$(pwd)/ncnn/build/src" CGO_LDFLAGS=-L$(pwd)/ncnn/build/src go run cmd/generate/generate.go
```

### Generating data from real images

The `cmd/split/split.go` tool can be used to generate training data from real images. We place those in `images/real`. The source images that we use internally are proprietary, so we can't share them here.
