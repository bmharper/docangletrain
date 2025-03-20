package main

import (
	"fmt"
	"os"

	"github.com/bmharper/cimg/v2"
	"github.com/bmharper/docangletrain/pkg/imagesplit"
)

// This program splits an image up into tiles and dumps them into jpg files.
// We use this to visually inspect whether the training images look similar enough to the real world samples.

const TileSize = 32
const NumTiles = 50

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	img, err := cimg.ReadFile(os.Args[1])
	check(err)
	tiles := imagesplit.SplitImage(img, NumTiles, TileSize)
	os.MkdirAll("samples", 0755)
	for i := 0; i < len(tiles); i++ {
		tiles[i].WriteJPEG(fmt.Sprintf("samples/tile_%d.jpg", i), cimg.MakeCompressParams(cimg.Sampling420, 95, 0), 0644)
	}
}
