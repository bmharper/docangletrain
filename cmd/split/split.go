package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/bmharper/cimg/v2"
	"github.com/bmharper/docangletrain/pkg/imagesplit"
	"github.com/tdewolff/argp"
)

// This program splits an image up into tiles and dumps them into jpg files.
// We use this to visually inspect whether the training images look similar enough to the real world samples.

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	tileSize := 32
	numTiles := 50
	inputFilename := ""
	outputDir := ""
	cmd := argp.New("Split image into tiles")
	cmd.AddOpt(&tileSize, "s", "size", "Tile size in pixels")
	cmd.AddOpt(&numTiles, "n", "num", "Number of tiles to create")
	cmd.AddArg(&inputFilename, "input", "Input image file")
	cmd.AddArg(&outputDir, "output", "Output directory for tiles")
	cmd.Parse()
	img, err := cimg.ReadFile(inputFilename)
	check(err)
	tiles := imagesplit.SplitImage(img, numTiles, tileSize)
	for angle := 0; angle < 4; angle++ {
		rotated := []*cimg.Image{}
		if angle == 0 {
			rotated = tiles
		} else {
			for _, org := range tiles {
				r := cimg.NewImage(tileSize, tileSize, org.Format)
				cimg.Rotate(org, r, float64(angle)*90.0*math.Pi/180.0, nil)
				rotated = append(rotated, r)
			}
		}
		dir := fmt.Sprintf("%v/%v", outputDir, angle)
		writeTilesIntoDirectory(rotated, dir)
	}
}

func writeTilesIntoDirectory(tiles []*cimg.Image, outputDir string) {
	os.MkdirAll(outputDir, 0755)
	files, err := filepath.Glob(fmt.Sprintf("%v/*", outputDir))
	check(err)
	// Start index at max(existing) + 1
	index := 0
	for _, file := range files {
		fn, _, _ := strings.Cut(filepath.Base(file), ".")
		i, err := strconv.Atoi(fn)
		check(err)
		index = max(index, i)
	}
	for i := 0; i < len(tiles); i++ {
		index++
		tiles[i].WriteJPEG(fmt.Sprintf("%v/%05d.jpg", outputDir, index), cimg.MakeCompressParams(cimg.Sampling420, 95, 0), 0644)
	}
}
