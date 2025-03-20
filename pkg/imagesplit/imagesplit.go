package imagesplit

import (
	"math/rand/v2"
	"slices"

	"github.com/bmharper/cimg/v2"
)

// #include "imagesplit.h"
import "C"

type tile struct {
	img        *cimg.Image
	perplexity int32
	//x          int32
	//y          int32
}

// Split an image up into size x size square tiles, and return numTiles samples
func SplitImage(img *cimg.Image, numTiles, size int) []*cimg.Image {
	const MaxSize = 2000
	const EdgePadding = 0.1 // Ignore this much of the image from all sides, because the dense text tends to be in the center of the page

	// Use random sampling to avoid getting unlucky with strided sampling
	prng := rand.New(rand.NewPCG(123, 456))

	if img.Format != cimg.PixelFormatGRAY {
		img = img.ToGray()
	}
	if img.Width > img.Height && img.Width > MaxSize {
		img = cimg.ResizeNew(img, MaxSize, img.Height*MaxSize/img.Width, nil)
	} else if img.Height > img.Width && img.Height > MaxSize {
		img = cimg.ResizeNew(img, img.Width*MaxSize/img.Height, MaxSize, nil)
	}

	x1 := int(float64(img.Width) * EdgePadding)
	x2 := int(float64(img.Width) * (1.0 - EdgePadding))
	y1 := int(float64(img.Height) * EdgePadding)
	y2 := int(float64(img.Height) * (1.0 - EdgePadding))

	nTilesX := (x2-x1)/size - 1
	nTilesY := (y2-y1)/size - 1
	if nTilesX < 0 || nTilesY < 0 {
		return nil
	}
	totalTiles := nTilesX * nTilesY
	tiles := make([]tile, totalTiles)
	for y := 0; y < nTilesY; y++ {
		for x := 0; x < nTilesX; x++ {
			px := x1 + int(x)*size
			py := y1 + int(y)*size
			crop := img.ReferenceCrop(px, py, px+size, py+size)
			tiles[y*nTilesX+x] = tile{
				img:        crop,
				perplexity: int32(C.horizontal_perplexity((*C.byte)(&crop.Pixels[0]), C.int(crop.Width), C.int(crop.Height), C.int(crop.Stride))),
				//x:          int32(x),
				//y:          int32(y),
			}
		}
	}
	slices.SortFunc(tiles, func(i, j tile) int {
		return int(j.perplexity - i.perplexity)
	})
	// Pick the top numTiles * 3 (by perplexity), and then shuffle them
	tiles = tiles[:min(totalTiles, numTiles*3)]
	prng.Shuffle(len(tiles), func(i, j int) {
		tiles[i], tiles[j] = tiles[j], tiles[i]
	})
	samples := make([]*cimg.Image, numTiles)
	for i := 0; i < numTiles; i++ {
		tile := tiles[i]
		//x := x1 + int(tile.x)*size
		//y := y1 + int(tile.y)*size
		//samples[i] = img.ReferenceCrop(x, y, x+size, y+size)
		samples[i] = tile.img
	}

	return samples
}
