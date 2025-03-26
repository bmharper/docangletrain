package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/bmharper/cimg/v2"
	"github.com/bmharper/textorient"
	"github.com/tdewolff/canvas"
	"github.com/tdewolff/canvas/renderers/rasterizer"
)

// Generate synthetic data by rendering text into a canvas at one of 4 different angles,
// and writing out those little training images into jpegs.

const ImageSize = 32
const NumTrain = 1000
const NumVal = 100
const MinAngle = 0
const MaxAngle = 270
const AngleStep = 90
const MinPerplexity = 0.1

func main() {
	g := generator{}
	g.init()

	datasets := []string{"train", "val"}
	numImages := []int{NumTrain, NumVal}
	seed := 0

	for ds := range datasets {
		class := 0
		for angle := MinAngle; angle <= MaxAngle; angle += AngleStep {
			fmt.Printf("Generating %v %v %v\n", datasets[ds], class, angle)
			dir := fmt.Sprintf("images/%v/%v", datasets[ds], class)
			if err := os.RemoveAll(dir); err != nil {
				panic(err)
			}
			if err := os.MkdirAll(dir, 0755); err != nil {
				panic(err)
			}
			nGenerated := 0
			for i := 0; nGenerated < numImages[ds]; i++ {
				if g.generate(dir, seed, nGenerated, angle) {
					nGenerated++
				}
				if i > numImages[ds]*3 {
					panic("Rejecting too many. Lower perplexity threshold")
				}
				seed++
			}
			class++
		}
	}
}

type generator struct {
	fonts []*canvas.FontFamily
}

func (g *generator) init() {
	files, err := filepath.Glob("fonts/*.ttf")
	if err != nil {
		panic("Unable to find fonts: " + err.Error())
	}
	if len(files) == 0 {
		panic("No font files found")
	}
	for _, file := range files {
		font := canvas.NewFontFamily(filepath.Base(file))
		if err := font.LoadFontFile(file, canvas.FontRegular); err != nil {
			panic(err)
		}
		g.fonts = append(g.fonts, font)
	}
}

func (g *generator) generate(outputDir string, seed, imageIdx, angle int) bool {
	// The canvas library panics every now and then on invalid polygons (error: "next node for result polygon is nil, probably buggy intersection code")
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Error generating image %v: %v\n", imageIdx, r)
		}
	}()

	rng := rand.New(rand.NewPCG(uint64(seed), uint64(seed)))

	c := canvas.New(ImageSize, ImageSize)
	ctx := canvas.NewContext(c)

	p := &canvas.Path{}
	p.MoveTo(0, 0)
	p.LineTo(ImageSize, 0)
	p.LineTo(ImageSize, ImageSize)
	p.LineTo(0, ImageSize)
	p.Close()
	ctx.SetFillColor(canvas.White)
	ctx.DrawPath(0, 0, p)

	//p = &canvas.Path{}
	//p.MoveTo(10, 10)
	//p.Arc(10, 10, 90, 0, 180)
	//p.Close()
	//ctx.SetFillColor(canvas.Red)
	//ctx.DrawPath(5, 5, p)

	font := g.fonts[rng.IntN(len(g.fonts))]
	weight := canvas.FontRegular
	wf := rng.Float64()
	if wf <= 0.05 {
		weight = canvas.FontLight
	} else if wf <= 0.4 {
		weight = canvas.FontExtraBold
	} else if wf <= 0.8 {
		weight = canvas.FontBold
	} else {
		weight = canvas.FontRegular
	}
	x := -10 + rng.Float64()*10.0
	y := -30 + rng.Float64()*30.0
	height := 40.0 + rng.Float64()*40.0
	lineHeight := height*0.4 + height*rng.Float64()*0.3

	face := font.Face(height, canvas.Black, weight, canvas.FontNormal)
	for i := 0; i < 6; i++ {
		ctx.DrawText(x, y, canvas.NewTextLine(face, g.word(rng), canvas.Left))
		y += lineHeight
	}

	fn := fmt.Sprintf("%v/%03d.jpg", outputDir, imageIdx)
	img := rasterizer.Draw(c, canvas.DPMM(4), canvas.DefaultColorSpace)
	im, err := cimg.FromImage(img, true)
	if err != nil {
		panic(err)
	}
	downsize := cimg.ResizeNew(im, ImageSize, ImageSize, nil)
	if textorient.Perplexity(downsize) < MinPerplexity {
		return false
	}
	rotated := cimg.NewImage(downsize.Width, downsize.Height, downsize.Format)
	cimg.Rotate(downsize, rotated, float64(angle)*math.Pi/180.0, nil)
	if err := rotated.WriteJPEG(fn, cimg.MakeCompressParams(cimg.Sampling420, 95, 0), 0644); err != nil {
		panic(err)
	}
	return true
}

func (g *generator) word(rng *rand.Rand) string {
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	digits := []rune("012345678901234567890123456789.,-_+()")
	word := make([]rune, 0, 10)
	if rng.IntN(10) <= 4 {
		for i := 0; i < 20; i++ {
			word = append(word, digits[rng.IntN(len(digits))])
		}
	} else {
		for i := 0; i < 20; i++ {
			word = append(word, letters[rng.IntN(len(letters))])
		}
	}
	return string(word)
}
