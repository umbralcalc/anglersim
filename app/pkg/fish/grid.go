package fish

import (
	"encoding/json"
	"fmt"
	"sort"
)

// Grid is a pre-computed projection grid over the three scenario axes
// (climate °C, flow %, DO %). Each cell holds aggregate statistics over
// the fitted population of NFPD sites under that scenario, evaluated at
// each year up to MaxHorizon.
//
// All arrays are stored flat in row-major order so the wasm-side lookup
// is a single offset computation. Layout:
//
//	Median, Lo90, Hi90      shape [Nc, Nf, Nd, Nh]
//	RegionalPct             shape [Nc, Nf, Nd, Nr, Nh]
//	BandCounts              shape [Nc, Nf, Nd, Nb, Nh]
//
// BaselineLogN is a single scalar — the historical mean log-density
// across all sites — used as the dashboard's reference line.
type Grid struct {
	ClimateAxis []float64 `json:"climate_axis"`
	FlowAxis    []float64 `json:"flow_axis"`
	DOAxis      []float64 `json:"do_axis"`

	Regions []string `json:"regions"`
	Bands   []string `json:"bands"`

	MaxHorizon int `json:"max_horizon"`

	BaselineLogN float64 `json:"baseline_log_n"`

	Median []float64 `json:"median"`
	Lo90   []float64 `json:"lo90"`
	Hi90   []float64 `json:"hi90"`

	RegionalPct []float64 `json:"regional_pct"`
	BandCounts  []float64 `json:"band_counts"`
}

// LoadGrid parses a Grid from its JSON encoding.
func LoadGrid(raw []byte) (*Grid, error) {
	g := &Grid{}
	if err := json.Unmarshal(raw, g); err != nil {
		return nil, err
	}
	if err := g.Validate(); err != nil {
		return nil, err
	}
	return g, nil
}

// Validate confirms the flat arrays match the axis dimensions.
func (g *Grid) Validate() error {
	nc, nf, nd, nh := len(g.ClimateAxis), len(g.FlowAxis), len(g.DOAxis), g.MaxHorizon
	nr, nb := len(g.Regions), len(g.Bands)
	if nc == 0 || nf == 0 || nd == 0 || nh == 0 {
		return fmt.Errorf("empty axis: c=%d f=%d d=%d h=%d", nc, nf, nd, nh)
	}
	wantTraj := nc * nf * nd * nh
	for name, a := range map[string][]float64{"median": g.Median, "lo90": g.Lo90, "hi90": g.Hi90} {
		if len(a) != wantTraj {
			return fmt.Errorf("%s: want %d values, got %d", name, wantTraj, len(a))
		}
	}
	if len(g.RegionalPct) != nc*nf*nd*nr*nh {
		return fmt.Errorf("regional_pct: want %d values, got %d", nc*nf*nd*nr*nh, len(g.RegionalPct))
	}
	if len(g.BandCounts) != nc*nf*nd*nb*nh {
		return fmt.Errorf("band_counts: want %d values, got %d", nc*nf*nd*nb*nh, len(g.BandCounts))
	}
	for name, a := range map[string][]float64{"climate": g.ClimateAxis, "flow": g.FlowAxis, "do": g.DOAxis} {
		if !sort.Float64sAreSorted(a) {
			return fmt.Errorf("%s axis must be sorted ascending", name)
		}
	}
	return nil
}

// axisIndex returns (lo, hi, frac) for trilinear interpolation. frac is
// in [0,1]; when v sits exactly on a knot or outside the axis range both
// indices coincide and frac is 0.
func axisIndex(axis []float64, v float64) (int, int, float64) {
	if v <= axis[0] {
		return 0, 0, 0
	}
	last := len(axis) - 1
	if v >= axis[last] {
		return last, last, 0
	}
	hi := sort.SearchFloat64s(axis, v)
	if hi == 0 {
		return 0, 0, 0
	}
	lo := hi - 1
	span := axis[hi] - axis[lo]
	if span == 0 {
		return lo, lo, 0
	}
	return lo, hi, (v - axis[lo]) / span
}

// interpolator captures the eight grid neighbours plus their weights for
// a (climate, flow, DO) query point. Year-indexed lookups reuse these.
type interpolator struct {
	g      *Grid
	cLo, cHi int
	fLo, fHi int
	dLo, dHi int
	cFrac, fFrac, dFrac float64
}

func (g *Grid) interpolator(climate, flowPct, doPct float64) interpolator {
	cLo, cHi, cFrac := axisIndex(g.ClimateAxis, climate)
	fLo, fHi, fFrac := axisIndex(g.FlowAxis, flowPct)
	dLo, dHi, dFrac := axisIndex(g.DOAxis, doPct)
	return interpolator{g: g,
		cLo: cLo, cHi: cHi, cFrac: cFrac,
		fLo: fLo, fHi: fHi, fFrac: fFrac,
		dLo: dLo, dHi: dHi, dFrac: dFrac,
	}
}

// trajOffset computes the flat offset into Median/Lo90/Hi90 for one
// (c, f, d) corner.
func (in interpolator) trajOffset(ci, fi, di int) int {
	nf, nd, nh := len(in.g.FlowAxis), len(in.g.DOAxis), in.g.MaxHorizon
	return ((ci*nf+fi)*nd+di)*nh
}

// trajSlice returns the (median, lo90, hi90) trajectory of length
// MaxHorizon trilinearly interpolated at (climate, flow, DO).
func (g *Grid) trajSlice(in interpolator) (median, lo, hi []float64) {
	nh := g.MaxHorizon
	median = make([]float64, nh)
	lo = make([]float64, nh)
	hi = make([]float64, nh)
	corners := [8]struct {
		w  float64
		ci int
		fi int
		di int
	}{
		{(1 - in.cFrac) * (1 - in.fFrac) * (1 - in.dFrac), in.cLo, in.fLo, in.dLo},
		{(1 - in.cFrac) * (1 - in.fFrac) * in.dFrac, in.cLo, in.fLo, in.dHi},
		{(1 - in.cFrac) * in.fFrac * (1 - in.dFrac), in.cLo, in.fHi, in.dLo},
		{(1 - in.cFrac) * in.fFrac * in.dFrac, in.cLo, in.fHi, in.dHi},
		{in.cFrac * (1 - in.fFrac) * (1 - in.dFrac), in.cHi, in.fLo, in.dLo},
		{in.cFrac * (1 - in.fFrac) * in.dFrac, in.cHi, in.fLo, in.dHi},
		{in.cFrac * in.fFrac * (1 - in.dFrac), in.cHi, in.fHi, in.dLo},
		{in.cFrac * in.fFrac * in.dFrac, in.cHi, in.fHi, in.dHi},
	}
	for _, c := range corners {
		if c.w == 0 {
			continue
		}
		off := in.trajOffset(c.ci, c.fi, c.di)
		for y := 0; y < nh; y++ {
			median[y] += c.w * g.Median[off+y]
			lo[y] += c.w * g.Lo90[off+y]
			hi[y] += c.w * g.Hi90[off+y]
		}
	}
	return median, lo, hi
}

// regionalAt returns the per-region median % density change at year y
// (1-indexed: y=1 is the first projection year).
func (g *Grid) regionalAt(in interpolator, year int) []float64 {
	if year < 1 {
		year = 1
	}
	if year > g.MaxHorizon {
		year = g.MaxHorizon
	}
	nf, nd, nh, nr := len(g.FlowAxis), len(g.DOAxis), g.MaxHorizon, len(g.Regions)
	out := make([]float64, nr)
	corners := [8]struct {
		w  float64
		ci int
		fi int
		di int
	}{
		{(1 - in.cFrac) * (1 - in.fFrac) * (1 - in.dFrac), in.cLo, in.fLo, in.dLo},
		{(1 - in.cFrac) * (1 - in.fFrac) * in.dFrac, in.cLo, in.fLo, in.dHi},
		{(1 - in.cFrac) * in.fFrac * (1 - in.dFrac), in.cLo, in.fHi, in.dLo},
		{(1 - in.cFrac) * in.fFrac * in.dFrac, in.cLo, in.fHi, in.dHi},
		{in.cFrac * (1 - in.fFrac) * (1 - in.dFrac), in.cHi, in.fLo, in.dLo},
		{in.cFrac * (1 - in.fFrac) * in.dFrac, in.cHi, in.fLo, in.dHi},
		{in.cFrac * in.fFrac * (1 - in.dFrac), in.cHi, in.fHi, in.dLo},
		{in.cFrac * in.fFrac * in.dFrac, in.cHi, in.fHi, in.dHi},
	}
	for _, c := range corners {
		if c.w == 0 {
			continue
		}
		base := (((c.ci*nf+c.fi)*nd+c.di)*nr)*nh + (year - 1)
		for r := 0; r < nr; r++ {
			out[r] += c.w * g.RegionalPct[base+r*nh]
		}
	}
	return out
}

// bandsAt returns the per-band site counts at year y. Counts are real
// (not integer) because trilinear interpolation produces fractional
// values; the renderer reads them as bar heights.
func (g *Grid) bandsAt(in interpolator, year int) []float64 {
	if year < 1 {
		year = 1
	}
	if year > g.MaxHorizon {
		year = g.MaxHorizon
	}
	nf, nd, nh, nb := len(g.FlowAxis), len(g.DOAxis), g.MaxHorizon, len(g.Bands)
	out := make([]float64, nb)
	corners := [8]struct {
		w  float64
		ci int
		fi int
		di int
	}{
		{(1 - in.cFrac) * (1 - in.fFrac) * (1 - in.dFrac), in.cLo, in.fLo, in.dLo},
		{(1 - in.cFrac) * (1 - in.fFrac) * in.dFrac, in.cLo, in.fLo, in.dHi},
		{(1 - in.cFrac) * in.fFrac * (1 - in.dFrac), in.cLo, in.fHi, in.dLo},
		{(1 - in.cFrac) * in.fFrac * in.dFrac, in.cLo, in.fHi, in.dHi},
		{in.cFrac * (1 - in.fFrac) * (1 - in.dFrac), in.cHi, in.fLo, in.dLo},
		{in.cFrac * (1 - in.fFrac) * in.dFrac, in.cHi, in.fLo, in.dHi},
		{in.cFrac * in.fFrac * (1 - in.dFrac), in.cHi, in.fHi, in.dLo},
		{in.cFrac * in.fFrac * in.dFrac, in.cHi, in.fHi, in.dHi},
	}
	for _, c := range corners {
		if c.w == 0 {
			continue
		}
		base := (((c.ci*nf+c.fi)*nd+c.di)*nb)*nh + (year - 1)
		for b := 0; b < nb; b++ {
			out[b] += c.w * g.BandCounts[base+b*nh]
		}
	}
	return out
}
