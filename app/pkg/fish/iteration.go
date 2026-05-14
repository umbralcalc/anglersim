package fish

import (
	"math"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// Canvas geometry. Three stacked panels — trajectory on top, regional
// breakdown in the middle, site-level distribution on the bottom. The
// horizontal padding (ChartX0/ChartX1) is shared across all three so
// vertical alignment is implicit.
//
// Inter-panel gaps are sized to fit the upper panel's bar labels (~25
// canvas px) and the lower panel's section title (~25 canvas px) plus
// breathing room, so the bumped 14/16-px label fonts can't collide.
const (
	CanvasWidth  = 660
	CanvasHeight = 600

	ChartX0 = 85
	ChartX1 = 635

	TrajY0     = 50
	TrajY1     = 170
	TrajHeight = TrajY1 - TrajY0

	RegionY0     = 240
	RegionY1     = 360
	RegionHeight = RegionY1 - RegionY0

	DistY0     = 430
	DistY1     = 550
	DistHeight = DistY1 - DistY0
)

// Action-state vector layout. Sliders write into a 4-vector on the
// scenario_snapshot partition; downstream visualization partitions read
// it via upstream wiring.
const (
	IdxClimateC   = 0
	IdxFlowPct    = 1
	IdxDOPct      = 2
	IdxHorizonY   = 3
	ScenarioWidth = 4
)

// SubPointsPerYear controls how many interpolated trajectory points are
// emitted between consecutive year data points. Higher values make the
// median trajectory render as a near-continuous line rather than a
// sparse dot sequence.
const SubPointsPerYear = 5

// Slider defaults sit in the centre of each axis where possible. The
// climate default of 0 matches the plan: the reader's first interaction
// is to drag the climate slider and watch the trajectory shift.
var DefaultScenario = []float64{0.0, 0.0, 0.0, 20.0}

// Y-axis half-spans around BaselineLogN (in log-density units). Picked
// so even a +3°C / drought scenario stays inside the chart and ~80% of
// the chart area is occupied at default horizon.
const (
	YAboveBaseline = 1.0
	YBelowBaseline = 3.5
)

// densityToY maps a log-density value to a canvas y coordinate inside
// the trajectory panel, clamping at the panel bounds.
func densityToY(logN, baseline float64) float64 {
	top := baseline + YAboveBaseline
	bot := baseline - YBelowBaseline
	frac := (top - logN) / (top - bot)
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	return TrajY0 + frac*TrajHeight
}

// ScenarioSnapshotIteration is the action-state partition. Sliders
// write into action_state_values; the iteration latches that vector
// and re-emits it each step so downstream visualization partitions
// have a stable upstream value to read.
//
// Output state width: ScenarioWidth (4).
type ScenarioSnapshotIteration struct{}

func (s *ScenarioSnapshotIteration) Configure(int, *simulator.Settings) {}

func (s *ScenarioSnapshotIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	out := make([]float64, ScenarioWidth)
	actions, ok := params.GetOk("action_state_values")
	if !ok || len(actions) < ScenarioWidth {
		copy(out, DefaultScenario)
		return out
	}
	copy(out, actions[:ScenarioWidth])
	if out[IdxHorizonY] < 1 {
		out[IdxHorizonY] = 1
	}
	return out
}

// scenarioFromUpstream is a small adapter: each downstream iteration
// reads the snapshot slice via upstream wiring, defaulting to the
// initial values if the upstream slice is short.
func scenarioFromUpstream(params *simulator.Params) (climate, flowPct, doPct float64, horizon int) {
	s, _ := params.GetOk("scenario_values")
	if len(s) < ScenarioWidth {
		return DefaultScenario[0], DefaultScenario[1], DefaultScenario[2], int(DefaultScenario[IdxHorizonY])
	}
	climate = s[IdxClimateC]
	flowPct = s[IdxFlowPct]
	doPct = s[IdxDOPct]
	horizon = int(math.Round(s[IdxHorizonY]))
	if horizon < 1 {
		horizon = 1
	}
	return
}

// TrajectoryBandIteration emits the 90% credible band as thin vertical
// rectangles, one per projection year, suitable for AddRectangleSet.
// State layout: 4 floats per year × MaxHorizon.
//
// Years beyond the current horizon slider are emitted as zero-height
// rectangles, which AddRectangleSet skips.
type TrajectoryBandIteration struct {
	Grid *Grid
}

func (t *TrajectoryBandIteration) Configure(int, *simulator.Settings) {}

func (t *TrajectoryBandIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	climate, flowPct, doPct, horizon := scenarioFromUpstream(params)
	if horizon > t.Grid.MaxHorizon {
		horizon = t.Grid.MaxHorizon
	}
	in := t.Grid.interpolator(climate, flowPct, doPct)
	_, lo, hi := t.Grid.trajSlice(in)

	out := make([]float64, 4*t.Grid.MaxHorizon)
	if horizon <= 0 {
		return out
	}
	denom := float64(horizon)
	width := float64(ChartX1-ChartX0) / denom
	// Fill the slot — leaving the rectangles abutting reads as a
	// continuous band rather than disconnected vertical sticks.
	barW := width
	for y := 0; y < horizon; y++ {
		px := float64(ChartX0) + (float64(y)+0.5)*width
		pyLo := densityToY(lo[y], t.Grid.BaselineLogN)
		pyHi := densityToY(hi[y], t.Grid.BaselineLogN)
		// rectangleSet centres on (x, y), so the centre of the band is
		// the midpoint of (pyHi, pyLo) and the height is their span.
		centerY := (pyLo + pyHi) / 2
		height := pyLo - pyHi
		if height < 1 {
			height = 1
		}
		out[y*4+0] = px
		out[y*4+1] = centerY
		out[y*4+2] = barW
		out[y*4+3] = height
	}
	return out
}

// offCanvas is the sentinel coordinate emitted for out-of-horizon
// points. The renderer draws at it (no NaN filtering in the canvas
// API) but the position sits far outside the canvas so the dot is
// clipped. NaN would be cleaner but the stochadex harness rejects it.
const offCanvas = -9999.0

// TrajectoryMedianIteration emits the median projection as a dense
// series of (x, y) points consumed by AddPointSet. Points are linearly
// interpolated between annual data points at SubPointsPerYear
// resolution so the rendered dot sequence reads as a continuous line.
// Slots beyond the current horizon are emitted at offCanvas
// coordinates so the renderer's draw call lands outside the visible
// area.
type TrajectoryMedianIteration struct {
	Grid *Grid
}

// MedianStateWidth is the output width of TrajectoryMedianIteration.
func MedianStateWidth(maxHorizon int) int { return 2 * maxHorizon * SubPointsPerYear }

func (t *TrajectoryMedianIteration) Configure(int, *simulator.Settings) {}

func (t *TrajectoryMedianIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	climate, flowPct, doPct, horizon := scenarioFromUpstream(params)
	if horizon > t.Grid.MaxHorizon {
		horizon = t.Grid.MaxHorizon
	}
	in := t.Grid.interpolator(climate, flowPct, doPct)
	median, _, _ := t.Grid.trajSlice(in)

	out := make([]float64, MedianStateWidth(t.Grid.MaxHorizon))
	for i := range out {
		out[i] = offCanvas
	}
	if horizon <= 0 {
		return out
	}
	denom := float64(horizon)
	width := float64(ChartX1-ChartX0) / denom
	subStep := 1.0 / float64(SubPointsPerYear)
	totalSlots := horizon * SubPointsPerYear
	for i := 0; i < totalSlots; i++ {
		yIdx := i / SubPointsPerYear
		k := i % SubPointsPerYear
		frac := float64(k) * subStep
		// Skip sub-points past the final data point — they would
		// dangle past the line's natural end.
		if yIdx >= horizon-1 && k > 0 {
			continue
		}
		var dens float64
		if k == 0 {
			dens = median[yIdx]
		} else {
			dens = median[yIdx]*(1-frac) + median[yIdx+1]*frac
		}
		px := float64(ChartX0) + (float64(yIdx)+0.5+frac)*width
		py := densityToY(dens, t.Grid.BaselineLogN)
		out[i*2+0] = px
		out[i*2+1] = py
	}
	return out
}

// RegionalBarsIteration emits a (x, y, w, h) rectangle per region for
// AddRectangleSet. Bars hang from the top of the regional panel and
// grow downward when the regional % density change is negative
// (decline). Improvements (positive change) draw as upward stubs above
// the panel-top line so the reader can tell they aren't missing data.
type RegionalBarsIteration struct {
	Grid *Grid
}

func (r *RegionalBarsIteration) Configure(int, *simulator.Settings) {}

func (r *RegionalBarsIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	climate, flowPct, doPct, horizon := scenarioFromUpstream(params)
	if horizon > r.Grid.MaxHorizon {
		horizon = r.Grid.MaxHorizon
	}
	in := r.Grid.interpolator(climate, flowPct, doPct)
	vals := r.Grid.regionalAt(in, horizon)

	n := len(vals)
	out := make([]float64, 4*n)
	if n == 0 {
		return out
	}
	slot := float64(ChartX1-ChartX0) / float64(n)
	barW := math.Min(slot*0.65, 38)
	// 100% loss occupies the full panel height; clamp anything beyond.
	const maxAbsPct = 100.0
	const minMarker = 3.0
	for i, v := range vals {
		px := float64(ChartX0) + (float64(i)+0.5)*slot
		mag := math.Abs(v)
		if mag > maxAbsPct {
			mag = maxAbsPct
		}
		height := mag / maxAbsPct * float64(RegionHeight-6)
		if height < minMarker {
			height = minMarker
		}
		// Decline (v <= 0): bar grows down from panel top. Improvement
		// (v > 0): bar grows up from the panel-top baseline.
		var centerY float64
		if v <= 0 {
			centerY = float64(RegionY0) + height/2
		} else {
			centerY = float64(RegionY0) - height/2
		}
		out[i*4+0] = px
		out[i*4+1] = centerY
		out[i*4+2] = barW
		out[i*4+3] = height
	}
	return out
}

// DistributionBarsIteration emits one (x, y, w, h) rectangle per
// outcome band (stable / declining / critical / extinct) for the
// bottom panel. Bar heights are normalised against the total site
// count so the panel always uses its full vertical range; empty bands
// still emit a minimum-height marker so the reader can see "0" rather
// than a missing bar.
type DistributionBarsIteration struct {
	Grid *Grid
}

func (d *DistributionBarsIteration) Configure(int, *simulator.Settings) {}

func (d *DistributionBarsIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	climate, flowPct, doPct, horizon := scenarioFromUpstream(params)
	if horizon > d.Grid.MaxHorizon {
		horizon = d.Grid.MaxHorizon
	}
	in := d.Grid.interpolator(climate, flowPct, doPct)
	vals := d.Grid.bandsAt(in, horizon)

	n := len(vals)
	out := make([]float64, 4*n)
	if n == 0 {
		return out
	}
	maxVal := 0.0
	totalSites := 0.0
	for _, v := range vals {
		totalSites += v
		if v > maxVal {
			maxVal = v
		}
	}
	denomNorm := totalSites
	if denomNorm <= 0 {
		denomNorm = 1
	}
	scaleRef := maxVal / denomNorm
	if scaleRef <= 0 {
		scaleRef = 1
	}
	slot := float64(ChartX1-ChartX0) / float64(n)
	barW := math.Min(slot*0.55, 80)
	const minMarker = 3.0
	for i, v := range vals {
		px := float64(ChartX0) + (float64(i)+0.5)*slot
		share := v / denomNorm
		// Tallest bar reaches ~95% of the panel.
		height := share / scaleRef * float64(DistHeight-6) * 0.95
		if height < minMarker {
			height = minMarker
		}
		centerY := float64(DistY1) - height/2
		out[i*4+0] = px
		out[i*4+1] = centerY
		out[i*4+2] = barW
		out[i*4+3] = height
	}
	return out
}

// BaselineLineIteration emits a single (x, y, w, h) rectangle drawn as
// a thin horizontal strip at BaselineLogN on the trajectory chart's
// y-axis. It acts as the "no-change" reference so the reader can see
// slider effects as shifts relative to baseline.
type BaselineLineIteration struct {
	Grid *Grid
}

func (b *BaselineLineIteration) Configure(int, *simulator.Settings) {}

func (b *BaselineLineIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	y := densityToY(b.Grid.BaselineLogN, b.Grid.BaselineLogN)
	centerX := float64(ChartX0+ChartX1) / 2
	width := float64(ChartX1 - ChartX0)
	return []float64{centerX, y, width, 1.2}
}
