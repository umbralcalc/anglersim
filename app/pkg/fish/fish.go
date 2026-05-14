// Package fish is the dexetera dashboard for the brown-trout climate
// vulnerability post. Sliders select a (climate, flow, DO, horizon)
// scenario; the wasm-side iterations look up the pre-computed
// projection grid embedded into the binary, trilinearly interpolate
// between cells, and emit (x, y, w, h) rectangle sets for the
// dexetera canvas renderer.
//
// See app/cmd/fish/{precompute,generate,register_step} for the grid
// pre-computation CLI, the widget codegen, and the wasm entry-point
// respectively.
package fish

import (
	"github.com/umbralcalc/dexetera/pkg/dashboard"
)

// Visual palette — kept consistent with the rugby post in the same
// collection. Blue is the simulation output colour; the magenta
// accent on the sliders is applied at codegen time (see
// cmd/fish/generate/main.go).
const (
	colorBackground = "#fafafa"
	colorTrajLine   = "#3c78d8"
	colorTrajBand   = "#bcd0ee"
	colorBaseline   = "#9a9a9a"
	colorBars       = "#3c78d8"
	colorBarsHist   = "#7d8aa1"
	colorAxis       = "#2c3e50"
	colorDivider    = "#e3e6ec"
)

// NewConfig returns the dashboard.Config for the fish climate-
// vulnerability widget. Order of renderers matters: later renderers
// draw on top of earlier ones.
func NewConfig() *dashboard.Config {
	vb := dashboard.NewVisualizationBuilder().
		WithCanvas(CanvasWidth, CanvasHeight).
		WithBackground(colorBackground).
		WithUpdateInterval(0).
		// Trajectory panel frame.
		AddLine("", ChartX0, TrajY0, ChartX1, TrajY0, &dashboard.LineOptions{
			Color: colorDivider, Width: 1,
		}).
		AddLine("", ChartX0, TrajY1, ChartX1, TrajY1, &dashboard.LineOptions{
			Color: colorAxis, Width: 1,
		}).
		// 90% band beneath the median dots.
		AddRectangleSet("trajectory_band", 0, 0, &dashboard.ShapeOptions{
			FillColor: colorTrajBand,
		}).
		// Baseline reference strip — dashed-looking thin horizontal bar
		// at the historical mean log-density. Drawn before the median
		// dots so the dots overlay it.
		AddRectangleSet("baseline_line", 0, 0, &dashboard.ShapeOptions{
			FillColor: colorBaseline,
		}).
		// Median trajectory — dense interpolated dots that read as a
		// continuous line at canvas scale.
		AddPointSet("trajectory_median", &dashboard.PointSetOptions{
			FillColor: colorTrajLine,
			Radius:    2,
		}).
		// Regional panel frame.
		AddLine("", ChartX0, RegionY0, ChartX1, RegionY0, &dashboard.LineOptions{
			Color: colorAxis, Width: 1,
		}).
		AddLine("", ChartX0, RegionY1, ChartX1, RegionY1, &dashboard.LineOptions{
			Color: colorDivider, Width: 1,
		}).
		AddRectangleSet("regional_bars", 0, 0, &dashboard.ShapeOptions{
			FillColor: colorBars,
		}).
		// Distribution panel frame.
		AddLine("", ChartX0, DistY0, ChartX1, DistY0, &dashboard.LineOptions{
			Color: colorDivider, Width: 1,
		}).
		AddLine("", ChartX0, DistY1, ChartX1, DistY1, &dashboard.LineOptions{
			Color: colorAxis, Width: 1,
		})

	// Single rectangleSet for the distribution panel — slate-grey to
	// echo the rugby post's outcome-distribution histogram.
	vb = vb.AddRectangleSet("distribution_bars", 0, 0, &dashboard.ShapeOptions{
		FillColor: colorBarsHist,
	})

	// Canvas-side bar labels. Placing them on the canvas immediately
	// below each panel ties each label set to its panel — the DOM
	// equivalent would put both label rows below the whole canvas,
	// leaving the reader to guess which is which.
	titleOpts := &dashboard.TextOptions{
		Color: colorAxis, FontSize: 16, TextAlign: "center",
		FontFamily: "system-ui, -apple-system, sans-serif",
	}
	barLabelOpts := &dashboard.TextOptions{
		Color: colorAxis, FontSize: 14, TextAlign: "center",
		FontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
	}
	centerX := (ChartX0 + ChartX1) / 2

	// Section title above the trajectory panel — same on-canvas style
	// as the regional / distribution titles so the three panels read
	// as a consistent stack.
	vb = vb.AddText("", "Projected log-density across sites (median + 90% band)",
		centerX, TrajY0-14, titleOpts)

	// Section title above the regional panel.
	vb = vb.AddText("", "% density change by region at the projection horizon",
		centerX, RegionY0-14, titleOpts)
	// Nine region abbreviations under the regional bars.
	regionLabels := [9]string{"D&C", "East", "K&SL", "North", "NE", "Nrn", "Wess", "West", "York"}
	regionSlot := float64(ChartX1-ChartX0) / float64(len(regionLabels))
	for i, lbl := range regionLabels {
		x := float64(ChartX0) + (float64(i)+0.5)*regionSlot
		vb = vb.AddText("", lbl, int(x), RegionY1+22, barLabelOpts)
	}

	// Section title above the distribution panel.
	vb = vb.AddText("", "Site count by outcome category at the projection horizon",
		centerX, DistY0-14, titleOpts)
	bandLabels := [4]string{"Stable", "Declining", "Critical >50%", "Extinct >90%"}
	bandSlot := float64(ChartX1-ChartX0) / float64(len(bandLabels))
	for i, lbl := range bandLabels {
		x := float64(ChartX0) + (float64(i)+0.5)*bandSlot
		vb = vb.AddText("", lbl, int(x), DistY1+22, barLabelOpts)
	}

	vis := vb.Build()

	cfg := dashboard.NewConfigBuilder("fish").
		WithDescription("Fish ecosystem vulnerability: drag the climate, flow, and water-quality sliders to see how projected brown trout density shifts across England's rivers over the chosen horizon.").
		WithServerPartition("scenario_snapshot").
		WithServerPartition("trajectory_band").
		WithServerPartition("trajectory_median").
		WithServerPartition("regional_bars").
		WithServerPartition("distribution_bars").
		WithServerPartition("baseline_line").
		WithActionStatePartition("scenario_snapshot").
		WithVisualization(vis).
		WithSimulation(BuildFishSimulation).
		WithSlider(dashboard.Slider{
			Name:       "climate_c",
			Label:      "Climate warming (°C)",
			Partition:  "scenario_snapshot",
			ValueIndex: IdxClimateC,
			Min:        0,
			Max:        3,
			Step:       0.25,
			Default:    DefaultScenario[IdxClimateC],
			Decimals:   2,
		}).
		WithSlider(dashboard.Slider{
			Name:       "flow_pct",
			Label:      "Flow change (%)",
			Partition:  "scenario_snapshot",
			ValueIndex: IdxFlowPct,
			Min:        -50,
			Max:        50,
			Step:       5,
			Default:    DefaultScenario[IdxFlowPct],
			Decimals:   0,
		}).
		WithSlider(dashboard.Slider{
			Name:       "do_pct",
			Label:      "Water quality / DO change (%)",
			Partition:  "scenario_snapshot",
			ValueIndex: IdxDOPct,
			Min:        -25,
			Max:        25,
			Step:       2.5,
			Default:    DefaultScenario[IdxDOPct],
			Decimals:   1,
		}).
		WithSlider(dashboard.Slider{
			Name:       "horizon_y",
			Label:      "Projection horizon (years)",
			Partition:  "scenario_snapshot",
			ValueIndex: IdxHorizonY,
			Min:        5,
			Max:        30,
			Step:       1,
			Default:    DefaultScenario[IdxHorizonY],
			Decimals:   0,
		}).
		WithReadout(dashboard.Readout{
			Partition: "scenario_snapshot",
			Template:  "climate +{v0}°C · flow {v1}% · DO {v2}% · year {v3}",
			Decimals:  1,
		}).
		WithResetButton().
		WithInlineDriver(40)

	return cfg.Build()
}
