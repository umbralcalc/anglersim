package fish

import (
	"math"
	"sync"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// shortGen returns the fish generator clamped to a small number of
// steps so individual tests stay sub-second. Every visualization
// iteration is stateless on its own history, so a few steps suffice
// to exercise the wiring.
func shortGen(steps int) *simulator.ConfigGenerator {
	gen := BuildFishSimulation()
	gen.SetSimulation(&simulator.SimulationConfig{
		OutputCondition:      &simulator.EveryStepOutputCondition{},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: steps},
		TimestepFunction:     &simulator.ConstantTimestepFunction{Stepsize: 1.0},
		InitTimeValue:        0.0,
	})
	return gen
}

func TestBuildFishSimulation(t *testing.T) {
	t.Run("harness", func(t *testing.T) {
		settings, implementations := shortGen(4).GenerateConfigs()
		if err := simulator.RunWithHarnesses(settings, implementations); err != nil {
			t.Fatalf("harness failed: %v", err)
		}
	})

	t.Run("partition shapes and visualisation invariants", func(t *testing.T) {
		gen := shortGen(3)
		settings, implementations := gen.GenerateConfigs()
		store := simulator.NewStateTimeStorage()
		implementations.OutputFunction = &simulator.StateTimeStorageOutputFunction{Store: store}
		coordinator := simulator.NewPartitionCoordinator(settings, implementations)
		var wg sync.WaitGroup
		for !coordinator.ReadyToTerminate() {
			coordinator.Step(&wg)
		}

		grid := LoadEmbeddedGrid()

		// scenario_snapshot: 4 floats latched from action_state_values.
		snap := store.GetValues("scenario_snapshot")
		if len(snap) == 0 {
			t.Fatal("no scenario_snapshot output")
		}
		if got := len(snap[0]); got != ScenarioWidth {
			t.Errorf("scenario_snapshot width=%d, want %d", got, ScenarioWidth)
		}

		// trajectory_band: 4 floats per horizon year, all non-negative
		// w/h. Negatives would silently disappear from the canvas.
		band := store.GetValues("trajectory_band")
		bandLast := band[len(band)-1]
		if len(bandLast) != 4*grid.MaxHorizon {
			t.Errorf("trajectory_band width=%d, want %d", len(bandLast), 4*grid.MaxHorizon)
		}
		for i := 0; i+3 < len(bandLast); i += 4 {
			if bandLast[i+2] < 0 || bandLast[i+3] < 0 {
				t.Errorf("band rect %d has negative size (w=%v, h=%v)",
					i/4, bandLast[i+2], bandLast[i+3])
			}
		}

		// trajectory_median: 2 floats per (year × SubPointsPerYear).
		// In-range entries sit inside the trajectory panel; everything
		// past the last data point is off-canvas so it clips cleanly.
		med := store.GetValues("trajectory_median")
		medLast := med[len(med)-1]
		if len(medLast) != MedianStateWidth(grid.MaxHorizon) {
			t.Errorf("trajectory_median width=%d, want %d",
				len(medLast), MedianStateWidth(grid.MaxHorizon))
		}
		horizon := int(snap[len(snap)-1][IdxHorizonY])
		// At least the year-aligned data points (k=0) within horizon
		// must sit inside the panel.
		for y := 0; y < horizon; y++ {
			i := y * SubPointsPerYear
			x, py := medLast[i*2+0], medLast[i*2+1]
			if x < ChartX0 || x > ChartX1 {
				t.Errorf("median year %d x=%v outside chart [%d, %d]",
					y, x, ChartX0, ChartX1)
			}
			if py < TrajY0 || py > TrajY1 {
				t.Errorf("median year %d py=%v outside trajectory panel",
					y, py)
			}
		}
		// Slots after the last drawn sub-point must be off-canvas.
		lastDrawn := (horizon-1)*SubPointsPerYear + 1
		for i := lastDrawn; i < grid.MaxHorizon*SubPointsPerYear; i++ {
			x, py := medLast[i*2+0], medLast[i*2+1]
			if x != offCanvas || py != offCanvas {
				t.Errorf("median slot %d should be off-canvas, got (%v, %v)",
					i, x, py)
			}
		}

		// regional_bars: 4 floats per region; non-negative w/h.
		reg := store.GetValues("regional_bars")
		regLast := reg[len(reg)-1]
		if len(regLast) != 4*len(grid.Regions) {
			t.Errorf("regional_bars width=%d, want %d", len(regLast), 4*len(grid.Regions))
		}
		for i := 0; i+3 < len(regLast); i += 4 {
			if regLast[i+2] < 0 || regLast[i+3] < 0 {
				t.Errorf("region bar %d has negative size (w=%v, h=%v)",
					i/4, regLast[i+2], regLast[i+3])
			}
		}

		// distribution_bars: 4 floats per band, non-negative w/h, with
		// every bar at least the minimum-marker height so empty bands
		// remain visible.
		dist := store.GetValues("distribution_bars")
		distLast := dist[len(dist)-1]
		if len(distLast) != 4*len(grid.Bands) {
			t.Errorf("distribution_bars width=%d, want %d", len(distLast), 4*len(grid.Bands))
		}
		for i := 0; i+3 < len(distLast); i += 4 {
			if distLast[i+2] <= 0 || distLast[i+3] <= 0 {
				t.Errorf("dist bar %d has non-positive size (w=%v, h=%v)",
					i/4, distLast[i+2], distLast[i+3])
			}
		}

		// baseline_line: single 4-float rectangle.
		base := store.GetValues("baseline_line")
		baseLast := base[len(base)-1]
		if len(baseLast) != 4 {
			t.Errorf("baseline_line width=%d, want 4", len(baseLast))
		}
		if baseLast[3] <= 0 {
			t.Errorf("baseline_line height should be positive, got %v", baseLast[3])
		}
	})
}

func TestGridInterpolation(t *testing.T) {
	grid := LoadEmbeddedGrid()

	t.Run("knot points return exact values", func(t *testing.T) {
		c, f, d := grid.ClimateAxis[0], grid.FlowAxis[0], grid.DOAxis[0]
		in := grid.interpolator(c, f, d)
		median, lo, hi := grid.trajSlice(in)
		if math.IsNaN(median[0]) || math.IsNaN(lo[0]) || math.IsNaN(hi[0]) {
			t.Errorf("knot lookup returned NaN: median=%v lo=%v hi=%v",
				median[0], lo[0], hi[0])
		}
		// At a knot, the interpolated trajectory must exactly match
		// the stored slice (no rounding drift from blending).
		off := in.trajOffset(0, 0, 0)
		if median[0] != grid.Median[off] {
			t.Errorf("knot median mismatch: got %v, want %v", median[0], grid.Median[off])
		}
	})

	t.Run("queries clamp to axis bounds", func(t *testing.T) {
		// A query above the climate max should clamp to the last
		// knot's trajectory.
		last := grid.ClimateAxis[len(grid.ClimateAxis)-1]
		clamped := grid.interpolator(last+10, 0, 0)
		exact := grid.interpolator(last, 0, 0)
		mc, _, _ := grid.trajSlice(clamped)
		me, _, _ := grid.trajSlice(exact)
		if math.Abs(mc[0]-me[0]) > 1e-9 {
			t.Errorf("clamp mismatch: %v vs %v", mc[0], me[0])
		}
	})
}
