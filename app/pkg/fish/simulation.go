package fish

import (
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// MaxSteps caps the simulation length. Because every visualization
// partition recomputes its rectangle layout from the latest sliders
// each step, there is no accumulation that needs many steps to settle —
// any positive cap is fine. A large value just lets the dashboard run
// indefinitely without restarting.
const MaxSteps = 1_000_000

// BuildFishSimulation builds the stochadex ConfigGenerator for the fish
// dashboard. The grid lookup is purely a function of the slider state,
// so every iteration is stateless on its own state history; the engine
// loop exists only to give the wasm runtime something to call.
func BuildFishSimulation() *simulator.ConfigGenerator {
	grid := LoadEmbeddedGrid()

	scenarioSnapshot := &simulator.PartitionConfig{
		Name:      "scenario_snapshot",
		Iteration: &ScenarioSnapshotIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"action_state_values": append([]float64(nil), DefaultScenario...),
		}),
		InitStateValues:   append([]float64(nil), DefaultScenario...),
		StateHistoryDepth: 1,
		Seed:              0,
	}

	trajBand := &simulator.PartitionConfig{
		Name:      "trajectory_band",
		Iteration: &TrajectoryBandIteration{Grid: grid},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"scenario_values": {Upstream: "scenario_snapshot"},
		},
		InitStateValues:   make([]float64, 4*grid.MaxHorizon),
		StateHistoryDepth: 1,
		Seed:              0,
	}

	trajMedian := &simulator.PartitionConfig{
		Name:      "trajectory_median",
		Iteration: &TrajectoryMedianIteration{Grid: grid},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"scenario_values": {Upstream: "scenario_snapshot"},
		},
		InitStateValues:   make([]float64, MedianStateWidth(grid.MaxHorizon)),
		StateHistoryDepth: 1,
		Seed:              0,
	}

	regionalBars := &simulator.PartitionConfig{
		Name:      "regional_bars",
		Iteration: &RegionalBarsIteration{Grid: grid},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"scenario_values": {Upstream: "scenario_snapshot"},
		},
		InitStateValues:   make([]float64, 4*len(grid.Regions)),
		StateHistoryDepth: 1,
		Seed:              0,
	}

	distributionBars := &simulator.PartitionConfig{
		Name:      "distribution_bars",
		Iteration: &DistributionBarsIteration{Grid: grid},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"scenario_values": {Upstream: "scenario_snapshot"},
		},
		InitStateValues:   make([]float64, 4*len(grid.Bands)),
		StateHistoryDepth: 1,
		Seed:              0,
	}

	baselineLine := &simulator.PartitionConfig{
		Name:              "baseline_line",
		Iteration:         &BaselineLineIteration{Grid: grid},
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   make([]float64, 4),
		StateHistoryDepth: 1,
		Seed:              0,
	}

	gen := simulator.NewConfigGenerator()
	for _, p := range []*simulator.PartitionConfig{
		scenarioSnapshot,
		trajBand,
		trajMedian,
		regionalBars,
		distributionBars,
		baselineLine,
	} {
		gen.SetPartition(p)
	}

	gen.SetSimulation(&simulator.SimulationConfig{
		OutputCondition:      &simulator.EveryStepOutputCondition{},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: MaxSteps},
		TimestepFunction:     &simulator.ConstantTimestepFunction{Stepsize: 1.0},
		InitTimeValue:        0.0,
	})
	return gen
}
