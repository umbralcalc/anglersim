package population

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/general"
	"github.com/umbralcalc/stochadex/pkg/simulator"
	"gonum.org/v1/gonum/floats/scalar"
)

func TestRickerIteration(t *testing.T) {
	t.Run(
		"test that ricker runs and produces sensible output",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml(
				"./ricker_settings.yaml",
			)
			// Zero noise for deterministic test
			settings.Iterations[1].Params.Map["process_noise_sd"] = []float64{0.0}

			iterations := []simulator.Iteration{
				&general.ParamValuesIteration{},
				&RickerIteration{},
			}
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}
			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 50,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
			}
			coordinator := simulator.NewPartitionCoordinator(
				settings,
				implementations,
			)
			coordinator.Run()

			// Verify first step manually:
			// logN0 = -2.0, r0 = 0.5, alpha = 1.0
			// covs = [0.5, 12.0, 9.0], betas = [0.1, -0.02, 0.05]
			// envEffect = 0.1*0.5 + (-0.02)*12.0 + 0.05*9.0 = 0.05 - 0.24 + 0.45 = 0.26
			// logN1 = -2.0 + 0.5 + 0.26 - 1.0*exp(-2.0)
			//       = -2.0 + 0.5 + 0.26 - 0.1353... = -1.3753...
			logN0 := -2.0
			r0 := 0.5
			alpha := 1.0
			envEffect := 0.1*0.5 + (-0.02)*12.0 + 0.05*9.0
			expectedLogN1 := logN0 + r0 + envEffect - alpha*math.Exp(logN0)

			// Population partition is index 1
			popStates := store.GetValues("population")
			if len(popStates) < 2 {
				t.Fatalf("expected at least 2 stored states, got %d", len(popStates))
			}
			// popStates[0] is the initial state, popStates[1] is after first step
			gotLogN1 := popStates[1][0]
			if !scalar.EqualWithinAbsOrRel(gotLogN1, expectedLogN1, 1e-10, 1e-10) {
				t.Errorf("step 1: got logN=%v, want %v", gotLogN1, expectedLogN1)
			}

			// After many deterministic steps the population should reach
			// an equilibrium where logN stabilises (not NaN or ±Inf)
			lastLogN := popStates[len(popStates)-1][0]
			if math.IsNaN(lastLogN) || math.IsInf(lastLogN, 0) {
				t.Errorf("population diverged: final logN=%v", lastLogN)
			}
		},
	)
	t.Run(
		"test that ricker runs with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml(
				"./ricker_settings.yaml",
			)
			iterations := []simulator.Iteration{
				&general.ParamValuesIteration{},
				&RickerIteration{},
			}
			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 100,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
			}
			if err := simulator.RunWithHarnesses(settings, implementations); err != nil {
				t.Errorf("test harness failed: %v", err)
			}
		},
	)
}
