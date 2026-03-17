package simulate

// CovariateModifier specifies additive shifts to environmental covariates.
// Flow and DO shifts are computed as fractional changes relative to each
// site's historical mean (e.g., FlowPct=0.15 means +15% flow).
// Temperature shift is absolute (degrees Celsius).
type CovariateModifier struct {
	TempShift float64 // additive shift to mean temperature (°C)
	FlowPct   float64 // fractional change in mean flow (e.g., 0.15 = +15%)
	DOPct     float64 // fractional change in dissolved oxygen (e.g., 0.15 = +15%)
}

// Scenario defines a policy simulation scenario.
type Scenario struct {
	Name        string
	Description string
	Modifier    CovariateModifier
}

// Baseline returns a null scenario (no intervention) for comparison.
func Baseline() Scenario {
	return Scenario{
		Name:        "baseline",
		Description: "No intervention — historical covariate distribution",
	}
}

// ClimateChange1C returns a +1°C warming scenario.
func ClimateChange1C() Scenario {
	return Scenario{
		Name:        "climate_1c",
		Description: "Climate change: +1°C mean annual water temperature",
		Modifier:    CovariateModifier{TempShift: 1.0},
	}
}

// ClimateChange2C returns a +2°C warming scenario.
func ClimateChange2C() Scenario {
	return Scenario{
		Name:        "climate_2c",
		Description: "Climate change: +2°C mean annual water temperature",
		Modifier:    CovariateModifier{TempShift: 2.0},
	}
}

// ReducedAbstraction returns a scenario with +15% river flow from
// reduced water abstraction.
func ReducedAbstraction() Scenario {
	return Scenario{
		Name:        "low_abstraction",
		Description: "Reduced abstraction: +15% mean river flow",
		Modifier:    CovariateModifier{FlowPct: 0.15},
	}
}

// Drought returns a scenario with -25% river flow.
func Drought() Scenario {
	return Scenario{
		Name:        "drought",
		Description: "Drought: -25% mean river flow",
		Modifier:    CovariateModifier{FlowPct: -0.25},
	}
}

// WaterQualityImprovement returns a scenario with +15% dissolved oxygen
// from pollution reduction.
func WaterQualityImprovement() Scenario {
	return Scenario{
		Name:        "wq_improvement",
		Description: "Water quality improvement: +15% dissolved oxygen",
		Modifier:    CovariateModifier{DOPct: 0.15},
	}
}

// CombinedClimateOxygenation returns a combined scenario: +2°C warming
// mitigated by +15% dissolved oxygen improvement.
func CombinedClimateOxygenation() Scenario {
	return Scenario{
		Name:        "combined_2c_oxy",
		Description: "Combined: +2°C warming with +15% DO improvement",
		Modifier:    CovariateModifier{TempShift: 2.0, DOPct: 0.15},
	}
}

// AllScenarios returns all predefined scenarios.
func AllScenarios() []Scenario {
	return []Scenario{
		Baseline(),
		ClimateChange1C(),
		ClimateChange2C(),
		ReducedAbstraction(),
		Drought(),
		WaterQualityImprovement(),
		CombinedClimateOxygenation(),
	}
}

// ScenarioByName looks up a predefined scenario by name.
// Returns the scenario and true if found, zero value and false otherwise.
func ScenarioByName(name string) (Scenario, bool) {
	for _, s := range AllScenarios() {
		if s.Name == name {
			return s, true
		}
	}
	return Scenario{}, false
}
