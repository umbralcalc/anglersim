package inference

import (
	"math"
	"math/rand/v2"
	"testing"

	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
	"gonum.org/v1/gonum/floats"
)

func TestUniformPrior(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	p := &stdinf.UniformPrior{Lo: -1.0, Hi: 2.0}
	n := 10000
	samples := make([]float64, n)
	for i := range n {
		samples[i] = p.Sample(rng)
		if !p.InSupport(samples[i]) {
			t.Fatalf("sample %f out of support", samples[i])
		}
	}
	mean := floats.Sum(samples) / float64(n)
	expectedMean := 0.5
	if math.Abs(mean-expectedMean) > 0.1 {
		t.Errorf("mean %.4f, expected ~%.4f", mean, expectedMean)
	}

	// LogPDF in support
	lp := p.LogPDF(0.5)
	expected := -math.Log(3.0)
	if math.Abs(lp-expected) > 1e-10 {
		t.Errorf("LogPDF(0.5)=%.6f, expected %.6f", lp, expected)
	}
	// LogPDF outside support
	if !math.IsInf(p.LogPDF(-2.0), -1) {
		t.Error("expected -Inf outside support")
	}
}

func TestTruncatedNormalPrior(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	p := &stdinf.TruncatedNormalPrior{Mu: 0.0, Sigma: 1.0, Lo: -3.0, Hi: 3.0}
	n := 10000
	samples := make([]float64, n)
	for i := range n {
		samples[i] = p.Sample(rng)
		if !p.InSupport(samples[i]) {
			t.Fatalf("sample %f out of support", samples[i])
		}
	}
	mean := floats.Sum(samples) / float64(n)
	if math.Abs(mean) > 0.1 {
		t.Errorf("mean %.4f, expected ~0.0", mean)
	}
	if !math.IsInf(p.LogPDF(5.0), -1) {
		t.Error("expected -Inf outside support")
	}
}

func TestHalfNormalPrior(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	p := &stdinf.HalfNormalPrior{Sigma: 1.0}
	n := 10000
	for range n {
		s := p.Sample(rng)
		if s < 0 {
			t.Fatalf("negative sample %f", s)
		}
	}
	if !math.IsInf(p.LogPDF(-0.1), -1) {
		t.Error("expected -Inf for negative x")
	}
	// PDF at x=0 should be finite
	if math.IsInf(p.LogPDF(0.0), 0) {
		t.Error("expected finite LogPDF at 0")
	}
}

func TestLogNormalPrior(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	p := &stdinf.LogNormalPrior{Mu: 0.0, Sigma: 1.0}
	n := 10000
	for range n {
		s := p.Sample(rng)
		if s <= 0 {
			t.Fatalf("non-positive sample %f", s)
		}
	}
	if !math.IsInf(p.LogPDF(0.0), -1) {
		t.Error("expected -Inf at x=0")
	}
	if !math.IsInf(p.LogPDF(-1.0), -1) {
		t.Error("expected -Inf for negative x")
	}
	// LogPDF at x=1 with mu=0,sigma=1: log(1)=0, so d=0, logpdf = -0 - 0 - 0 - 0.5*log(2pi)
	lp := p.LogPDF(1.0)
	expected := -0.5 * math.Log(2*math.Pi)
	if math.Abs(lp-expected) > 1e-10 {
		t.Errorf("LogPDF(1.0)=%.6f, expected %.6f", lp, expected)
	}
}

func TestDefaultRickerPriors(t *testing.T) {
	priors := DefaultRickerPriors()
	if len(priors) != 8 {
		t.Fatalf("expected 8 priors, got %d", len(priors))
	}
	rng := rand.New(rand.NewPCG(99, 100))
	for i, p := range priors {
		s := p.Sample(rng)
		if !p.InSupport(s) {
			t.Errorf("prior %d: sample %f not in support", i, s)
		}
		lp := p.LogPDF(s)
		if math.IsInf(lp, -1) || math.IsNaN(lp) {
			t.Errorf("prior %d: LogPDF(%f) = %f", i, s, lp)
		}
	}
}
