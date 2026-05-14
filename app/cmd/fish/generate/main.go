// generate emits the fish widget shell (widget.html, test.html,
// build.sh) into app/fish/. Re-run whenever the dashboard.Config in
// pkg/fish changes shape (controls, partitions, visualisation).
//
//	cd app && go run ./cmd/fish/generate
//
// After codegen, the emitted HTML is rewritten to:
//   - swap dexetera's default-blue slider accent for the explainer
//     collection's magenta;
//   - inject DOM captions and tick labels around the canvas (the
//     renderer's text element uses white-on-canvas and would be
//     invisible against the light background);
//   - inject the "pre-computed scenarios" honesty note that the post's
//     plan calls for, distinguishing this widget from rugby's live-wasm
//     simulation;
//   - inject the one-shot "flow has a small effect" hint that fires
//     when the reader drags an inert slider for the first time.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/umbralcalc/anglersim/app/pkg/fish"
	"github.com/umbralcalc/dexetera/pkg/dashboard"
)

// actionColor is the magenta from the Acting on Simulated Systems
// collection — used to signal "this is what the reader controls".
// Replaces dexetera's default blue (#3c78d8) on the slider track and
// the slider readout text.
const actionColor = "#b0447a"

func main() {
	runtimeURL := flag.String("runtime-url", "",
		"absolute URL the blog will serve dexetera's runtime/ folder from "+
			"(e.g. https://example.com/assets/dexetera/runtime/). "+
			"Leave empty for local preview via test.html.")
	wasmURL := flag.String("wasm-url", "",
		"absolute URL the blog will serve main.wasm from. "+
			"Leave empty for local preview.")
	flag.Parse()

	cfg := fish.NewConfig()
	dashboard.MustGenerateWidget(cfg, dashboard.WidgetOptions{
		RuntimeBaseURL: *runtimeURL,
		WasmURL:        *wasmURL,
	})

	for _, name := range []string{"widget.html", "test.html"} {
		path := filepath.Join(cfg.Name, name)
		if err := rewriteWidget(path); err != nil {
			fmt.Fprintf(os.Stderr, "rewrite %s: %v\n", path, err)
			os.Exit(1)
		}
	}
}

// rewriteWidget applies the post-codegen patches to one emitted HTML
// file. Replacements are anchored on enough surrounding context that
// they don't accidentally touch unrelated occurrences.
func rewriteWidget(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	out := string(data)
	widgetID := extractWidgetID(out)

	// 1. Swap dexetera's blue slider accent + readout for the action
	//    magenta. Two CSS rules, anchored on enough context to be
	//    unambiguous.
	out = mustReplace(out,
		"accent-color: #3c78d8",
		"accent-color: "+actionColor,
	)
	out = mustReplace(out,
		".slider-readout { grid-area: readout; text-align: right; color: #3c78d8;",
		".slider-readout { grid-area: readout; text-align: right; color: "+actionColor+";",
	)

	// 2. Inject the additional CSS for canvas captions, the honesty
	//    note, and the slider hint. Insert just before the closing
	//    </style> of the widget's scoped stylesheet.
	extraCSS := strings.ReplaceAll(extraCSSTemplate, "{{.WidgetID}}", widgetID)
	out = mustReplace(out, "</style>", extraCSS+"</style>")

	// 3. Inject DOM captions around the canvas. The trajectory chart
	//    sits above the regional bars, which sit above the
	//    distribution bars; captions are emitted in render order so
	//    the reader scans top-to-bottom naturally.
	out = mustReplace(out,
		fmt.Sprintf(`<canvas width="%d" height="%d"></canvas>`, fish.CanvasWidth, fish.CanvasHeight),
		captionMarkup(fish.CanvasWidth, fish.CanvasHeight),
	)

	// 4. Inject the "pre-computed scenarios" honesty note just below
	//    the widget description. The note explains why this widget
	//    differs from rugby's live-wasm pattern — the rugby disclaimer
	//    SVG sits *above* the widget on the post and doesn't carry the
	//    pre-computed caveat.
	out = mustReplace(out,
		`</p>`+"\n"+`<div class="dashboard">`,
		`</p>`+precomputedNote+"\n"+`<div class="dashboard">`,
	)

	// 5. Inject the one-shot inert-slider hint markup + JS. The hint
	//    fires the first time the reader drags the flow or DO slider
	//    and auto-dismisses after a few seconds.
	out = mustReplace(out,
		`<p class="status" data-status>`,
		inertSliderHintMarkup+`<p class="status" data-status>`,
	)
	out = mustReplace(out,
		`publishActions();
        startWorker(renderer);`,
		`publishActions();
        startWorker(renderer);
        wireInertSliderHint();`,
	)
	// Anchor on the closing tag — there's only one </script> in the
	// emitted snippet, so this targets the IIFE wrapping the whole
	// widget rather than the inner slidersByPartition IIFE that also
	// ends with `})();`.
	out = mustReplace(out,
		"})();\n</script>",
		inertSliderHintScript+"\n})();\n</script>",
	)

	return os.WriteFile(path, []byte(out), 0644)
}

// mustReplace replaces the first occurrence of old with new, returning
// the result or panicking if the anchor is missing — codegen drift
// breaking these patches should be a hard error, not a silent skip.
func mustReplace(s, old, new string) string {
	if !strings.Contains(s, old) {
		panic(fmt.Sprintf("expected fragment not found: %q", old))
	}
	return strings.Replace(s, old, new, 1)
}

func extractWidgetID(html string) string {
	const marker = `id="`
	i := strings.Index(html, marker)
	if i < 0 {
		return "dexetera"
	}
	i += len(marker)
	end := strings.Index(html[i:], `"`)
	if end < 0 {
		return "dexetera"
	}
	return html[i : i+end]
}

// captionMarkup wraps the canvas with no DOM captions — every panel
// title and bar label sits on the canvas itself (via AddText in
// pkg/fish/fish.go) so all three panels share the same titling
// convention and each label set stays unambiguously associated with
// its panel.
func captionMarkup(canvasWidth, canvasHeight int) string {
	return fmt.Sprintf(`<canvas width="%d" height="%d"></canvas>`, canvasWidth, canvasHeight)
}

const extraCSSTemplate = `#{{.WidgetID}} .canvas-caption { margin: 0; font-size: 0.85rem; color: #2c3e50; opacity: 0.75; text-align: center; }
#{{.WidgetID}} .canvas-caption-top { margin-bottom: 0.1em; }
#{{.WidgetID}} .precomputed-note { margin: -0.4em 0 1em; padding: 0.5em 0.8em; background: rgba(60,120,216,0.06); border-left: 3px solid #3c78d8; color: #2c3e50; opacity: 0.85; font-size: 0.9rem; }
#{{.WidgetID}} .slider-hint { margin: 0.4em 0 0; padding: 0.5em 0.7em; background: rgba(176,68,122,0.1); border-left: 3px solid ` + actionColor + `; color: #2c3e50; font-size: 0.9rem; opacity: 0; max-height: 0; overflow: hidden; transition: opacity 0.3s ease, max-height 0.3s ease, margin 0.3s ease; }
#{{.WidgetID}} .slider-hint.visible { opacity: 1; max-height: 6em; margin-top: 0.6em; }
`

const precomputedNote = `<p class="precomputed-note">Note: the projections shown are pre-computed scenarios from the fitted model — the dashboard interpolates between cells in a static grid rather than running a fresh simulation each tick.</p>`

const inertSliderHintMarkup = `<p class="slider-hint" data-slider-hint>Flow and dissolved-oxygen sliders barely shift the projection in this model — try the climate slider for the dominant signal.</p>
    `

const inertSliderHintScript = `
    function wireInertSliderHint() {
        var hint = $('[data-slider-hint]');
        if (!hint) return;
        var inert = ['flow_pct', 'do_pct'];
        var shown = false;
        var dismissTimer = null;
        function show() {
            if (shown) return;
            shown = true;
            hint.classList.add('visible');
            dismissTimer = setTimeout(function () { hint.classList.remove('visible'); }, 7000);
        }
        for (var i = 0; i < inert.length; i++) {
            var el = $('[data-slider="' + inert[i] + '"]');
            if (el) el.addEventListener('input', show, { once: true });
        }
    }
`
