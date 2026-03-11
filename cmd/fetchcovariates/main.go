package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/umbralcalc/anglersim/pkg/data"
)

func main() {
	sitesFile := flag.String("sites", "dat/brown_trout_sites.csv", "path to brown_trout_sites.csv")
	panelFile := flag.String("panel", "dat/brown_trout_panel.csv", "path to brown_trout_panel.csv")
	outDir := flag.String("out", "dat/hydrology", "output directory for cached station data")
	maxDist := flag.Float64("dist", 10.0, "max distance (km) for station matching")
	minYears := flag.Int("min-years", 20, "minimum survey years for site inclusion")
	flag.Parse()

	// Create output directory
	if err := os.MkdirAll(*outDir, 0755); err != nil {
		log.Fatalf("creating output dir: %v", err)
	}

	// Step 1: Load sites and filter by min years
	sites := loadSites(*sitesFile, *minYears)
	log.Printf("Loaded %d sites with >= %d survey years", len(sites), *minYears)

	// Step 2: Match sites to flow stations (or load cached mapping)
	mappingFile := filepath.Join(*outDir, "site_flow_mapping.csv")
	mappings := matchOrLoadFlowStations(sites, mappingFile, *maxDist)

	matched := 0
	uniqueStations := make(map[string]bool)
	for _, m := range mappings {
		if m.measureID != "" {
			matched++
			uniqueStations[m.measureID] = true
		}
	}
	log.Printf("Matched %d/%d sites to %d unique flow stations", matched, len(sites), len(uniqueStations))

	// Step 3: Fetch daily readings for each unique station (with per-station CSV cache)
	fetchFlowReadings(mappings, *outDir)

	// Step 4: Build annual flow stats and join to panel
	joinFlowToPanel(*panelFile, mappings, *outDir)
}

type site struct {
	id       int
	name     string
	easting  int
	northing int
	numYears int
}

type flowMapping struct {
	siteID       int
	siteName     string
	stationLabel string
	measureID    string
	distMetres   float64
}

func loadSites(path string, minYears int) []site {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("opening sites: %v", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	headers, err := r.Read()
	if err != nil {
		log.Fatalf("reading header: %v", err)
	}

	idx := make(map[string]int)
	for i, h := range headers {
		idx[h] = i
	}

	var sites []site
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		ny, _ := strconv.Atoi(record[idx["NUM_YEARS"]])
		if ny < minYears {
			continue
		}
		id, _ := strconv.Atoi(record[idx["SITE_ID"]])
		e, _ := strconv.Atoi(record[idx["EASTING"]])
		n, _ := strconv.Atoi(record[idx["NORTHING"]])
		sites = append(sites, site{
			id:       id,
			name:     record[idx["SITE_NAME"]],
			easting:  e,
			northing: n,
			numYears: ny,
		})
	}
	return sites
}

func matchOrLoadFlowStations(sites []site, mappingFile string, maxDistKm float64) []flowMapping {
	// Try loading existing mapping
	if mappings, err := loadFlowMapping(mappingFile); err == nil && len(mappings) == len(sites) {
		log.Printf("Loaded cached flow mapping from %s", mappingFile)
		return mappings
	}

	log.Printf("Matching %d sites to flow stations (%.0f km radius)...", len(sites), maxDistKm)
	mappings := make([]flowMapping, len(sites))

	for i, s := range sites {
		station := data.FindNearestFlowStation(s.easting, s.northing, maxDistKm)
		mappings[i] = flowMapping{
			siteID:   s.id,
			siteName: s.name,
		}
		if station != nil {
			de := float64(station.Easting - s.easting)
			dn := float64(station.Northing - s.northing)
			mappings[i].stationLabel = station.Label
			mappings[i].measureID = station.DailyMeanFlowMeasureID()
			mappings[i].distMetres = math.Sqrt(de*de + dn*dn)
		}

		if (i+1)%10 == 0 {
			log.Printf("  matched %d/%d sites", i+1, len(sites))
			time.Sleep(500 * time.Millisecond)
		}
	}

	saveFlowMapping(mappingFile, mappings)
	return mappings
}

func loadFlowMapping(path string) ([]flowMapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	headers, err := r.Read()
	if err != nil {
		return nil, err
	}

	idx := make(map[string]int)
	for i, h := range headers {
		idx[h] = i
	}

	var mappings []flowMapping
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		id, _ := strconv.Atoi(record[idx["SITE_ID"]])
		dist, _ := strconv.ParseFloat(record[idx["DIST_METRES"]], 64)
		mappings = append(mappings, flowMapping{
			siteID:       id,
			siteName:     record[idx["SITE_NAME"]],
			stationLabel: record[idx["STATION_LABEL"]],
			measureID:    record[idx["MEASURE_ID"]],
			distMetres:   dist,
		})
	}
	return mappings, nil
}

func saveFlowMapping(path string, mappings []flowMapping) {
	f, err := os.Create(path)
	if err != nil {
		log.Printf("Warning: could not save mapping: %v", err)
		return
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"SITE_ID", "SITE_NAME", "STATION_LABEL", "MEASURE_ID", "DIST_METRES"})
	for _, m := range mappings {
		w.Write([]string{
			strconv.Itoa(m.siteID),
			m.siteName,
			m.stationLabel,
			m.measureID,
			fmt.Sprintf("%.1f", m.distMetres),
		})
	}
	w.Flush()
	log.Printf("Saved flow mapping to %s", path)
}

func fetchFlowReadings(mappings []flowMapping, outDir string) {
	// Collect unique measure IDs
	uniqueMeasures := make(map[string]string) // measureID -> stationLabel
	for _, m := range mappings {
		if m.measureID != "" {
			uniqueMeasures[m.measureID] = m.stationLabel
		}
	}

	log.Printf("Fetching daily readings for %d unique flow stations...", len(uniqueMeasures))

	done := 0
	total := len(uniqueMeasures)
	for measureID, label := range uniqueMeasures {
		stationFile := filepath.Join(outDir, sanitizeFilename(measureID)+".csv")

		// Skip if already cached
		if _, err := os.Stat(stationFile); err == nil {
			done++
			continue
		}

		log.Printf("  [%d/%d] Fetching %s (%s)...", done+1, total, label, measureID)

		readings := data.GetDailyFlowReadings(measureID, "1970-01-01", "2025-12-31")

		if len(readings) > 0 {
			saveReadings(stationFile, readings)
			log.Printf("    -> %d readings saved", len(readings))
		} else {
			// Save empty file so we don't retry
			saveReadings(stationFile, nil)
			log.Printf("    -> no readings")
		}

		done++
		time.Sleep(500 * time.Millisecond)
	}

	log.Printf("All flow readings fetched/cached")
}

func saveReadings(path string, readings []data.HydrologyReading) {
	f, err := os.Create(path)
	if err != nil {
		log.Printf("Warning: could not save readings to %s: %v", path, err)
		return
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"DATE", "VALUE"})
	for _, r := range readings {
		w.Write([]string{r.Date, fmt.Sprintf("%.6f", r.Value)})
	}
	w.Flush()
}

func loadReadings(path string) []data.HydrologyReading {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Read() // skip header

	var readings []data.HydrologyReading
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		val, _ := strconv.ParseFloat(record[1], 64)
		readings = append(readings, data.HydrologyReading{
			Date:  record[0],
			Value: val,
		})
	}
	return readings
}

func sanitizeFilename(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "/", "_"), ":", "_")
}

func joinFlowToPanel(panelFile string, mappings []flowMapping, outDir string) {
	// Build measureID -> site IDs lookup
	siteToMeasure := make(map[int]string)
	for _, m := range mappings {
		if m.measureID != "" {
			siteToMeasure[m.siteID] = m.measureID
		}
	}

	// Load and aggregate flow data per unique measure
	type yearKey struct {
		measureID string
		year      int
	}
	flowStats := make(map[yearKey]data.AnnualFlowStats)

	processedMeasures := make(map[string]bool)
	for _, m := range mappings {
		if m.measureID == "" || processedMeasures[m.measureID] {
			continue
		}
		processedMeasures[m.measureID] = true

		stationFile := filepath.Join(outDir, sanitizeFilename(m.measureID)+".csv")
		readings := loadReadings(stationFile)
		if len(readings) == 0 {
			continue
		}

		stats := data.AggregateAnnualFlow(readings)
		for _, s := range stats {
			flowStats[yearKey{m.measureID, s.Year}] = s
		}
	}

	// Load panel CSV
	pf, err := os.Open(panelFile)
	if err != nil {
		log.Fatalf("opening panel: %v", err)
	}
	defer pf.Close()

	pr := csv.NewReader(pf)
	panelHeaders, err := pr.Read()
	if err != nil {
		log.Fatalf("reading panel header: %v", err)
	}

	pidx := make(map[string]int)
	for i, h := range panelHeaders {
		pidx[h] = i
	}

	var panelRows [][]string
	for {
		record, err := pr.Read()
		if err != nil {
			break
		}
		panelRows = append(panelRows, record)
	}

	// Write joined output
	outFile := filepath.Join(outDir, "brown_trout_panel_with_flow.csv")
	of, err := os.Create(outFile)
	if err != nil {
		log.Fatalf("creating output: %v", err)
	}
	defer of.Close()

	w := csv.NewWriter(of)
	flowCols := []string{"MEAN_FLOW", "MAX_FLOW", "MIN_FLOW", "Q10_FLOW", "Q90_FLOW", "FLOW_NUM_DAYS"}
	w.Write(append(panelHeaders, flowCols...))

	joined := 0
	total := 0
	for _, row := range panelRows {
		siteID, _ := strconv.Atoi(row[pidx["SITE_ID"]])
		year, _ := strconv.Atoi(row[pidx["YEAR"]])

		measureID := siteToMeasure[siteID]
		var flowVals []string

		if measureID != "" {
			if s, ok := flowStats[yearKey{measureID, year}]; ok {
				flowVals = []string{
					fmt.Sprintf("%.6f", s.MeanFlow),
					fmt.Sprintf("%.6f", s.MaxFlow),
					fmt.Sprintf("%.6f", s.MinFlow),
					fmt.Sprintf("%.6f", s.Q10Flow),
					fmt.Sprintf("%.6f", s.Q90Flow),
					strconv.Itoa(s.NumDays),
				}
				joined++
			}
		}

		if flowVals == nil {
			flowVals = []string{"", "", "", "", "", ""}
		}

		w.Write(append(row, flowVals...))
		total++
	}

	w.Flush()

	// Also save site-level matching stats
	log.Printf("Joined panel: %d/%d rows have flow data -> %s", joined, total, outFile)

	// Print summary of distance distribution
	var dists []float64
	for _, m := range mappings {
		if m.measureID != "" {
			dists = append(dists, m.distMetres)
		}
	}
	sort.Float64s(dists)
	if len(dists) > 0 {
		log.Printf("Distance to nearest flow station (m): median=%.0f, max=%.0f",
			dists[len(dists)/2], dists[len(dists)-1])
	}
}
