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
	flowDir := flag.String("flow-dir", "dat/hydrology", "output directory for cached flow data")
	wqDir := flag.String("wq-dir", "dat/water_quality", "output directory for cached WQ data")
	outFile := flag.String("out", "dat/brown_trout_panel_with_covariates.csv", "output panel CSV with all covariates")
	maxDist := flag.Float64("dist", 10.0, "max distance (km) for station matching")
	minYears := flag.Int("min-years", 20, "minimum survey years for site inclusion")
	flag.Parse()

	// Create output directories
	for _, dir := range []string{*flowDir, *wqDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("creating output dir %s: %v", dir, err)
		}
	}

	// Step 1: Load sites and filter by min years
	sites := loadSites(*sitesFile, *minYears)
	log.Printf("Loaded %d sites with >= %d survey years", len(sites), *minYears)

	// Step 2: Match sites to flow stations (or load cached mapping)
	flowMappingFile := filepath.Join(*flowDir, "site_flow_mapping.csv")
	flowMappings := matchOrLoadFlowStations(sites, flowMappingFile, *maxDist)
	logFlowMatchStats(flowMappings, len(sites))

	// Step 3: Fetch daily flow readings (with per-station CSV cache)
	fetchFlowReadings(flowMappings, *flowDir)

	// Step 4: Match sites to WQ sampling points (or load cached mapping)
	wqMappingFile := filepath.Join(*wqDir, "site_wq_mapping.csv")
	wqMappings := matchOrLoadWQPoints(sites, wqMappingFile, *maxDist)
	logWQMatchStats(wqMappings, len(sites))

	// Step 5: Fetch WQ observations (with per-point CSV cache)
	fetchWQObservations(wqMappings, *wqDir)

	// Step 6: Build annual stats and join everything to panel
	joinCovariates(*panelFile, flowMappings, *flowDir, wqMappings, *wqDir, *outFile)
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

type wqMapping struct {
	siteID   int
	siteName string
	notation string
	pointName string
	distKm   float64
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

// --- Flow station matching and fetching ---

func matchOrLoadFlowStations(sites []site, mappingFile string, maxDistKm float64) []flowMapping {
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

func logFlowMatchStats(mappings []flowMapping, totalSites int) {
	matched := 0
	uniqueStations := make(map[string]bool)
	for _, m := range mappings {
		if m.measureID != "" {
			matched++
			uniqueStations[m.measureID] = true
		}
	}
	log.Printf("Flow: matched %d/%d sites to %d unique stations", matched, totalSites, len(uniqueStations))
}

func fetchFlowReadings(mappings []flowMapping, outDir string) {
	uniqueMeasures := make(map[string]string)
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

		if _, err := os.Stat(stationFile); err == nil {
			done++
			continue
		}

		log.Printf("  [%d/%d] Fetching %s (%s)...", done+1, total, label, measureID)

		readings := data.GetDailyFlowReadings(measureID, "1970-01-01", "2025-12-31")

		if len(readings) > 0 {
			saveFlowReadings(stationFile, readings)
			log.Printf("    -> %d readings saved", len(readings))
		} else {
			saveFlowReadings(stationFile, nil)
			log.Printf("    -> no readings")
		}

		done++
		time.Sleep(500 * time.Millisecond)
	}

	log.Printf("All flow readings fetched/cached")
}

func saveFlowReadings(path string, readings []data.HydrologyReading) {
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

func loadFlowReadings(path string) []data.HydrologyReading {
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

// --- WQ sampling point matching and fetching ---

func matchOrLoadWQPoints(sites []site, mappingFile string, maxDistKm float64) []wqMapping {
	if mappings, err := loadWQMapping(mappingFile); err == nil && len(mappings) == len(sites) {
		log.Printf("Loaded cached WQ mapping from %s", mappingFile)
		return mappings
	}

	log.Printf("Matching %d sites to WQ sampling points (%.0f km radius)...", len(sites), maxDistKm)
	mappings := make([]wqMapping, len(sites))

	for i, s := range sites {
		pt := data.FindNearestWQPoint(s.easting, s.northing, maxDistKm)
		mappings[i] = wqMapping{
			siteID:   s.id,
			siteName: s.name,
		}
		if pt != nil {
			mappings[i].notation = pt.Notation
			mappings[i].pointName = pt.Name
			mappings[i].distKm = pt.DistKm
		}

		if (i+1)%10 == 0 {
			log.Printf("  matched %d/%d sites", i+1, len(sites))
			time.Sleep(500 * time.Millisecond)
		}
	}

	saveWQMapping(mappingFile, mappings)
	return mappings
}

func loadWQMapping(path string) ([]wqMapping, error) {
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

	var mappings []wqMapping
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		id, _ := strconv.Atoi(record[idx["SITE_ID"]])
		dist, _ := strconv.ParseFloat(record[idx["DIST_KM"]], 64)
		mappings = append(mappings, wqMapping{
			siteID:    id,
			siteName:  record[idx["SITE_NAME"]],
			notation:  record[idx["WQ_NOTATION"]],
			pointName: record[idx["WQ_POINT_NAME"]],
			distKm:    dist,
		})
	}
	return mappings, nil
}

func saveWQMapping(path string, mappings []wqMapping) {
	f, err := os.Create(path)
	if err != nil {
		log.Printf("Warning: could not save WQ mapping: %v", err)
		return
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"SITE_ID", "SITE_NAME", "WQ_NOTATION", "WQ_POINT_NAME", "DIST_KM"})
	for _, m := range mappings {
		w.Write([]string{
			strconv.Itoa(m.siteID),
			m.siteName,
			m.notation,
			m.pointName,
			fmt.Sprintf("%.3f", m.distKm),
		})
	}
	w.Flush()
	log.Printf("Saved WQ mapping to %s", path)
}

func logWQMatchStats(mappings []wqMapping, totalSites int) {
	matched := 0
	uniquePoints := make(map[string]bool)
	for _, m := range mappings {
		if m.notation != "" {
			matched++
			uniquePoints[m.notation] = true
		}
	}
	log.Printf("WQ: matched %d/%d sites to %d unique sampling points", matched, totalSites, len(uniquePoints))
}

func fetchWQObservations(mappings []wqMapping, outDir string) {
	// Collect unique sampling point notations
	uniquePoints := make(map[string]string) // notation -> pointName
	for _, m := range mappings {
		if m.notation != "" {
			uniquePoints[m.notation] = m.pointName
		}
	}

	determinands := []struct {
		code string
		name string
	}{
		{data.DetTemperature, "temp"},
		{data.DetDissolvedO2, "do"},
		{data.DetAmmonia, "nh3"},
		{data.DetBOD, "bod"},
	}

	log.Printf("Fetching WQ observations for %d unique sampling points...", len(uniquePoints))

	done := 0
	total := len(uniquePoints)
	for notation, name := range uniquePoints {
		// Check if all 4 determinand files exist (use temp as sentinel)
		sentinel := filepath.Join(outDir, sanitizeFilename(notation)+"_temp.csv")
		if _, err := os.Stat(sentinel); err == nil {
			done++
			continue
		}

		log.Printf("  [%d/%d] Fetching %s (%s)...", done+1, total, name, notation)

		for _, det := range determinands {
			obs := data.GetWQObservations(notation, det.code, "1970-01-01", "2025-12-31")
			outFile := filepath.Join(outDir, sanitizeFilename(notation)+"_"+det.name+".csv")
			saveWQObs(outFile, obs)
			log.Printf("    %s: %d observations", det.name, len(obs))
			time.Sleep(200 * time.Millisecond)
		}

		done++
	}

	log.Printf("All WQ observations fetched/cached")
}

func saveWQObs(path string, obs []data.WQObservation) {
	f, err := os.Create(path)
	if err != nil {
		log.Printf("Warning: could not save WQ obs to %s: %v", path, err)
		return
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"DATE", "VALUE"})
	for _, o := range obs {
		w.Write([]string{o.Date, fmt.Sprintf("%.6f", o.Value)})
	}
	w.Flush()
}

func loadWQObs(path string) []data.WQObservation {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Read() // skip header

	var obs []data.WQObservation
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		val, _ := strconv.ParseFloat(record[1], 64)
		obs = append(obs, data.WQObservation{
			Date:  record[0],
			Value: val,
		})
	}
	return obs
}

// --- Join all covariates to panel ---

func joinCovariates(
	panelFile string,
	flowMappings []flowMapping, flowDir string,
	wqMappings []wqMapping, wqDir string,
	outFile string,
) {
	// Build flow stats lookup: (measureID, year) -> stats
	type yearKey struct {
		id   string
		year int
	}

	siteToMeasure := make(map[int]string)
	for _, m := range flowMappings {
		if m.measureID != "" {
			siteToMeasure[m.siteID] = m.measureID
		}
	}

	flowStats := make(map[yearKey]data.AnnualFlowStats)
	processedMeasures := make(map[string]bool)
	for _, m := range flowMappings {
		if m.measureID == "" || processedMeasures[m.measureID] {
			continue
		}
		processedMeasures[m.measureID] = true

		stationFile := filepath.Join(flowDir, sanitizeFilename(m.measureID)+".csv")
		readings := loadFlowReadings(stationFile)
		if len(readings) == 0 {
			continue
		}

		for _, s := range data.AggregateAnnualFlow(readings) {
			flowStats[yearKey{m.measureID, s.Year}] = s
		}
	}

	// Build WQ stats lookup: (notation, year) -> stats
	siteToNotation := make(map[int]string)
	for _, m := range wqMappings {
		if m.notation != "" {
			siteToNotation[m.siteID] = m.notation
		}
	}

	wqStats := make(map[yearKey]data.AnnualWQStats)
	processedNotations := make(map[string]bool)
	for _, m := range wqMappings {
		if m.notation == "" || processedNotations[m.notation] {
			continue
		}
		processedNotations[m.notation] = true

		prefix := filepath.Join(wqDir, sanitizeFilename(m.notation))
		tempObs := loadWQObs(prefix + "_temp.csv")
		doObs := loadWQObs(prefix + "_do.csv")
		nh3Obs := loadWQObs(prefix + "_nh3.csv")
		bodObs := loadWQObs(prefix + "_bod.csv")

		for _, s := range data.AggregateAnnualWQ(tempObs, doObs, nh3Obs, bodObs) {
			wqStats[yearKey{m.notation, s.Year}] = s
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
	of, err := os.Create(outFile)
	if err != nil {
		log.Fatalf("creating output: %v", err)
	}
	defer of.Close()

	w := csv.NewWriter(of)

	flowCols := []string{"MEAN_FLOW", "MAX_FLOW", "MIN_FLOW", "Q10_FLOW", "Q90_FLOW", "FLOW_NUM_DAYS"}
	wqCols := []string{"MEAN_TEMP", "MAX_TEMP", "MEAN_DO", "MIN_DO", "MEAN_AMMONIA", "MAX_AMMONIA", "MEAN_BOD", "MAX_BOD"}
	allNewCols := append(flowCols, wqCols...)
	w.Write(append(panelHeaders, allNewCols...))

	flowJoined := 0
	wqJoined := 0
	total := 0

	emptyFlow := make([]string, len(flowCols))
	emptyWQ := make([]string, len(wqCols))

	for _, row := range panelRows {
		siteID, _ := strconv.Atoi(row[pidx["SITE_ID"]])
		year, _ := strconv.Atoi(row[pidx["YEAR"]])

		// Flow columns
		var flowVals []string
		if mid := siteToMeasure[siteID]; mid != "" {
			if s, ok := flowStats[yearKey{mid, year}]; ok {
				flowVals = []string{
					fmt.Sprintf("%.6f", s.MeanFlow),
					fmt.Sprintf("%.6f", s.MaxFlow),
					fmt.Sprintf("%.6f", s.MinFlow),
					fmt.Sprintf("%.6f", s.Q10Flow),
					fmt.Sprintf("%.6f", s.Q90Flow),
					strconv.Itoa(s.NumDays),
				}
				flowJoined++
			}
		}
		if flowVals == nil {
			flowVals = emptyFlow
		}

		// WQ columns
		var wqVals []string
		if notation := siteToNotation[siteID]; notation != "" {
			if s, ok := wqStats[yearKey{notation, year}]; ok {
				wqVals = []string{
					fmtNaN(s.MeanTemp),
					fmtNaN(s.MaxTemp),
					fmtNaN(s.MeanDO),
					fmtNaN(s.MinDO),
					fmtNaN(s.MeanAmmonia),
					fmtNaN(s.MaxAmmonia),
					fmtNaN(s.MeanBOD),
					fmtNaN(s.MaxBOD),
				}
				wqJoined++
			}
		}
		if wqVals == nil {
			wqVals = emptyWQ
		}

		outRow := append(row, flowVals...)
		outRow = append(outRow, wqVals...)
		w.Write(outRow)
		total++
	}

	w.Flush()

	log.Printf("Joined panel: %d/%d rows have flow data, %d/%d have WQ data -> %s",
		flowJoined, total, wqJoined, total, outFile)

	// Print distance summaries
	printDistStats("flow station", flowDistances(flowMappings))
	printDistStats("WQ sampling point", wqDistances(wqMappings))
}

func fmtNaN(v float64) string {
	if math.IsNaN(v) {
		return ""
	}
	return fmt.Sprintf("%.4f", v)
}

func flowDistances(mappings []flowMapping) []float64 {
	var dists []float64
	for _, m := range mappings {
		if m.measureID != "" {
			dists = append(dists, m.distMetres)
		}
	}
	return dists
}

func wqDistances(mappings []wqMapping) []float64 {
	var dists []float64
	for _, m := range mappings {
		if m.notation != "" {
			dists = append(dists, m.distKm*1000) // convert to metres for consistency
		}
	}
	return dists
}

func printDistStats(label string, dists []float64) {
	sort.Float64s(dists)
	if len(dists) > 0 {
		log.Printf("Distance to nearest %s (m): median=%.0f, max=%.0f",
			label, dists[len(dists)/2], dists[len(dists)-1])
	}
}

func sanitizeFilename(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "/", "_"), ":", "_")
}
