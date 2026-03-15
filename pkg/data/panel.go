package data

import (
	"bufio"
	"encoding/csv"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

// SiteData holds the time series for a single site, ready for model fitting.
type SiteData struct {
	SiteID        int
	Years         []float64
	LogDensity    [][]float64 // [T][1] — observed log-density per year
	Covariates    [][]float64 // [T][K] — environmental covariates per year
	NumCovariates int
}

// LoadSiteTimeSeries reads a panel CSV with covariates and extracts the time
// series for a single site. Rows with zero density are skipped.
func LoadSiteTimeSeries(panelFile string, siteID int) *SiteData {
	f, err := os.Open(panelFile)
	if err != nil {
		panic("opening panel: " + err.Error())
	}
	defer f.Close()

	r := csv.NewReader(f)
	headers, err := r.Read()
	if err != nil {
		panic("reading header: " + err.Error())
	}

	idx := make(map[string]int)
	for i, h := range headers {
		idx[h] = i
	}

	var years []float64
	var logDensities [][]float64
	var covariates [][]float64

	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		id, _ := strconv.Atoi(record[idx["SITE_ID"]])
		if id != siteID {
			continue
		}

		year, _ := strconv.ParseFloat(record[idx["YEAR"]], 64)
		density, _ := strconv.ParseFloat(record[idx["DENSITY"]], 64)
		if density <= 0 {
			continue
		}

		flow := parseFloatDefault(record[idx["MEAN_FLOW"]])
		temp := parseFloatDefault(record[idx["MEAN_TEMP"]])
		do := parseFloatDefault(record[idx["MEAN_DO"]])

		years = append(years, year)
		logDensities = append(logDensities, []float64{math.Log(density)})
		covariates = append(covariates, []float64{flow, temp, do})
	}

	if len(years) == 0 {
		panic("no data found for site " + strconv.Itoa(siteID))
	}

	return &SiteData{
		SiteID:        siteID,
		Years:         years,
		LogDensity:    logDensities,
		Covariates:    covariates,
		NumCovariates: 3,
	}
}

// LoadAllSiteTimeSeries reads the panel CSV once and returns time series
// for every site. Sites with zero valid rows (density <= 0) are omitted.
func LoadAllSiteTimeSeries(panelFile string) map[int]*SiteData {
	f, err := os.Open(panelFile)
	if err != nil {
		panic("opening panel: " + err.Error())
	}
	defer f.Close()

	r := csv.NewReader(f)
	headers, err := r.Read()
	if err != nil {
		panic("reading header: " + err.Error())
	}

	idx := make(map[string]int)
	for i, h := range headers {
		idx[h] = i
	}

	type siteAccum struct {
		years      []float64
		logDensity [][]float64
		covariates [][]float64
	}
	sites := make(map[int]*siteAccum)

	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		id, _ := strconv.Atoi(record[idx["SITE_ID"]])
		year, _ := strconv.ParseFloat(record[idx["YEAR"]], 64)
		density, _ := strconv.ParseFloat(record[idx["DENSITY"]], 64)
		if density <= 0 {
			continue
		}

		flow := parseFloatDefault(record[idx["MEAN_FLOW"]])
		temp := parseFloatDefault(record[idx["MEAN_TEMP"]])
		do := parseFloatDefault(record[idx["MEAN_DO"]])

		acc, ok := sites[id]
		if !ok {
			acc = &siteAccum{}
			sites[id] = acc
		}
		acc.years = append(acc.years, year)
		acc.logDensity = append(acc.logDensity, []float64{math.Log(density)})
		acc.covariates = append(acc.covariates, []float64{flow, temp, do})
	}

	result := make(map[int]*SiteData, len(sites))
	for id, acc := range sites {
		result[id] = &SiteData{
			SiteID:        id,
			Years:         acc.years,
			LogDensity:    acc.logDensity,
			Covariates:    acc.covariates,
			NumCovariates: 3,
		}
	}
	return result
}

func parseFloatDefault(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0.0
	}
	return v
}

// SiteYearRecord holds the prepared data for one site in one year.
type SiteYearRecord struct {
	SiteID      int
	SiteName    string
	Area        string
	Easting     int
	Northing    int
	Year        int
	Count       float64 // ALL_RUNS total (summed if multiple surveys)
	SurveyArea  float64 // fished area in m²
	Density     float64 // count / survey area (fish per m²)
	NumSurveys  int     // surveys contributing to this year
	Method      string  // dominant survey method
	ZeroCatch   bool    // true if site was surveyed this year but species not caught
}

// PanelConfig controls how the panel is built.
type PanelConfig struct {
	// TargetSpecies is the species name to extract (e.g. "Brown / sea trout").
	TargetSpecies string

	// MinSurveyYears filters to sites with at least this many distinct survey
	// years for the target species (including zero-catch years). Set to 0 to
	// keep all sites.
	MinSurveyYears int

	// ElectrofishingOnly restricts to electric fishing methods when true.
	ElectrofishingOnly bool
}

func isElectrofishing(method string) bool {
	return strings.Contains(method, "ELECTRIC FISHING")
}

// BuildPanel scans the counts CSV and returns a clean site x year panel for the
// configured species. It handles:
//   - multiple surveys per site-year (sums counts, sums areas)
//   - zero-catch detection (site surveyed but species absent)
//   - optional method filtering
//   - density calculation (count / survey area)
func BuildPanel(countsCSVPath string, cfg PanelConfig) []SiteYearRecord {
	f, err := os.Open(countsCSVPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	if !scanner.Scan() {
		panic("empty file")
	}
	header := strings.TrimPrefix(scanner.Text(), "\xef\xbb\xbf")
	r := csv.NewReader(strings.NewReader(header))
	cols, err := r.Read()
	if err != nil {
		panic(err)
	}
	colIdx := make(map[string]int)
	for i, c := range cols {
		colIdx[c] = i
	}

	iSiteID := colIdx["SITE_ID"]
	iSiteName := colIdx["SITE_NAME"]
	iArea := colIdx["AREA"]
	iEasting := colIdx["SURVEY_RANKED_EASTING"]
	iNorthing := colIdx["SURVEY_RANKED_NORTHING"]
	iYear := colIdx["EVENT_DATE_YEAR"]
	iSpecies := colIdx["SPECIES_NAME"]
	iAllRuns := colIdx["ALL_RUNS"]
	iFishedArea := colIdx["FISHED_AREA"]
	iMethod := colIdx["SURVEY_METHOD"]
	iSurveyID := colIdx["SURVEY_ID"]
	iZeroCatch := colIdx["ZERO_CATCH"]
	iStatus := colIdx["SURVEY_STATUS"]

	type siteKey struct {
		siteID int
		year   int
	}

	type siteInfo struct {
		name     string
		area     string
		easting  int
		northing int
	}

	type accumulator struct {
		count      float64
		fishedArea float64
		surveys    map[int]bool
		methods    map[string]int
		zeroCatch  bool
	}

	data := make(map[siteKey]*accumulator)
	siteInfoMap := make(map[int]*siteInfo)

	// Track all surveyed site-years (including zero-catch) so we can mark
	// absence of the target species.
	surveyedSiteYears := make(map[siteKey]bool)

	for scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		record, err := r.Read()
		if err != nil || len(record) <= iStatus {
			continue
		}

		// Skip non-completed surveys
		if record[iStatus] != "Completed" {
			continue
		}

		siteID, _ := strconv.Atoi(record[iSiteID])
		year, _ := strconv.Atoi(record[iYear])
		method := record[iMethod]
		species := record[iSpecies]
		zeroCatch := record[iZeroCatch] == "Yes"
		surveyID, _ := strconv.Atoi(record[iSurveyID])

		if cfg.ElectrofishingOnly && !isElectrofishing(method) && !zeroCatch {
			continue
		}

		key := siteKey{siteID, year}

		// Store site metadata
		if _, ok := siteInfoMap[siteID]; !ok {
			easting, _ := strconv.Atoi(record[iEasting])
			northing, _ := strconv.Atoi(record[iNorthing])
			siteInfoMap[siteID] = &siteInfo{
				name:     record[iSiteName],
				area:     record[iArea],
				easting:  easting,
				northing: northing,
			}
		}

		// Mark this site-year as surveyed
		surveyedSiteYears[key] = true

		if zeroCatch {
			// Zero-catch row — no species recorded at all this survey
			acc, ok := data[key]
			if !ok {
				acc = &accumulator{
					surveys:   make(map[int]bool),
					methods:   make(map[string]int),
					zeroCatch: true,
				}
				data[key] = acc
			}
			acc.surveys[surveyID] = true
			acc.methods[method]++
			fa, _ := strconv.ParseFloat(record[iFishedArea], 64)
			if fa > 0 {
				acc.fishedArea += fa
			}
			continue
		}

		// Species-specific row
		if species == cfg.TargetSpecies {
			acc, ok := data[key]
			if !ok {
				acc = &accumulator{
					surveys: make(map[int]bool),
					methods: make(map[string]int),
				}
				data[key] = acc
			}
			acc.zeroCatch = false // species was found
			count, _ := strconv.ParseFloat(record[iAllRuns], 64)
			acc.count += count
			fa, _ := strconv.ParseFloat(record[iFishedArea], 64)
			if fa > 0 {
				acc.fishedArea += fa
			}
			acc.surveys[surveyID] = true
			acc.methods[method]++
		}
	}

	// For surveyed site-years where the target species was NOT recorded and
	// it's not already a zero-catch row, add a zero-catch entry.
	for key := range surveyedSiteYears {
		if _, hasData := data[key]; !hasData {
			data[key] = &accumulator{
				surveys:   make(map[int]bool),
				methods:   make(map[string]int),
				zeroCatch: true,
			}
		}
	}

	// Count years per site for filtering
	siteYearCounts := make(map[int]int)
	for key := range data {
		siteYearCounts[key.siteID]++
	}

	// Build output records
	records := make([]SiteYearRecord, 0, len(data))
	for key, acc := range data {
		if cfg.MinSurveyYears > 0 && siteYearCounts[key.siteID] < cfg.MinSurveyYears {
			continue
		}

		info := siteInfoMap[key.siteID]
		if info == nil {
			continue
		}

		// Find dominant method
		bestMethod := ""
		bestCount := 0
		for m, c := range acc.methods {
			if c > bestCount {
				bestMethod = m
				bestCount = c
			}
		}

		density := 0.0
		if acc.fishedArea > 0 {
			density = acc.count / acc.fishedArea
		}

		records = append(records, SiteYearRecord{
			SiteID:     key.siteID,
			SiteName:   info.name,
			Area:       info.area,
			Easting:    info.easting,
			Northing:   info.northing,
			Year:       key.year,
			Count:      acc.count,
			SurveyArea: acc.fishedArea,
			Density:    density,
			NumSurveys: len(acc.surveys),
			Method:     bestMethod,
			ZeroCatch:  acc.zeroCatch,
		})
	}

	// Sort by site then year
	sort.Slice(records, func(i, j int) bool {
		if records[i].SiteID != records[j].SiteID {
			return records[i].SiteID < records[j].SiteID
		}
		return records[i].Year < records[j].Year
	})

	return records
}

// PanelToDataFrame converts panel records to a gota dataframe.
func PanelToDataFrame(records []SiteYearRecord) dataframe.DataFrame {
	n := len(records)
	siteIDs := make([]int, n)
	siteNames := make([]string, n)
	areas := make([]string, n)
	eastings := make([]int, n)
	northings := make([]int, n)
	years := make([]int, n)
	counts := make([]float64, n)
	surveyAreas := make([]float64, n)
	densities := make([]float64, n)
	numSurveys := make([]int, n)
	methods := make([]string, n)
	zeroCatches := make([]bool, n)

	for i, r := range records {
		siteIDs[i] = r.SiteID
		siteNames[i] = r.SiteName
		areas[i] = r.Area
		eastings[i] = r.Easting
		northings[i] = r.Northing
		years[i] = r.Year
		counts[i] = r.Count
		surveyAreas[i] = r.SurveyArea
		densities[i] = r.Density
		numSurveys[i] = r.NumSurveys
		methods[i] = r.Method
		zeroCatches[i] = r.ZeroCatch
	}

	return dataframe.New(
		series.New(siteIDs, series.Int, "SITE_ID"),
		series.New(siteNames, series.String, "SITE_NAME"),
		series.New(areas, series.String, "AREA"),
		series.New(eastings, series.Int, "EASTING"),
		series.New(northings, series.Int, "NORTHING"),
		series.New(years, series.Int, "YEAR"),
		series.New(counts, series.Float, "COUNT"),
		series.New(surveyAreas, series.Float, "SURVEY_AREA"),
		series.New(densities, series.Float, "DENSITY"),
		series.New(numSurveys, series.Int, "NUM_SURVEYS"),
		series.New(methods, series.String, "METHOD"),
		series.New(zeroCatches, series.Bool, "ZERO_CATCH"),
	)
}

// PanelSiteSummary returns a summary dataframe with one row per site.
func PanelSiteSummary(records []SiteYearRecord) dataframe.DataFrame {
	type siteAccum struct {
		name      string
		area      string
		easting   int
		northing  int
		years     []int
		totalFish float64
		zeroes    int
	}

	sites := make(map[int]*siteAccum)
	for _, r := range records {
		s, ok := sites[r.SiteID]
		if !ok {
			s = &siteAccum{
				name:     r.SiteName,
				area:     r.Area,
				easting:  r.Easting,
				northing: r.Northing,
			}
			sites[r.SiteID] = s
		}
		s.years = append(s.years, r.Year)
		s.totalFish += r.Count
		if r.ZeroCatch {
			s.zeroes++
		}
	}

	type siteSummary struct {
		id        int
		name      string
		area      string
		easting   int
		northing  int
		numYears  int
		firstYear int
		lastYear  int
		meanCount float64
		zeroRate  float64
	}

	summaries := make([]siteSummary, 0, len(sites))
	for id, s := range sites {
		sort.Ints(s.years)
		n := len(s.years)
		summaries = append(summaries, siteSummary{
			id:        id,
			name:      s.name,
			area:      s.area,
			easting:   s.easting,
			northing:  s.northing,
			numYears:  n,
			firstYear: s.years[0],
			lastYear:  s.years[n-1],
			meanCount: s.totalFish / float64(n),
			zeroRate:  float64(s.zeroes) / float64(n),
		})
	}

	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].numYears > summaries[j].numYears
	})

	n := len(summaries)
	ids := make([]int, n)
	names := make([]string, n)
	areasOut := make([]string, n)
	eastingsOut := make([]int, n)
	northingsOut := make([]int, n)
	numYears := make([]int, n)
	firstYears := make([]int, n)
	lastYears := make([]int, n)
	meanCounts := make([]float64, n)
	zeroRates := make([]float64, n)

	for i, s := range summaries {
		ids[i] = s.id
		names[i] = s.name
		areasOut[i] = s.area
		eastingsOut[i] = s.easting
		northingsOut[i] = s.northing
		numYears[i] = s.numYears
		firstYears[i] = s.firstYear
		lastYears[i] = s.lastYear
		meanCounts[i] = s.meanCount
		zeroRates[i] = s.zeroRate
	}

	return dataframe.New(
		series.New(ids, series.Int, "SITE_ID"),
		series.New(names, series.String, "SITE_NAME"),
		series.New(areasOut, series.String, "AREA"),
		series.New(eastingsOut, series.Int, "EASTING"),
		series.New(northingsOut, series.Int, "NORTHING"),
		series.New(numYears, series.Int, "NUM_YEARS"),
		series.New(firstYears, series.Int, "FIRST_YEAR"),
		series.New(lastYears, series.Int, "LAST_YEAR"),
		series.New(meanCounts, series.Float, "MEAN_COUNT"),
		series.New(zeroRates, series.Float, "ZERO_RATE"),
	)
}
