package data

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

const waterQualityBaseURL = "https://environment.data.gov.uk/water-quality"

// WaterQuality determinand codes.
const (
	DetTemperature = "0076" // Temperature of Water (°C)
	DetDissolvedO2 = "9924" // Oxygen, Dissolved as O2 (mg/l)
	DetAmmonia     = "0111" // Ammoniacal Nitrogen as N (mg/l)
	DetBOD         = "0085" // BOD: 5 Day ATU (mg/l)
)

// WQSamplingPoint represents a water quality sampling point.
type WQSamplingPoint struct {
	Notation string  // e.g. "SW-70512058"
	Name     string
	Lat      float64
	Long     float64
	DistKm   float64 // distance from search point
	Type     string  // e.g. "FRESHWATER - RIVERS"
	Status   string  // "OPEN" or "CLOSED"
}

// WQObservation represents a single water quality measurement.
type WQObservation struct {
	Date          string
	Determinand   string
	DeterminandID string
	Value         float64
	Unit          string
}

// AnnualWQStats holds annual water quality summary for one year.
type AnnualWQStats struct {
	Year           int
	MeanTemp       float64 // mean water temperature (°C)
	MaxTemp        float64
	MeanDO         float64 // mean dissolved oxygen (mg/l)
	MinDO          float64
	MeanAmmonia    float64 // mean ammoniacal nitrogen (mg/l)
	MaxAmmonia     float64
	MeanBOD        float64 // mean BOD (mg/l)
	MaxBOD         float64
	NumTempSamples int
	NumDOSamples   int
	NumNH3Samples  int
	NumBODSamples  int
}

func waterQualityGet(endpoint string, params map[string]string, accept string) ([]byte, error) {
	u, err := url.Parse(waterQualityBaseURL + endpoint)
	if err != nil {
		return nil, err
	}
	q := u.Query()
	for k, v := range params {
		q.Set(k, v)
	}
	u.RawQuery = q.Encode()

	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", accept)
	if accept == "text/csv" {
		req.Header.Set("CSV-Header", "present")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body[:min(200, len(body))]))
	}

	return io.ReadAll(resp.Body)
}

// wqGeoJSON is the response structure for GeoJSON sampling point queries.
type wqGeoJSON struct {
	TotalItems int `json:"totalItems"`
	Member     []struct {
		Geometry struct {
			Coordinates []float64 `json:"coordinates"` // [long, lat]
		} `json:"geometry"`
		Properties struct {
			Name     string `json:"name"`
			Status   string `json:"status"`
			Type     string `json:"type"`
			Notation string `json:"notation"`
		} `json:"properties"`
	} `json:"member"`
}

// haversineDistKm computes the great-circle distance between two lat/long
// points in kilometres.
func haversineDistKm(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371.0 // Earth radius in km
	dLat := (lat2 - lat1) * math.Pi / 180.0
	dLon := (lon2 - lon1) * math.Pi / 180.0
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*math.Pi/180.0)*math.Cos(lat2*math.Pi/180.0)*
			math.Sin(dLon/2)*math.Sin(dLon/2)
	return R * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

// FindNearestWQPoint finds the nearest river water quality sampling point
// to the given easting/northing within maxDistKm. Returns nil if none found.
func FindNearestWQPoint(easting, northing int, maxDistKm float64) *WQSamplingPoint {
	lat, long := EastingNorthingToLatLong(easting, northing)

	body, err := waterQualityGet("/sampling-point", map[string]string{
		"latitude":          fmt.Sprintf("%.6f", lat),
		"longitude":         fmt.Sprintf("%.6f", long),
		"radius":            fmt.Sprintf("%.1f", maxDistKm),
		"samplingPointType": "F6", // freshwater rivers
		"limit":             "50",
	}, "application/geo+json")
	if err != nil {
		return nil
	}

	var resp wqGeoJSON
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil
	}

	var best *WQSamplingPoint
	bestDist := math.MaxFloat64

	for _, m := range resp.Member {
		if len(m.Geometry.Coordinates) != 2 {
			continue
		}
		ptLong := m.Geometry.Coordinates[0]
		ptLat := m.Geometry.Coordinates[1]

		dist := haversineDistKm(lat, long, ptLat, ptLong)

		if dist < bestDist {
			bestDist = dist
			best = &WQSamplingPoint{
				Notation: m.Properties.Notation,
				Name:     m.Properties.Name,
				Lat:      ptLat,
				Long:     ptLong,
				Type:     m.Properties.Type,
				Status:   m.Properties.Status,
			}
		}
	}

	if best != nil {
		best.DistKm = bestDist
	}

	return best
}

// GetWQObservations fetches water quality observations for a sampling point
// and determinand between two dates. Uses CSV format for efficiency.
func GetWQObservations(notation, determinandCode, startDate, endDate string) []WQObservation {
	var allObs []WQObservation

	// Paginate through results (API limit cap is 250)
	skip := 0
	limit := 250
	for {
		body, err := waterQualityGet(
			fmt.Sprintf("/sampling-point/%s/observation", notation),
			map[string]string{
				"determinand": determinandCode,
				"dateFrom":    startDate,
				"dateTo":      endDate,
				"limit":       strconv.Itoa(limit),
				"skip":        strconv.Itoa(skip),
			},
			"text/csv",
		)
		if err != nil {
			break
		}

		scanner := bufio.NewScanner(strings.NewReader(string(body)))

		// Read header
		if !scanner.Scan() {
			break
		}
		headerLine := scanner.Text()
		headers, err := csv.NewReader(strings.NewReader(headerLine)).Read()
		if err != nil {
			break
		}

		colIdx := make(map[string]int)
		for i, h := range headers {
			colIdx[h] = i
		}

		iTime, hasTime := colIdx["phenomenonTime"]
		iResult, hasResult := colIdx["result"]
		iDet, hasDet := colIdx["determinand.prefLabel"]
		iDetN, hasDetN := colIdx["determinand.notation"]
		iUnit, hasUnit := colIdx["unit"]

		if !hasTime || !hasResult {
			break
		}

		count := 0
		for scanner.Scan() {
			line := scanner.Text()
			record, err := csv.NewReader(strings.NewReader(line)).Read()
			if err != nil {
				continue
			}

			val, err := strconv.ParseFloat(record[iResult], 64)
			if err != nil {
				continue
			}

			// Extract date from ISO datetime
			dateStr := record[iTime]
			if len(dateStr) >= 10 {
				dateStr = dateStr[:10]
			}

			obs := WQObservation{
				Date:  dateStr,
				Value: val,
			}
			if hasDet {
				obs.Determinand = record[iDet]
			}
			if hasDetN {
				obs.DeterminandID = record[iDetN]
			}
			if hasUnit {
				obs.Unit = record[iUnit]
			}

			allObs = append(allObs, obs)
			count++
		}

		if count < limit {
			break // no more pages
		}
		skip += limit
		time.Sleep(200 * time.Millisecond)
	}

	sort.Slice(allObs, func(i, j int) bool {
		return allObs[i].Date < allObs[j].Date
	})

	return allObs
}

// AggregateAnnualWQ computes annual water quality statistics from observations
// across multiple determinands. Pass observations from all determinands together.
func AggregateAnnualWQ(tempObs, doObs, nh3Obs, bodObs []WQObservation) []AnnualWQStats {
	years := make(map[int]bool)

	byYear := func(obs []WQObservation) map[int][]float64 {
		m := make(map[int][]float64)
		for _, o := range obs {
			if len(o.Date) < 4 {
				continue
			}
			year, _ := strconv.Atoi(o.Date[:4])
			if year > 0 {
				m[year] = append(m[year], o.Value)
				years[year] = true
			}
		}
		return m
	}

	tempByYear := byYear(tempObs)
	doByYear := byYear(doObs)
	nh3ByYear := byYear(nh3Obs)
	bodByYear := byYear(bodObs)

	sortedYears := make([]int, 0, len(years))
	for y := range years {
		sortedYears = append(sortedYears, y)
	}
	sort.Ints(sortedYears)

	meanOf := func(vals []float64) float64 {
		if len(vals) == 0 {
			return math.NaN()
		}
		sum := 0.0
		for _, v := range vals {
			sum += v
		}
		return sum / float64(len(vals))
	}

	maxOf := func(vals []float64) float64 {
		if len(vals) == 0 {
			return math.NaN()
		}
		m := vals[0]
		for _, v := range vals[1:] {
			if v > m {
				m = v
			}
		}
		return m
	}

	minOf := func(vals []float64) float64 {
		if len(vals) == 0 {
			return math.NaN()
		}
		m := vals[0]
		for _, v := range vals[1:] {
			if v < m {
				m = v
			}
		}
		return m
	}

	stats := make([]AnnualWQStats, len(sortedYears))
	for i, year := range sortedYears {
		stats[i] = AnnualWQStats{
			Year:           year,
			MeanTemp:       meanOf(tempByYear[year]),
			MaxTemp:        maxOf(tempByYear[year]),
			MeanDO:         meanOf(doByYear[year]),
			MinDO:          minOf(doByYear[year]),
			MeanAmmonia:    meanOf(nh3ByYear[year]),
			MaxAmmonia:     maxOf(nh3ByYear[year]),
			MeanBOD:        meanOf(bodByYear[year]),
			MaxBOD:         maxOf(bodByYear[year]),
			NumTempSamples: len(tempByYear[year]),
			NumDOSamples:   len(doByYear[year]),
			NumNH3Samples:  len(nh3ByYear[year]),
			NumBODSamples:  len(bodByYear[year]),
		}
	}

	return stats
}

// SeasonalWQStats holds seasonal water quality statistics for one year.
type SeasonalWQStats struct {
	Year           int
	SummerMeanTemp float64 // mean water temp Jun-Sep (°C)
	SummerNumTemp  int
	WinterMeanDO   float64 // mean DO Dec(Y-1)-Feb(Y) (mg/l)
	WinterNumDO    int
}

// AggregateSeasonalWQ computes seasonal water quality stats:
// - Summer mean temperature (Jun-Sep)
// - Winter mean dissolved oxygen (Dec of previous year through Feb)
func AggregateSeasonalWQ(tempObs, doObs []WQObservation) []SeasonalWQStats {
	// Summer temp: group by calendar year, filter Jun-Sep
	summerTemp := make(map[int][]float64)
	for _, o := range tempObs {
		if len(o.Date) < 7 {
			continue
		}
		year, _ := strconv.Atoi(o.Date[:4])
		month, _ := strconv.Atoi(o.Date[5:7])
		if year > 0 && month >= 6 && month <= 9 {
			summerTemp[year] = append(summerTemp[year], o.Value)
		}
	}

	// Winter DO: Dec of year Y assigned to winter Y+1; Jan-Feb of year Y assigned to winter Y
	winterDO := make(map[int][]float64)
	for _, o := range doObs {
		if len(o.Date) < 7 {
			continue
		}
		year, _ := strconv.Atoi(o.Date[:4])
		month, _ := strconv.Atoi(o.Date[5:7])
		if year <= 0 {
			continue
		}
		switch {
		case month == 12:
			winterDO[year+1] = append(winterDO[year+1], o.Value)
		case month <= 2:
			winterDO[year] = append(winterDO[year], o.Value)
		}
	}

	// Collect all years that have at least one stat
	years := make(map[int]bool)
	for y := range summerTemp {
		years[y] = true
	}
	for y := range winterDO {
		years[y] = true
	}

	sortedYears := make([]int, 0, len(years))
	for y := range years {
		sortedYears = append(sortedYears, y)
	}
	sort.Ints(sortedYears)

	meanOf := func(vals []float64) float64 {
		if len(vals) == 0 {
			return math.NaN()
		}
		sum := 0.0
		for _, v := range vals {
			sum += v
		}
		return sum / float64(len(vals))
	}

	stats := make([]SeasonalWQStats, len(sortedYears))
	for i, year := range sortedYears {
		stats[i] = SeasonalWQStats{
			Year:           year,
			SummerMeanTemp: meanOf(summerTemp[year]),
			SummerNumTemp:  len(summerTemp[year]),
			WinterMeanDO:   meanOf(winterDO[year]),
			WinterNumDO:    len(winterDO[year]),
		}
	}

	return stats
}

// AnnualWQToDataFrame converts annual water quality stats to a gota dataframe.
func AnnualWQToDataFrame(stats []AnnualWQStats) dataframe.DataFrame {
	n := len(stats)
	years := make([]int, n)
	meanTemps := make([]float64, n)
	maxTemps := make([]float64, n)
	meanDOs := make([]float64, n)
	minDOs := make([]float64, n)
	meanNH3s := make([]float64, n)
	maxNH3s := make([]float64, n)
	meanBODs := make([]float64, n)
	maxBODs := make([]float64, n)
	nTemp := make([]int, n)
	nDO := make([]int, n)
	nNH3 := make([]int, n)
	nBOD := make([]int, n)

	for i, s := range stats {
		years[i] = s.Year
		meanTemps[i] = s.MeanTemp
		maxTemps[i] = s.MaxTemp
		meanDOs[i] = s.MeanDO
		minDOs[i] = s.MinDO
		meanNH3s[i] = s.MeanAmmonia
		maxNH3s[i] = s.MaxAmmonia
		meanBODs[i] = s.MeanBOD
		maxBODs[i] = s.MaxBOD
		nTemp[i] = s.NumTempSamples
		nDO[i] = s.NumDOSamples
		nNH3[i] = s.NumNH3Samples
		nBOD[i] = s.NumBODSamples
	}

	return dataframe.New(
		series.New(years, series.Int, "YEAR"),
		series.New(meanTemps, series.Float, "MEAN_TEMP"),
		series.New(maxTemps, series.Float, "MAX_TEMP"),
		series.New(meanDOs, series.Float, "MEAN_DO"),
		series.New(minDOs, series.Float, "MIN_DO"),
		series.New(meanNH3s, series.Float, "MEAN_AMMONIA"),
		series.New(maxNH3s, series.Float, "MAX_AMMONIA"),
		series.New(meanBODs, series.Float, "MEAN_BOD"),
		series.New(maxBODs, series.Float, "MAX_BOD"),
		series.New(nTemp, series.Int, "N_TEMP"),
		series.New(nDO, series.Int, "N_DO"),
		series.New(nNH3, series.Int, "N_NH3"),
		series.New(nBOD, series.Int, "N_BOD"),
	)
}

// MatchPanelSitesToWQPoints finds the nearest river water quality sampling point
// for each site in the panel summary. Returns a dataframe mapping site IDs to
// sampling point details.
func MatchPanelSitesToWQPoints(
	siteSummary dataframe.DataFrame,
	maxDistKm float64,
) dataframe.DataFrame {
	n := siteSummary.Nrow()
	siteIDs := make([]int, n)
	siteNames := make([]string, n)
	wqNotations := make([]string, n)
	wqNames := make([]string, n)
	wqLats := make([]float64, n)
	wqLongs := make([]float64, n)
	distKms := make([]float64, n)
	matched := make([]bool, n)

	eastings := siteSummary.Col("EASTING")
	northings := siteSummary.Col("NORTHING")
	ids := siteSummary.Col("SITE_ID")
	names := siteSummary.Col("SITE_NAME")

	for i := 0; i < n; i++ {
		siteIDs[i], _ = ids.Elem(i).Int()
		siteNames[i] = names.Elem(i).String()

		e, _ := eastings.Elem(i).Int()
		no, _ := northings.Elem(i).Int()

		pt := FindNearestWQPoint(e, no, maxDistKm)
		if pt != nil {
			wqNotations[i] = pt.Notation
			wqNames[i] = pt.Name
			wqLats[i] = pt.Lat
			wqLongs[i] = pt.Long
			distKms[i] = pt.DistKm
			matched[i] = true
		}

		// Rate limit
		if i > 0 && i%10 == 0 {
			time.Sleep(500 * time.Millisecond)
		}
	}

	return dataframe.New(
		series.New(siteIDs, series.Int, "SITE_ID"),
		series.New(siteNames, series.String, "SITE_NAME"),
		series.New(wqNotations, series.String, "WQ_NOTATION"),
		series.New(wqNames, series.String, "WQ_POINT_NAME"),
		series.New(wqLats, series.Float, "WQ_LAT"),
		series.New(wqLongs, series.Float, "WQ_LONG"),
		series.New(distKms, series.Float, "DIST_KM"),
		series.New(matched, series.Bool, "MATCHED"),
	)
}
