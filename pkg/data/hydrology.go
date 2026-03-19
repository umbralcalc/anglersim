package data

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

const hydrologyBaseURL = "https://environment.data.gov.uk/hydrology"

// HydrologyStation represents a station from the EA Hydrology API.
type HydrologyStation struct {
	ID         string
	Label      string
	Easting    int
	Northing   int
	Lat        float64
	Long       float64
	DateOpened string
	Measures   []HydrologyMeasure
}

// HydrologyMeasure represents a measure available at a station.
type HydrologyMeasure struct {
	ID        string
	Parameter string
	Period    int
}

// HydrologyReading represents a single reading.
type HydrologyReading struct {
	Date  string
	Value float64
}

// EastingNorthingToLatLong converts OS National Grid to WGS84 lat/long.
// Uses the simplified Helmert transformation — accurate to ~5m which is
// more than sufficient for finding the nearest station within km.
func EastingNorthingToLatLong(easting, northing int) (lat, long float64) {
	// OSGB36 ellipsoid constants
	a := 6377563.396
	b := 6356256.909
	e2 := (a*a - b*b) / (a * a)
	// National Grid origin
	n0 := -100000.0
	e0 := 400000.0
	phi0 := 49.0 * math.Pi / 180.0
	lam0 := -2.0 * math.Pi / 180.0
	f0 := 0.9996012717
	n := (a - b) / (a + b)
	n2 := n * n
	n3 := n * n * n

	E := float64(easting)
	N := float64(northing)

	phi := phi0
	M := 0.0
	for {
		phi = (N-n0-M)/(a*f0) + phi
		Ma := (1.0 + n + 5.0/4.0*n2 + 5.0/4.0*n3) * (phi - phi0)
		Mb := (3.0*n + 3.0*n2 + 21.0/8.0*n3) * math.Sin(phi-phi0) * math.Cos(phi+phi0)
		Mc := (15.0/8.0*n2 + 15.0/8.0*n3) * math.Sin(2.0*(phi-phi0)) * math.Cos(2.0*(phi+phi0))
		Md := 35.0 / 24.0 * n3 * math.Sin(3.0*(phi-phi0)) * math.Cos(3.0*(phi+phi0))
		M = b * f0 * (Ma - Mb + Mc - Md)
		if math.Abs(N-n0-M) < 0.01 {
			break
		}
	}

	sinPhi := math.Sin(phi)
	sin2Phi := sinPhi * sinPhi
	cosPhi := math.Cos(phi)
	tanPhi := math.Tan(phi)
	tan2Phi := tanPhi * tanPhi
	tan4Phi := tan2Phi * tan2Phi

	nu := a * f0 / math.Sqrt(1.0-e2*sin2Phi)
	rho := a * f0 * (1.0 - e2) / math.Pow(1.0-e2*sin2Phi, 1.5)
	eta2 := nu/rho - 1.0

	dE := E - e0

	VII := tanPhi / (2.0 * rho * nu)
	VIII := tanPhi / (24.0 * rho * nu * nu * nu) * (5.0 + 3.0*tan2Phi + eta2 - 9.0*tan2Phi*eta2)
	IX := tanPhi / (720.0 * rho * math.Pow(nu, 5)) * (61.0 + 90.0*tan2Phi + 45.0*tan4Phi)
	X := 1.0 / (cosPhi * nu)
	XI := 1.0 / (cosPhi * 6.0 * nu * nu * nu) * (nu/rho + 2.0*tan2Phi)
	XII := 1.0 / (cosPhi * 120.0 * math.Pow(nu, 5)) * (5.0 + 28.0*tan2Phi + 24.0*tan4Phi)

	lat = (phi - VII*dE*dE + VIII*math.Pow(dE, 4) - IX*math.Pow(dE, 6)) * 180.0 / math.Pi
	long = (lam0 + X*dE - XI*math.Pow(dE, 3) + XII*math.Pow(dE, 5)) * 180.0 / math.Pi

	return lat, long
}

// apiResponse is the generic JSON structure from the Hydrology API.
type apiResponse struct {
	Items []json.RawMessage `json:"items"`
}

type stationJSON struct {
	ID         string `json:"@id"`
	Label      string `json:"label"`
	Easting    int    `json:"easting"`
	Northing   int    `json:"northing"`
	Lat        float64 `json:"lat"`
	Long       float64 `json:"long"`
	DateOpened string `json:"dateOpened"`
	Measures   []struct {
		ID        string `json:"@id"`
		Parameter string `json:"parameter"`
		Period    int    `json:"period"`
	} `json:"measures"`
}

type readingJSON struct {
	Date     string  `json:"date"`
	DateTime string  `json:"dateTime"`
	Value    float64 `json:"value"`
}

var httpClient = &http.Client{Timeout: 30 * time.Second}

func hydrologyGet(endpoint string, params map[string]string) ([]byte, error) {
	u, err := url.Parse(hydrologyBaseURL + endpoint)
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
	req.Header.Set("Accept", "application/json")

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

// FindNearestFlowStation finds the nearest hydrology flow station to the given
// easting/northing within maxDistKm. Returns nil if none found.
func FindNearestFlowStation(easting, northing int, maxDistKm float64) *HydrologyStation {
	lat, long := EastingNorthingToLatLong(easting, northing)

	body, err := hydrologyGet("/id/stations", map[string]string{
		"lat":              fmt.Sprintf("%.6f", lat),
		"long":             fmt.Sprintf("%.6f", long),
		"dist":             fmt.Sprintf("%.1f", maxDistKm),
		"observedProperty": "http://environment.data.gov.uk/reference/def/op/waterFlow",
		"_limit":           "20",
	})
	if err != nil {
		return nil
	}

	var resp apiResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil
	}

	var bestStation *HydrologyStation
	bestDist := math.MaxFloat64

	for _, raw := range resp.Items {
		var s stationJSON
		if err := json.Unmarshal(raw, &s); err != nil {
			continue
		}
		de := float64(s.Easting - easting)
		dn := float64(s.Northing - northing)
		dist := math.Sqrt(de*de + dn*dn)

		if dist < bestDist {
			bestDist = dist
			measures := make([]HydrologyMeasure, len(s.Measures))
			for i, m := range s.Measures {
				measures[i] = HydrologyMeasure{
					ID:        m.ID,
					Parameter: m.Parameter,
					Period:    m.Period,
				}
			}
			bestStation = &HydrologyStation{
				ID:         s.ID,
				Label:      s.Label,
				Easting:    s.Easting,
				Northing:   s.Northing,
				Lat:        s.Lat,
				Long:       s.Long,
				DateOpened: s.DateOpened,
				Measures:   measures,
			}
		}
	}

	return bestStation
}

// DailyMeanFlowMeasureID returns the measure ID for daily mean flow at a station,
// or empty string if not available.
func (s *HydrologyStation) DailyMeanFlowMeasureID() string {
	for _, m := range s.Measures {
		if m.Parameter == "flow" && m.Period == 86400 {
			id := m.ID
			parts := strings.Split(id, "/")
			notation := parts[len(parts)-1]
			if strings.Contains(notation, "-flow-m-86400") {
				return notation
			}
		}
	}
	return ""
}

// GetDailyFlowReadings fetches daily mean flow readings for a measure between
// two dates (inclusive, format "2006-01-02").
func GetDailyFlowReadings(measureID, startDate, endDate string) []HydrologyReading {
	var allReadings []HydrologyReading

	// Fetch in yearly chunks to avoid hitting API limits
	start, _ := time.Parse("2006-01-02", startDate)
	end, _ := time.Parse("2006-01-02", endDate)

	for chunkStart := start; chunkStart.Before(end); {
		chunkEnd := chunkStart.AddDate(1, 0, 0)
		if chunkEnd.After(end) {
			chunkEnd = end
		}

		body, err := hydrologyGet(
			fmt.Sprintf("/id/measures/%s/readings", measureID),
			map[string]string{
				"mineq-date": chunkStart.Format("2006-01-02"),
				"maxeq-date": chunkEnd.Format("2006-01-02"),
				"_limit":     "500",
			},
		)
		if err != nil {
			chunkStart = chunkEnd
			continue
		}

		var resp apiResponse
		if err := json.Unmarshal(body, &resp); err != nil {
			chunkStart = chunkEnd
			continue
		}

		for _, raw := range resp.Items {
			var r readingJSON
			if err := json.Unmarshal(raw, &r); err != nil {
				continue
			}
			date := r.Date
			if date == "" {
				date = r.DateTime
			}
			allReadings = append(allReadings, HydrologyReading{
				Date:  date,
				Value: r.Value,
			})
		}

		chunkStart = chunkEnd
		time.Sleep(200 * time.Millisecond)
	}

	sort.Slice(allReadings, func(i, j int) bool {
		return allReadings[i].Date < allReadings[j].Date
	})

	return allReadings
}

// AnnualFlowStats holds summary flow statistics for one year.
type AnnualFlowStats struct {
	Year       int
	MeanFlow   float64 // mean daily flow (m³/s)
	MaxFlow    float64 // max daily flow (m³/s)
	MinFlow    float64 // min daily flow (m³/s)
	Q10Flow    float64 // 10th percentile (low flow indicator)
	Q90Flow    float64 // 90th percentile (high flow indicator)
	NumDays    int     // days with readings
}

// AggregateAnnualFlow computes annual summary statistics from daily readings.
func AggregateAnnualFlow(readings []HydrologyReading) []AnnualFlowStats {
	byYear := make(map[int][]float64)
	for _, r := range readings {
		if len(r.Date) < 4 {
			continue
		}
		year := 0
		fmt.Sscanf(r.Date[:4], "%d", &year)
		if year > 0 {
			byYear[year] = append(byYear[year], r.Value)
		}
	}

	years := make([]int, 0, len(byYear))
	for y := range byYear {
		years = append(years, y)
	}
	sort.Ints(years)

	stats := make([]AnnualFlowStats, len(years))
	for i, year := range years {
		vals := byYear[year]
		sort.Float64s(vals)
		n := len(vals)

		sum := 0.0
		for _, v := range vals {
			sum += v
		}

		stats[i] = AnnualFlowStats{
			Year:     year,
			MeanFlow: sum / float64(n),
			MinFlow:  vals[0],
			MaxFlow:  vals[n-1],
			Q10Flow:  vals[n/10],
			Q90Flow:  vals[9*n/10],
			NumDays:  n,
		}
	}

	return stats
}

// SeasonalFlowStats holds summer flow statistics for one year.
type SeasonalFlowStats struct {
	Year           int
	SummerMeanFlow float64 // mean daily flow Jun-Sep (m³/s)
	SummerNumDays  int     // days with readings in Jun-Sep
}

// AggregateSeasonalFlow computes summer (Jun-Sep) mean flow from daily readings.
func AggregateSeasonalFlow(readings []HydrologyReading) []SeasonalFlowStats {
	byYear := make(map[int][]float64)
	for _, r := range readings {
		if len(r.Date) < 10 {
			continue
		}
		year := 0
		month := 0
		fmt.Sscanf(r.Date[:4], "%d", &year)
		fmt.Sscanf(r.Date[5:7], "%d", &month)
		if year > 0 && month >= 6 && month <= 9 {
			byYear[year] = append(byYear[year], r.Value)
		}
	}

	years := make([]int, 0, len(byYear))
	for y := range byYear {
		years = append(years, y)
	}
	sort.Ints(years)

	stats := make([]SeasonalFlowStats, len(years))
	for i, year := range years {
		vals := byYear[year]
		sum := 0.0
		for _, v := range vals {
			sum += v
		}
		stats[i] = SeasonalFlowStats{
			Year:           year,
			SummerMeanFlow: sum / float64(len(vals)),
			SummerNumDays:  len(vals),
		}
	}

	return stats
}

// AnnualFlowToDataFrame converts annual flow stats to a gota dataframe.
func AnnualFlowToDataFrame(stats []AnnualFlowStats) dataframe.DataFrame {
	n := len(stats)
	years := make([]int, n)
	means := make([]float64, n)
	maxes := make([]float64, n)
	mins := make([]float64, n)
	q10s := make([]float64, n)
	q90s := make([]float64, n)
	days := make([]int, n)

	for i, s := range stats {
		years[i] = s.Year
		means[i] = s.MeanFlow
		maxes[i] = s.MaxFlow
		mins[i] = s.MinFlow
		q10s[i] = s.Q10Flow
		q90s[i] = s.Q90Flow
		days[i] = s.NumDays
	}

	return dataframe.New(
		series.New(years, series.Int, "YEAR"),
		series.New(means, series.Float, "MEAN_FLOW"),
		series.New(maxes, series.Float, "MAX_FLOW"),
		series.New(mins, series.Float, "MIN_FLOW"),
		series.New(q10s, series.Float, "Q10_FLOW"),
		series.New(q90s, series.Float, "Q90_FLOW"),
		series.New(days, series.Int, "NUM_DAYS"),
	)
}

// MatchPanelSitesToFlowStations finds the nearest flow station for each site
// in the panel summary. Returns a dataframe mapping site IDs to station details.
func MatchPanelSitesToFlowStations(
	siteSummary dataframe.DataFrame,
	maxDistKm float64,
) dataframe.DataFrame {
	n := siteSummary.Nrow()
	siteIDs := make([]int, n)
	siteNames := make([]string, n)
	stationLabels := make([]string, n)
	stationEastings := make([]int, n)
	stationNorthings := make([]int, n)
	distMetres := make([]float64, n)
	measureIDs := make([]string, n)
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

		station := FindNearestFlowStation(e, no, maxDistKm)
		if station != nil {
			de := float64(station.Easting - e)
			dn := float64(station.Northing - no)
			dist := math.Sqrt(de*de + dn*dn)

			stationLabels[i] = station.Label
			stationEastings[i] = station.Easting
			stationNorthings[i] = station.Northing
			distMetres[i] = dist
			measureIDs[i] = station.DailyMeanFlowMeasureID()
			matched[i] = true
		}

		// Rate limit: be kind to the API
		if i > 0 && i%10 == 0 {
			time.Sleep(500 * time.Millisecond)
		}
	}

	return dataframe.New(
		series.New(siteIDs, series.Int, "SITE_ID"),
		series.New(siteNames, series.String, "SITE_NAME"),
		series.New(stationLabels, series.String, "FLOW_STATION"),
		series.New(stationEastings, series.Int, "FLOW_EASTING"),
		series.New(stationNorthings, series.Int, "FLOW_NORTHING"),
		series.New(distMetres, series.Float, "DIST_METRES"),
		series.New(measureIDs, series.String, "FLOW_MEASURE_ID"),
		series.New(matched, series.Bool, "MATCHED"),
	)
}
