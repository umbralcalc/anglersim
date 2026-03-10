package data

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

// SiteCoverage holds survey coverage stats for a single site.
type SiteCoverage struct {
	SiteID     int
	SiteName   string
	Area       string
	NumYears   int
	YearSpan   int
	FirstYear  int
	LastYear   int
	NumSurveys int
	NumSpecies int
}

// CoverageReport holds aggregate coverage statistics.
type CoverageReport struct {
	TotalRecords  int
	NumSites      int
	NumSpecies    int
	MinYear       int
	MaxYear       int
	Sites         []SiteCoverage
	SpeciesCounts map[string]int
	YearCounts    map[int]int
	AreaCounts    map[string]int
}

// GetCoverageReport scans the counts CSV once and returns coverage statistics.
func GetCoverageReport(countsCSVPath string) CoverageReport {
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

	type siteAccum struct {
		name    string
		area    string
		years   map[int]bool
		species map[string]bool
		surveys map[int]bool
		first   int
		last    int
	}

	sites := make(map[int]*siteAccum)
	speciesGlobal := make(map[string]int)
	yearGlobal := make(map[int]int)
	areaGlobal := make(map[string]int)
	totalRows := 0

	siteIDIdx := colIdx["SITE_ID"]
	siteNameIdx := colIdx["SITE_NAME"]
	yearIdx := colIdx["EVENT_DATE_YEAR"]
	speciesIdx := colIdx["SPECIES_NAME"]
	surveyIdx := colIdx["SURVEY_ID"]
	areaIdx := colIdx["AREA"]

	for scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		record, err := r.Read()
		if err != nil || len(record) <= speciesIdx {
			continue
		}
		totalRows++

		siteID, _ := strconv.Atoi(record[siteIDIdx])
		year, _ := strconv.Atoi(record[yearIdx])
		species := record[speciesIdx]
		surveyID, _ := strconv.Atoi(record[surveyIdx])
		area := record[areaIdx]

		speciesGlobal[species]++
		yearGlobal[year]++
		areaGlobal[area]++

		s, ok := sites[siteID]
		if !ok {
			s = &siteAccum{
				name:    record[siteNameIdx],
				area:    area,
				years:   make(map[int]bool),
				species: make(map[string]bool),
				surveys: make(map[int]bool),
			}
			sites[siteID] = s
		}
		s.years[year] = true
		s.species[species] = true
		s.surveys[surveyID] = true
		if s.first == 0 || year < s.first {
			s.first = year
		}
		if year > s.last {
			s.last = year
		}
	}

	// Build site list
	siteList := make([]SiteCoverage, 0, len(sites))
	for id, s := range sites {
		siteList = append(siteList, SiteCoverage{
			SiteID:     id,
			SiteName:   s.name,
			Area:       s.area,
			NumYears:   len(s.years),
			YearSpan:   s.last - s.first + 1,
			FirstYear:  s.first,
			LastYear:   s.last,
			NumSurveys: len(s.surveys),
			NumSpecies: len(s.species),
		})
	}
	sort.Slice(siteList, func(i, j int) bool {
		return siteList[i].NumYears > siteList[j].NumYears
	})

	years := make([]int, 0, len(yearGlobal))
	for y := range yearGlobal {
		years = append(years, y)
	}
	sort.Ints(years)

	return CoverageReport{
		TotalRecords:  totalRows,
		NumSites:      len(sites),
		NumSpecies:    len(speciesGlobal),
		MinYear:       years[0],
		MaxYear:       years[len(years)-1],
		Sites:         siteList,
		SpeciesCounts: speciesGlobal,
		YearCounts:    yearGlobal,
		AreaCounts:    areaGlobal,
	}
}

// SitesWithMinYears returns sites with at least minYears distinct survey years.
func (r *CoverageReport) SitesWithMinYears(minYears int) []SiteCoverage {
	var result []SiteCoverage
	for _, s := range r.Sites {
		if s.NumYears >= minYears {
			result = append(result, s)
		}
	}
	return result
}

// SiteCoverageDataFrame returns a dataframe of all sites sorted by survey years.
func (r *CoverageReport) SiteCoverageDataFrame() dataframe.DataFrame {
	ids := make([]int, len(r.Sites))
	names := make([]string, len(r.Sites))
	areas := make([]string, len(r.Sites))
	numYears := make([]int, len(r.Sites))
	spans := make([]int, len(r.Sites))
	firstYears := make([]int, len(r.Sites))
	lastYears := make([]int, len(r.Sites))
	numSurveys := make([]int, len(r.Sites))
	numSpecies := make([]int, len(r.Sites))

	for i, s := range r.Sites {
		ids[i] = s.SiteID
		names[i] = s.SiteName
		areas[i] = s.Area
		numYears[i] = s.NumYears
		spans[i] = s.YearSpan
		firstYears[i] = s.FirstYear
		lastYears[i] = s.LastYear
		numSurveys[i] = s.NumSurveys
		numSpecies[i] = s.NumSpecies
	}

	return dataframe.New(
		series.New(ids, series.Int, "SITE_ID"),
		series.New(names, series.String, "SITE_NAME"),
		series.New(areas, series.String, "AREA"),
		series.New(numYears, series.Int, "NUM_YEARS"),
		series.New(spans, series.Int, "YEAR_SPAN"),
		series.New(firstYears, series.Int, "FIRST_YEAR"),
		series.New(lastYears, series.Int, "LAST_YEAR"),
		series.New(numSurveys, series.Int, "NUM_SURVEYS"),
		series.New(numSpecies, series.Int, "NUM_SPECIES"),
	)
}

// YearCountsDataFrame returns records-per-year as a dataframe.
func (r *CoverageReport) YearCountsDataFrame() dataframe.DataFrame {
	years := make([]int, 0, len(r.YearCounts))
	for y := range r.YearCounts {
		years = append(years, y)
	}
	sort.Ints(years)

	counts := make([]int, len(years))
	for i, y := range years {
		counts[i] = r.YearCounts[y]
	}

	return dataframe.New(
		series.New(years, series.Int, "YEAR"),
		series.New(counts, series.Int, "RECORDS"),
	)
}

// SpeciesCountsDataFrame returns species record counts as a dataframe.
func (r *CoverageReport) SpeciesCountsDataFrame() dataframe.DataFrame {
	type sc struct {
		name  string
		count int
	}
	spp := make([]sc, 0, len(r.SpeciesCounts))
	for name, count := range r.SpeciesCounts {
		spp = append(spp, sc{name, count})
	}
	sort.Slice(spp, func(i, j int) bool {
		return spp[i].count > spp[j].count
	})

	names := make([]string, len(spp))
	counts := make([]int, len(spp))
	for i, s := range spp {
		names[i] = s.name
		counts[i] = s.count
	}

	return dataframe.New(
		series.New(names, series.String, "SPECIES"),
		series.New(counts, series.Int, "RECORDS"),
	)
}

// PrintSummary prints a text summary of the coverage report.
func (r *CoverageReport) PrintSummary() {
	fmt.Printf("NFPD Counts Coverage Summary\n")
	fmt.Printf("============================\n")
	fmt.Printf("Total records:  %d\n", r.TotalRecords)
	fmt.Printf("Unique sites:   %d\n", r.NumSites)
	fmt.Printf("Unique species: %d\n", r.NumSpecies)
	fmt.Printf("Year range:     %d–%d\n\n", r.MinYear, r.MaxYear)

	buckets := []struct {
		label string
		min   int
		max   int
	}{
		{"1 year", 1, 1},
		{"2-4 years", 2, 4},
		{"5-9 years", 5, 9},
		{"10-14 years", 10, 14},
		{"15-19 years", 15, 19},
		{"20+ years", 20, 9999},
	}
	fmt.Println("Sites by survey years:")
	for _, b := range buckets {
		count := 0
		for _, s := range r.Sites {
			if s.NumYears >= b.min && s.NumYears <= b.max {
				count++
			}
		}
		fmt.Printf("  %-15s %d sites\n", b.label, count)
	}
}
