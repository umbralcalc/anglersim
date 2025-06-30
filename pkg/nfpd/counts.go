package nfpd

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func flatten(records [][]string) []string {
	lines := make([]string, len(records))
	for i, row := range records {
		lines[i] = strings.Join(row, ",")
	}
	return lines
}

func GetUniqueSitesDataFrameFromCountsCSV(csvPath string) dataframe.DataFrame {
	file, err := os.Open(csvPath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var headers []string
	siteNameIndex := -1
	siteIdIndex := -1
	siteByName := make(map[string]int)

	// Read header line
	if scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		headers, err = r.Read()
		if err != nil {
			panic(fmt.Errorf("failed to parse header: %v", err))
		}

		for i, h := range headers {
			if h == "SITE_ID" {
				siteIdIndex = i
			} else if h == "SITE_NAME" {
				siteNameIndex = i
				break
			}
		}
		if siteNameIndex == -1 {
			panic(fmt.Errorf("SITE_NAME column not found"))
		}
		if siteIdIndex == -1 {
			panic(fmt.Errorf("SITE_ID column not found"))
		}
	}

	// Scan remaining lines and extract unique site names
	for scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		record, err := r.Read()
		if err != nil ||
			len(record) <= siteNameIndex ||
			len(record) <= siteIdIndex {
			continue
		}
		name := record[siteNameIndex]
		site := record[siteIdIndex]
		if name != "" && site != "" {
			siteInt, err := strconv.Atoi(site)
			if err != nil {
				panic(err)
			}
			siteByName[name] = siteInt
		}
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	uniqueSites := make([]int, 0, len(siteByName))
	uniqueNames := make([]string, 0, len(siteByName))
	for name, site := range siteByName {
		uniqueSites = append(uniqueSites, site)
		uniqueNames = append(uniqueNames, name)
	}
	df := dataframe.New(
		series.New(uniqueSites, series.Int, "SITE_ID"),
		series.New(uniqueNames, series.String, "SITE_NAME"),
	)
	return df
}

func GetSiteCountsDataFrameFromCSV(
	csvPath string,
	siteName string,
) dataframe.DataFrame {
	file, err := os.Open(csvPath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var headers []string
	var filteredRows [][]string

	// Read and parse the header
	if scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		headers, err = r.Read()
		if err != nil {
			panic(err)
		}
		filteredRows = append(filteredRows, headers)
	}

	// Find index of SITE_NAME column
	siteNameIndex := -1
	for i, h := range headers {
		if h == "SITE_NAME" {
			siteNameIndex = i
			break
		}
	}
	if siteNameIndex == -1 {
		panic("SITE_NAME column not found")
	}

	// Filter rows using bufio.Scanner
	for scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		record, err := r.Read()
		if err != nil {
			continue // skip malformed line
		}
		if record[siteNameIndex] == siteName {
			filteredRows = append(filteredRows, record)
		}
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	// Create dataframe from filtered records
	fdf := dataframe.ReadCSV(
		strings.NewReader(strings.Join(flatten(filteredRows), "\n")))

	// Convert EVENT_DATE to timestamp
	timestamps := make([]float64, fdf.Nrow())
	for i, record := range fdf.Col("EVENT_DATE").Records() {
		t, err := time.Parse("02/01/2006", record)
		if err != nil {
			panic(err)
		}
		timestamps[i] = float64(t.Unix())
	}
	fdf = fdf.Mutate(series.New(timestamps, series.Float, "TIMESTAMP"))

	return fdf
}
