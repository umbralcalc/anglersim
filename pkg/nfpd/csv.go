package nfpd

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strings"
)

func GetUniqueSiteNames(csvPath string) ([]string, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var headers []string
	siteNameIndex := -1
	siteSet := make(map[string]struct{})

	// Read header line
	if scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		headers, err = r.Read()
		if err != nil {
			return nil, fmt.Errorf("failed to parse header: %v", err)
		}

		for i, h := range headers {
			if h == "SITE_NAME" {
				siteNameIndex = i
				break
			}
		}
		if siteNameIndex == -1 {
			return nil, fmt.Errorf("SITE_NAME column not found")
		}
	}

	// Scan remaining lines and extract unique site names
	for scanner.Scan() {
		line := scanner.Text()
		r := csv.NewReader(strings.NewReader(line))
		record, err := r.Read()
		if err != nil || len(record) <= siteNameIndex {
			continue
		}
		site := record[siteNameIndex]
		if site != "" {
			siteSet[site] = struct{}{}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	// Convert map keys to sorted slice
	uniqueSites := make([]string, 0, len(siteSet))
	for site := range siteSet {
		uniqueSites = append(uniqueSites, site)
	}
	sort.Strings(uniqueSites)

	return uniqueSites, nil
}
