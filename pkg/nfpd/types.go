package nfpd

import (
	"os"

	"github.com/go-gota/gota/dataframe"
)

func GetTypesDataFrameFromCSV(
	csvPath string,
) dataframe.DataFrame {
	f, err := os.Open(csvPath)
	if err != nil {
		panic(err)
	}
	df := dataframe.ReadCSV(f)
	return df
}
