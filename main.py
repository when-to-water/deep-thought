from dotenv import load_dotenv
import awswrangler as wr
import pandas as pd
import scipy.signal as signal
from numpy.polynomial import Polynomial

SENSOR_PLANT_MAPPING: dict = {
    "PWS_1": "Goldfruchtpalme",
    "PWS_2": "Pilea",
    "PWS_3": "Drachenbaum",
}
PLANTS: tuple[str, ...] = tuple(SENSOR_PLANT_MAPPING.values())
DISTANCE = 3
PROMINENCE = 2


def main() -> None:
    load_dotenv()

    df: pd.DataFrame = wr.timestream.query(
        'SELECT * FROM "when-to-water"."sensor-data"'
    )
    print(f"Retrieved {len(df)} records")

    df = general_transformations(df)

    newest_time: dict[str, pd.Timestamp] = {}
    newest_moisture: dict[str, float] = {}
    for plant in PLANTS:
        newest_time[plant] = df[df["plant"] == plant]["time"].max()
        newest_moisture[plant] = df[
            (df["plant"] == plant) & (df["time"] == newest_time[plant])
        ]["soil moisture in %"].iloc[0]

    df = identify_valleys_peaks(df)

    df = remove_ascends(df)

    polyfits: dict[str, Polynomial] = fit(df)

    print("done")


def general_transformations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df.rename(
        columns={"measure_value::double": "value", "sensor_name": "plant"}, inplace=True
    )
    df["value"] = df["value"].astype(float)

    df["plant"] = df["plant"].map(SENSOR_PLANT_MAPPING)
    df.dropna(inplace=True)
    # drop power
    df = df[df["measure_name"] != "power"]

    # add unit to measurement name
    df["measure_name"] = df["measure_name"].str.replace("_", " ") + " in " + df["unit"]

    # drop unit
    df.drop(columns=["unit"], inplace=True)

    # remove 0 moisture
    df = df[~((df["measure_name"] == "soil moisture in %") & (df["value"] == 0))]

    # Resample df to hourly measures
    df.set_index("time", inplace=True)
    df = df.groupby(["plant", "measure_name"]).resample("H").mean().reset_index()

    df.set_index(["time", "plant", "measure_name"], inplace=True)
    df = df.unstack().reset_index()
    df.columns = [
        " ".join(col).strip().replace("value ", "") for col in df.columns.values
    ]
    df.reset_index(drop=True, inplace=True)

    df.set_index("time", inplace=True)
    for plant in PLANTS:
        df[df["plant"] == plant] = df[df["plant"] == plant].interpolate(method="time")

    df.reset_index(inplace=True)

    return df


def identify_valleys_peaks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    all_peaks: list = []
    all_valleys: list = []

    for plant in PLANTS:
        df_plant = df[df["plant"] == plant]
        peaks = signal.find_peaks(
            df_plant["soil moisture in %"],
            distance=DISTANCE,
            prominence=PROMINENCE,
        )[0]
        valleys = signal.find_peaks(
            -df_plant["soil moisture in %"],
            distance=DISTANCE,
            prominence=PROMINENCE,
        )[0]
        # translate row to index
        all_peaks += [df_plant.index[peak] for peak in peaks]
        all_valleys += [df_plant.index[valley] for valley in valleys]

    df["peak"] = df.index.isin(all_peaks)
    df["valley"] = df.index.isin(all_valleys)

    return df


def remove_ascends(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    decending_dfs: list[pd.DataFrame] = []
    for plant in PLANTS:
        last_peak = -1
        last_valley = -1
        df_plant = df[df["plant"] == plant].copy()
        df_plant.reset_index(drop=True, inplace=True)
        for row in df_plant.itertuples():
            if row.peak:
                last_peak = row.Index
            if row.valley:
                last_valley = row.Index
                if last_peak > -1 and last_peak < last_valley:
                    df_candidate = df_plant.iloc[last_peak:last_valley].copy()
                    mininmum_dt = df_candidate["time"].min()
                    # offset in days
                    df_candidate["offset"] = (
                        (df_candidate["time"] - mininmum_dt).dt.total_seconds()
                        / 3600
                        / 24
                    )
                    if (
                        df_candidate.iloc[0]["soil moisture in %"]
                        < df_candidate.iloc[-1]["soil moisture in %"]
                    ):
                        continue

                    # remove outliers
                    # df_candidate = remove_outliers(df_candidate,["soil moisture in %"],1)
                    if df_candidate.empty:
                        continue

                    # normalize
                    df_candidate["soil moisture in %"] = df_candidate[
                        "soil moisture in %"
                    ] + (100 - df_candidate["soil moisture in %"].max())

                    decending_dfs.append(df_candidate)

    return pd.concat(decending_dfs)


def fit(df: pd.DataFrame) -> dict[str, Polynomial]:
    polyfits: dict = {}
    for plant in PLANTS:
        polyfits[plant] = Polynomial.fit(
            df[df["plant"] == plant]["offset"],
            df[df["plant"] == plant]["soil moisture in %"],
            1,
        )
    return polyfits


if __name__ == "__main__":
    main()
