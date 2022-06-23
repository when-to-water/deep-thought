import datetime
import json
import logging
import sys
from typing import Any

import awswrangler as wr
import pandas as pd
import scipy.signal as signal
from dotenv import load_dotenv
from numpy.polynomial import Polynomial

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] %(message)s", level=logging.INFO
)

SENSOR_PLANT_MAPPING: dict = {
    "PWS_1": "Goldfruchtpalme",
    "PWS_2": "Pilea",
    "PWS_3": "Drachenbaum",
}
PLANTS: tuple[str, ...] = tuple(SENSOR_PLANT_MAPPING.values())
DISTANCE = 3
PROMINENCE = 2

input_args = '{"Goldfruchtpalme": 50, "Pilea":30, "Drachenbaum": 20 }'


def main() -> None:
    try:
        min_moistures = prep_input(json.loads(input_args))
    except json.JSONDecodeError:
        logging.critical("Input was not well-formed JSON!")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error while parsing input: {e}")
        sys.exit(1)

    wanted_plants = tuple(min_moistures.keys())

    sensor_filter = "'{0}'".format(
        "','".join([sensor_name_by_plant(plant) for plant in wanted_plants])
    )

    df: pd.DataFrame = wr.timestream.query(
        f'SELECT * FROM "when-to-water"."sensor-data" WHERE "sensor_name" in ({sensor_filter})'  # nosec: query is safe. awswrangler does not allow query parameters.
    )
    logging.info(f"Retrieved {len(df)} records")

    df = general_transformations(df)

    newest_times: dict[str, pd.Timestamp] = {}
    newest_moistures: dict[str, float] = {}
    for plant in wanted_plants:
        newest_times[plant] = df[df["plant"] == plant]["time"].max()
        newest_moistures[plant] = df[
            (df["plant"] == plant) & (df["time"] == newest_times[plant])
        ]["soil moisture in %"].iloc[0]

    df = identify_valleys_peaks(df, newest_times)

    df = remove_ascends(df)

    polyfits: dict[str, Polynomial] = fit(df)

    next_watering: dict[str, tuple[float, float]] = calc_next_watering(
        min_moistures, newest_moistures, polyfits, newest_times
    )

    last_week_moistures: dict[str, tuple[float, ...]] = get_last_week_moistures(
        df, wanted_plants
    )

    return_dict: dict[str, dict[str, Any]] = {}

    for plant in wanted_plants:
        return_dict[plant] = {
            "last_measuring": {
                "time": (newest_times[plant] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'),  # recommended by pandas
                "value": newest_moistures[plant],
            },
            "next_watering": next_watering[plant],
            "last_week_moistures": last_week_moistures[plant],
        }

    return_json = json.dumps(return_dict)

    logging.info(f"Returning '{return_json}'")


def sensor_name_by_plant(wanted_plant: str) -> str:
    return [
        sensor_name
        for sensor_name, plant in SENSOR_PLANT_MAPPING.items()
        if plant == wanted_plant
    ][0]


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

    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    return df


def identify_valleys_peaks(
    df: pd.DataFrame, newest_times: dict[str, pd.Timestamp]
) -> pd.DataFrame:
    df = df.copy()

    all_peaks: list = []
    all_valleys: list = []
    last_extreme_peak: dict = {}

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
        all_peaks += list([df_plant.index[peak] for peak in peaks])
        all_valleys += list([df_plant.index[valley] for valley in valleys])

        last_extreme_peak[plant] = max(all_peaks) > max(all_valleys)
        if last_extreme_peak:
            all_valleys += list(df_plant.index[df_plant["time"] == newest_times[plant]])

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
                    # correcting last_valley
                    last_valley_iloc = last_valley + 1
                    df_candidate = df_plant.iloc[last_peak:last_valley_iloc].copy()
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
        polyfits[plant] = Polynomial.fit(  # type: ignore
            df[df["plant"] == plant]["offset"],
            df[df["plant"] == plant]["soil moisture in %"],
            1,
        )
    return polyfits


def prep_input(input_dict: dict[str, int]) -> dict[str, int]:
    input_dict = dict(
        (key.capitalize(), int(value)) for (key, value) in input_dict.items()
    )

    for key in input_dict.keys():
        if key not in PLANTS:
            logging.critical(f'"{key}" is not a plant we know.')
            sys.exit(1)

    for value in input_dict.values():
        if not (1 <= value <= 100):
            logging.critical(
                f'Minimum moisture of "{value}" % is not allowed for {key}.'
            )
            sys.exit(1)

    return input_dict


def calc_next_watering(
    min_moistures: dict[str, int],
    newest_moistures: dict[str, float],
    polyfits: dict[str, Polynomial],
    newest_times: dict[str, pd.Timestamp],
) -> dict[str, tuple[float, float]]:
    watering_days = {}
    for plant in min_moistures.keys():
        newest_moisture = newest_moistures[plant]
        root_current = (polyfits[plant] - newest_moisture).roots()[0]
        root_minimum = (polyfits[plant] - min_moistures[plant]).roots()[0]
        days_after_last_measurement = root_minimum - root_current
        days_after_now = days_after_last_measurement - (
            (datetime.datetime.now() - newest_times[plant]).total_seconds() / 3600 / 24
        )
        watering_days[plant] = (days_after_last_measurement, days_after_now)
    return watering_days


def get_last_week_moistures(
    df: pd.DataFrame, wanted_plants: tuple[str, ...]
) -> dict[str, tuple[float, ...]]:
    last_week_moistures = {}
    df = df.copy()
    df = df[df["time"] > datetime.datetime.now() - datetime.timedelta(days=7)]
    df.set_index("time", inplace=True)
    df = df.groupby(["plant"]).resample("D").median().reset_index()
    for plant in wanted_plants:
        last_week_moistures[plant] = tuple(
            df[df["plant"] == plant]["soil moisture in %"]
        )
    return last_week_moistures


if __name__ == "__main__":
    main()
