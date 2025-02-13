from typing import List, Callable

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinuteSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.second - 1.0) / 59.0)


class SecondOfMinuteCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.second - 1.0) / 59.0)


class MinuteOfHourSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.minute - 1.0) / 59.0)


class MinuteOfHourCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.minute - 1.0) / 59.0)


class HourOfDaySine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.hour - 1.0) / 23.0)


class HourOfDayCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.hour - 1.0) / 23.0)


class DayOfWeekSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.dayofweek - 1.0) / 6.0)


class DayOfWeekCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.dayofweek - 1.0) / 6.0)


class DayOfMonthSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.day - 1.0) / 30.0)


class DayOfMonthCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.day - 1.0) / 30.0)


class DayOfYearSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.dayofyear - 1.0) / 365.0)


class DayOfYearCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.dayofyear - 1.0) / 365.0)


class MonthOfYearSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.month - 1.0) / 11.0)


class MonthOfYearCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.month - 1.0) / 11.0)


class WeekOfYearSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        try:
            week = index.isocalendar().week
        except AttributeError:
            week = index.week
        return np.sin(2 * np.pi * (week - 1.0) / 52.0)


class WeekOfYearCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        try:
            week = index.isocalendar().week
        except AttributeError:
            week = index.week
        return np.cos(2 * np.pi * (week - 1.0) / 52.0)


def time_features_from_frequency_str(freq_str: str) -> List[Callable]:
    features_by_offsets = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.QuarterBegin: [MonthOfYearSine(), MonthOfYearCosine()],
        offsets.QuarterEnd: [MonthOfYearSine(), MonthOfYearCosine()],
        offsets.MonthBegin: [MonthOfYearSine(), MonthOfYearCosine()],
        offsets.MonthEnd: [MonthOfYearSine(), MonthOfYearCosine()],
        offsets.Week: [DayOfMonthSine(), DayOfMonthCosine(), WeekOfYearSine(), WeekOfYearCosine()],
        offsets.Day: [DayOfWeekSine(), DayOfWeekCosine(), DayOfMonthSine(), DayOfMonthCosine(), DayOfYearSine(), DayOfYearCosine()],
        offsets.BusinessDay: [DayOfWeekSine(), DayOfWeekCosine(), DayOfMonthSine(), DayOfMonthCosine(), DayOfYearSine(), DayOfYearCosine()],
        # Modify the Hour features to ensure correct range
        offsets.Hour: [
            DayOfYearSine(), DayOfYearCosine(),    # Yearly
            DayOfMonthSine(), DayOfMonthCosine(),  # Days go first (0-31)
            DayOfWeekSine(), DayOfWeekCosine(),    # Then weekdays (0-6) 
            HourOfDaySine(), HourOfDayCosine(),    # Then hours (0-23)
        ],
        offsets.Minute: [MinuteOfHourSine(), MinuteOfHourCosine()],
        offsets.Second: [SecondOfMinuteSine(), SecondOfMinuteCosine()]
    }

    offset = to_offset(freq_str)

    for offset_type, features in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return features

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:
    
    """

    for offset_cls in features_by_offsets:
        offset = offset_cls()
        supported_freq_msg += (
            f"\t{offset.freqstr.split('-')[0]} - {offset_cls.__name__}"
        )

    raise RuntimeError(supported_freq_msg)


def custom_time_features(dates, freq='h'):
    dates = pd.PeriodIndex(dates, freq=freq)
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
