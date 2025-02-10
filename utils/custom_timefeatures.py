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
        return np.sin(2 * np.pi * index.second / 60)


class SecondOfMinuteCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * index.second / 60)


class MinuteOfHourSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * index.minute / 60)


class MinuteOfHourCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * index.minute / 60)


class HourOfDaySine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * index.hour / 24)


class HourOfDayCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * index.hour / 24)


class DayOfWeekSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * index.dayofweek / 7)


class DayOfWeekCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * index.dayofweek / 7)


class DayOfMonthSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.day - 1) / 31)


class DayOfMonthCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.day - 1) / 31)


class DayOfYearSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.dayofyear - 1) / 366)


class DayOfYearCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.dayofyear - 1) / 366)


class MonthOfYearSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.sin(2 * np.pi * (index.month - 1) / 12)


class MonthOfYearCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.cos(2 * np.pi * (index.month - 1) / 12)


class WeekOfYearSine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        try:
            week = index.isocalendar().week
        except AttributeError:
            week = index.week
        return np.sin(2 * np.pi * (week - 1) / 53)


class WeekOfYearCosine(TimeFeature):
    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        try:
            week = index.isocalendar().week
        except AttributeError:
            week = index.week
        return np.cos(2 * np.pi * (week - 1) / 53)


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
        offsets.Hour: [HourOfDaySine(), HourOfDayCosine(), DayOfWeekSine(), DayOfWeekCosine(), DayOfMonthSine(), DayOfMonthCosine(), DayOfYearSine(), DayOfYearCosine()],
        offsets.Minute: [MinuteOfHourSine(), MinuteOfHourCosine(), HourOfDaySine(), HourOfDayCosine(), DayOfWeekSine(), DayOfWeekCosine(), DayOfMonthSine(), DayOfMonthCosine(), DayOfYearSine(), DayOfYearCosine()],
        offsets.Second: [SecondOfMinuteSine(), SecondOfMinuteCosine(), MinuteOfHourSine(), MinuteOfHourCosine(), HourOfDaySine(), HourOfDayCosine(), DayOfWeekSine(), DayOfWeekCosine(), DayOfMonthSine(), DayOfMonthCosine(), DayOfYearSine(), DayOfYearCosine()],
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
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
