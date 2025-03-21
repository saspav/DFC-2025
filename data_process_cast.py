import re
import os
import random
import numpy as np
import pandas as pd
import polars as pl
import joblib
import gc

import featuretools as ft
from woodwork.logical_types import Age, Categorical, Datetime

from io import StringIO
from datetime import timedelta
from glob import glob
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, normalize
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_log_error, explained_variance_score

from df_addons import memory_compression, memory_compression_pl
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

WORK_PATH = Path('Z:/python-datasets/4cast')
if not WORK_PATH.is_dir():
    WORK_PATH = Path('D:/python-datasets/4cast')

if not WORK_PATH.is_dir():
    WORK_PATH = Path('.')

DATASET_PATH = WORK_PATH.joinpath('data')

if not WORK_PATH.is_dir():
    WORK_PATH = Path('.')
    DATASET_PATH = WORK_PATH

MODEL_PATH = WORK_PATH.joinpath('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

WORK_PATH.joinpath('best_model').mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = WORK_PATH.joinpath('predictions')
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_LOGS = WORK_PATH.joinpath('scores_local.logs')
MODELS_LOGS_REG = WORK_PATH.joinpath('scores_local_reg.logs')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('/kaggle/input/dfc-2025-4cast')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('/kaggle/input')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('.')
    __file__ = Path('.')
    LOCAL_FILE = ''
else:
    LOCAL_FILE = '_local'

RANDOM_SEED = 127


def get_max_num(log_file=None):
    """Получение максимального номера итерации обучения моделей
    :param log_file: имя лог-файла с полным путем
    :return: максимальный номер
    """
    if log_file is None:
        log_file = MODELS_LOGS

    if not log_file.is_file():
        with open(log_file, mode='a') as log:
            log.write('num;mdl;fold;mdl_score;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;'
                      'f1_wght;tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        # Чтение файла как строки
        with open(log_file, encoding='utf-8') as file:
            file_rows = file.read()
        # Удаление переносов строки в кривых строках и загрузка файла в ДФ
        df = pd.read_csv(StringIO(file_rows.replace(',\n', ',')), sep=';', index_col=False)
        if df.empty:
            max_num = 0
        else:
            max_num = df.num.max()
    return int(max_num) if max_num is not None else 0


def clean_column_name(col_name):
    col_name = col_name.lower()  # преобразование в нижний регистр
    col_name = col_name.replace('(', '_')  # замена скобок на _
    col_name = col_name.replace(')', '')  # удаление закрывающих скобок
    col_name = col_name.replace('.', '_')  # замена точек на _
    col_name = re.sub(r'(?<=\d)_(?=\d)', '', col_name)  # удаление подчеркивания между числами
    return col_name


class DataTransform:
    def __init__(self, use_catboost=True, numeric_columns=None, category_columns=None,
                 features2drop=None, scaler=None, args_scaler=None, samples_paths=None,
                 **kwargs):
        """
        Преобразование данных
        :param use_catboost: данные готовятся для catboost
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param drop_first: из dummy переменных удалить первую колонку
        :param scaler: какой скайлер будем использовать
        :param args_scaler: аргументы для скайлера, например: степень для полином.преобразов.
        :param samples_paths: Путь, куда сохранять кусочки данных: None - не сохранять,
                              True - сохранять в DATASET_PATH, каталог
        """
        self.use_catboost = use_catboost
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.features2drop = [] if features2drop is None else features2drop
        self.exclude_columns = []
        self.comment = {}
        self.preprocessor = None
        self.scaler = scaler
        self.args_scaler = args_scaler
        self.samples_paths = samples_paths
        self.preprocess_files = 'preprocess_files.pkl'
        self.aggregate_path_file = 'aggregate_data_files.pkl'
        self.aggregate_inn_trn_file = 'aggregate_data_days_files.pkl'

    def set_category(self, df):
        for col_name in self.category_columns:
            if col_name in df.columns:
                df[col_name] = df[col_name].astype('category')
        return df

    def fit(self, df):
        """
        Формирование фич
        :param df: исходный ФД
        :return: ДФ с агрегациями
        """
        # если нет цифровых колонок --> заполним их
        if self.category_columns and not self.numeric_columns:
            self.numeric_columns = [col_name for col_name in df.columns if col_name
                                    not in self.category_columns + self.features2drop]
        # если нет категориальных колонок --> заполним их
        if self.numeric_columns and not self.category_columns:
            self.category_columns = [col_name for col_name in df.columns if col_name
                                     not in self.numeric_columns + self.features2drop]

        start_time = print_msg('Группировка по целевому признаку...')

        print_time(start_time)

        return df

    def set_category(self, df):
        for col_name in self.category_columns:
            if col_name in df.columns:
                df[col_name] = df[col_name].astype('category')
        return df

    def transform(self, df, model_columns=None):
        """
        Формирование остальных фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        # Сохраняем исходный индекс ДФ
        original_index = df.index

        df = self.set_category(df)

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df)

        return df

    def fit_transform(self, df, model_columns=None):
        """
        Fit + transform data
        :param df: исходный ФД
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        df = self.fit(df)
        df = self.transform(df, model_columns=model_columns)
        return df

    def preprocess_data(self, remake_file=False, use_sum_in_file=False, use_cls_trn=False,
                        use_cls_vect=False, fill_nan=True, sample=None, return_pandas=False,
                        **kwargs):
        """
        Предобработка данных
        :param remake_file: Переформировать файлы с агрегациями
        :param use_sum_in_file: Использовать файл с входными суммами
        :param use_cls_trn: Использовать данные о суммах транзакций по классам
        :param use_cls_vect: Использовать данные о частотности транзакций по классам
        :param fill_nan: заполняем пропуски в данных
        :param sample: вернуть ДФ из указанного количества inn_id
        :param return_pandas: вернуть ДФ из панд
        :return: ДФ
        """
        preprocess_files = None

        if self.preprocess_files:
            preprocess_files = WORK_PATH.joinpath(self.preprocess_files)

            if preprocess_files.is_file() and not remake_file and sample is None:
                start_time = print_msg('Читаю подготовленные данные...')
                with open(preprocess_files, 'rb') as in_file:
                    calendar, inn, profiles, inn_test = joblib.load(in_file)
                print_time(start_time)
                if return_pandas:
                    return (calendar.to_pandas(), inn.to_pandas(), profiles.to_pandas(),
                            inn_test.to_pandas())
                return calendar, inn, profiles, inn_test

        start_time = print_msg('Загрузка данных...')

        # Читаем календарь и отмечаем дни недели
        calendar = pl.read_csv(DATASET_PATH.joinpath("calendar.csv"),
                               try_parse_dates=True).with_columns(
            pl.col("week").cast(pl.UInt8),
            pl.col("date").dt.weekday().alias("day_of_week").cast(pl.Int8),
        )
        holidays = pd.read_csv(DATASET_PATH.joinpath("holidays.csv"), sep=';')
        holidays['HOLIDAY'] = pd.to_datetime(holidays['HOLIDAY'], format='%d.%m.%Y')
        # Преобразуем колонку 'HOLIDAY' в Polars Series
        holidays = pl.DataFrame(holidays[['HOLIDAY']]).with_columns(
            pl.col('HOLIDAY').cast(pl.Date))['HOLIDAY'].to_list()
        # Добавляем колонку is_workday
        calendar = calendar.with_columns(
            (pl.col("date").is_in(holidays).not_()).alias("is_workday")
        )
        if use_sum_in_file or use_cls_trn or use_cls_vect:
            file_target_series = DATASET_PATH / "target_series_sum_in.parquet"
            if not file_target_series.is_file():
                file_target_series = DATASET_PATH / "target_series_sum_in_sample.parquet"
            inn = pl.read_parquet(file_target_series).with_columns(
                pl.col("week").cast(pl.UInt8))
            # Удаляем колонки с суммами по классам транзакций
            if not use_cls_trn:
                # Колонки с суммами по классам транзакций
                trn_ce_columns = [col for col in inn.columns if col.startswith('trn_ce_')]
                inn = inn.drop(trn_ce_columns)
            if not use_cls_vect:
                # Колонки с суммами по классам транзакций
                trn_ce_columns = [col for col in inn.columns if col.startswith('tvect_')]
                inn = inn.drop(trn_ce_columns)
            # Удаляем колонки с входящими суммами
            if not use_sum_in_file:
                sum_in_columns = [col for col in inn.columns if '_inp' in col]
                inn = inn.drop(sum_in_columns)
        else:
            file_target_series = DATASET_PATH.joinpath("target_series.parquet")
            if not file_target_series.is_file():
                file_target_series = DATASET_PATH.joinpath("target_series_sample.parquet")
            inn = pl.read_parquet(file_target_series).with_columns(
                pl.col("week").cast(pl.UInt8))
            file_target_series_ext = DATASET_PATH.joinpath("target_series_extended.parquet")
            if file_target_series_ext.is_file():
                inn_ext = pl.read_parquet(file_target_series_ext).with_columns(
                    pl.col("week").cast(pl.UInt8))
                inn = pl.concat([inn, inn_ext])
                # print(inn["week"].unique())
            # print(inn)

        file_profiles = DATASET_PATH.joinpath("profiles.parquet")
        if not file_profiles.is_file():
            file_profiles = DATASET_PATH.joinpath("profiles_sample.parquet")
        profiles = pl.read_parquet(file_profiles)

        file_test = DATASET_PATH.joinpath("sample_submit.csv")
        if not file_test.is_file():
            file_test = DATASET_PATH.joinpath("sample_submit_sample.csv")
        inn_test = pl.read_csv(file_test, dtypes={"week": pl.UInt8, "predict": pl.Float64})

        if sample is not None:
            sample_id = inn['inn_id'].unique().to_list()[:sample]
            # Фильтруем DataFrame df_inn
            sample_inn = inn.filter(pl.col('inn_id').is_in(sample_id))
            # Фильтруем DataFrame profiles
            sample_profiles = profiles.filter(pl.col('inn_id').is_in(sample_id))
            # Фильтруем DataFrame sample_submit
            sample_submit = pl.read_csv(file_test,
                                        dtypes={"week": pl.UInt8, "predict": pl.Float64})
            sample_submit = sample_submit.filter(pl.col('inn_id').is_in(sample_id))

            inn, profiles, inn_test = sample_inn, sample_profiles, sample_submit

            if self.samples_paths is not None and self.samples_paths:
                if isinstance(self.samples_paths, int):
                    self.samples_paths = DATASET_PATH
                else:
                    self.samples_paths = Path(self.samples_paths)
                if use_sum_in_file or use_cls_trn:
                    sample_inn.write_parquet(
                        DATASET_PATH.joinpath("target_series_sum_in_sample.parquet"))
                else:
                    sample_inn.write_parquet(
                        DATASET_PATH.joinpath("target_series_sample.parquet"))
                sample_profiles.write_parquet(
                    DATASET_PATH.joinpath("profiles_sample.parquet"))
                sample_submit.write_csv(DATASET_PATH.joinpath("sample_submit_sample.csv"))

        print_time(start_time)

        if self.preprocess_files:
            save_time = print_msg('Сохраняем предобработанные данные...')
            with open(preprocess_files, 'wb') as file:
                joblib.dump((calendar, inn, profiles, inn_test), file, compress=7)
            print_time(save_time)

        if return_pandas:
            return (calendar.to_pandas(), inn.to_pandas(), profiles.to_pandas(),
                    inn_test.to_pandas())

        return calendar, inn, profiles, inn_test

    @staticmethod
    def _make_agg(df_inn: pl.DataFrame, delta=1, shift_start=1, shift_total=24, w_size=48):
        """
        Разные группировки
        :param df_inn:
        :param delta:
        :param shift_start:
        :param shift_total:
        :param w_size:
        :return:
        """
        df_res = None
        # Задаем количество для shift'ов
        shifts = range(shift_start, shift_start + shift_total)
        # Задаем квантили
        quantiles = np.linspace(0.05, 0.95, 19).round(2)  # 0.05, 0.10, ..., 0.95
        # Колонки с суммами по классам транзакций
        trn_ce_columns = [col for col in df_inn.columns if col.startswith('trn_ce_')]
        # Колонки для различных трансформаций
        transform_columns = [col for col in df_inn.columns
                             if col not in trn_ce_columns + ['inn_id', 'week', ]]
        # Обрабатывать будем таргет и ...
        for col in tqdm(transform_columns, desc=f'Преобразую колонки delta={delta}'):
            sf = f'{col}_'
            if col not in df_inn.columns:
                continue
            # print(f'delta={delta}, transform_column:{col}')
            # Создаем колонки для shift
            shift_columns = [
                pl.col(col).shift(i - (shift_start - delta)).over("inn_id").alias(
                    f"{sf}shift{i:02d}") for i in shifts]

            # Создаем колонки для rolling функции с window_size=w_size
            rolling_columns = []
            for w in [*range(4, 17, 4)] + [w_size]:
                rolling_columns.extend(
                    [pl.col(col).rolling_mean(window_size=w, min_periods=1).shift(
                        delta).over("inn_id").alias(f"{sf}rmen{w:02}"),
                     pl.col(col).rolling_median(window_size=4, min_periods=1).shift(
                         delta).over(
                         "inn_id").alias(f"{sf}rmed{w:02}"),
                     ])

            diffs = [1, 2, 3, 4, 5, 6, 7, 8, 12,
                     # 16,
                     # 20,
                     # 24
                     ]
            diffs = [*range(1, 17)] + [*range(17, 25)]
            diff_columns = [pl.col(col).diff(i - (shift_start - delta)).over("inn_id").alias(
                f"{sf}diff{i:02d}") for i in diffs]

            # Добавляем колонки для квантилей
            quantile_columns = [
                pl.col(col).rolling_quantile(
                    quantile=q,
                    window_size=w_size,
                    min_periods=1).shift(delta).over("inn_id").alias(
                    f"{sf}iq{int(q * 100):02d}")
                for q in quantiles
            ]

            # Объединяем все колонки в один список
            quantile_columns = []  # Пока удалим колонки с квантилями
            if 'count' in col:
                diff_columns = []  # Пока удалим колонки с разностями
                rolling_columns = []
            all_columns = shift_columns + rolling_columns + quantile_columns + diff_columns
            # Добавляем все колонки сразу с помощью метода with_columns
            if df_res is None:
                df_res = df_inn.with_columns(all_columns).sort(by=["inn_id", "week"])
            else:
                df_res = df_res.with_columns(all_columns).sort(by=["inn_id", "week"])

            # Скользящее окно на лагах
            shift_rolls = [pl.col(f"{sf}shift{i:02d}")
                               .rolling_mean(window_size=i, min_periods=1)
                               .shift(delta).over("inn_id").alias(f"{sf}_srmen{i:02d}")
                           for i in (4, 8, 12, 16) if f"{sf}shift{i:02d}" in df_res.columns]
            # df_res = df_res.with_columns(shift_rolls)

        if trn_ce_columns:
            # Создаем колонки для shift классов транзакций
            shift_columns = []
            shifts = [*range(1, 17)] + [*range(17, 25)]
            for i in shifts:
                cls_cols = [pl.col(col).shift(i - (shift_start - delta)).over("inn_id").alias(
                    f"shft{i:02d}_{col}") for col in trn_ce_columns]
                shift_columns.extend(cls_cols)

            df_res = df_res.with_columns(shift_columns)

        return df_res

    def make_agg_data(self, shift_start=1, shift_total=24, w_size=48, remake_file=False,
                      use_sum_in_file=False, use_cls_trn=False, use_cls_vect=False,
                      sample=None, add_agg_data=True, log_target=True, use_featuretools=False,
                      **kwargs):
        """
        Подсчет разных агрегированных статистик
        :param shift_start: Начальный сдвиг для группировок
        :param shift_total: Количество сдвигов для группировок
        :param w_size: Ширина окна для статистик
        :param remake_file: Формируем файлы снова или читаем с диска
        :param use_sum_in_file: Использовать файл с входными суммами
        :param use_cls_trn: Использовать данные о суммах транзакций по классам
        :param use_cls_vect: Использовать данные о частотности транзакций по классам
        :param sample: вернуть ДФ из указанного количества inn_id
        :param add_agg_data: Добавляем самодельную аггрегацию
        :param log_target: Взять логарифм от целевой переменной
        :param use_featuretools: Используем модуль featuretools
        :return: ДФ трейна и теста с агрегированными данными
        """

        # Определяем квартал для каждой даты с учетом дополнительных дней
        def determine_quarter(start_date):
            # Получаем список кварталов на протяжении 7 дней
            quarters = [(start_date + timedelta(days=d)).quarter for d in range(7)]
            # Возвращаем моду (самое часто встречающееся значение)
            return pd.Series(quarters).mode()[0]  # Берем первое значение моды

        aggregate_path_file = None

        if self.aggregate_path_file:
            aggregate_path_file = WORK_PATH.joinpath(self.aggregate_path_file)

            if aggregate_path_file.is_file() and add_agg_data and not remake_file:
                start_time = print_msg('Читаю подготовленные данные...')
                with open(aggregate_path_file, 'rb') as in_file:
                    train_df, test_df = joblib.load(in_file)
                print_time(start_time)
                return train_df, test_df

        # Загрузка предобработанных данных
        calendar, inn, profiles, inn_test = self.preprocess_data(
            remake_file=remake_file, use_sum_in_file=use_sum_in_file, use_cls_trn=use_cls_trn,
            use_cls_vect=use_cls_vect, sample=sample,
        )
        # print(calendar.shape, inn.shape, profiles.shape, inn_test.shape)

        drop_cols = ['trns_amount_out_ip', 'trns_count_out_ip', 'cnt_inn_out_ip',
                     # 'trns_amount_inp', 'trns_count_inp', 'cnt_inn_inp', 'date_inp',
                     'trns_amount_inp_ip', 'trns_count_inp_ip', 'cnt_inn_inp_ip',
                     ]
        drop_cols = [col for col in drop_cols if col in inn.columns]
        if drop_cols:
            inn = inn.drop(drop_cols)

        # print(inn.columns)
        # print(inn_test.columns)
        # exit()

        start_time = print_msg('Агрегация данных...')

        # Группировка по неделям с подсчетом рабочих дней и выходных
        weeks = calendar.group_by("week").agg(
            pl.col("date").min().alias("d1"),
            pl.col("date").max().alias("d2"),
            # Количество рабочих дней
            pl.col("is_workday").sum().alias("workdays"),
            # Количество выходных
            (pl.col("is_workday").count() - pl.col("is_workday").sum()).alias("weekends"),
        ).with_columns(pl.col("d1").dt.year().cast(pl.Int16).alias("year1"),
                       pl.col("d1").dt.month().alias("month1"),
                       pl.col("d1").dt.day().alias("day1"),
                       pl.col("d1").dt.ordinal_day().alias("dy1"),
                       pl.col("d2").dt.year().cast(pl.Int16).alias("year2"),
                       pl.col("d2").dt.month().alias("month2"),
                       pl.col("d2").dt.day().alias("day2"),
                       pl.col("d2").dt.ordinal_day().alias("dy2"),
                       ).with_columns(
            ((pl.col("month2") - pl.col("month1") + 12) % 12).alias("m"),
            (pl.col("year2") - pl.col("year1")).cast(pl.Int8).alias("y"),
            pl.when((pl.col("month1") > 2) & (pl.col("year1") % 4 == 0)).then(
                pl.col("dy1") - 1).otherwise(pl.col("dy1")).alias("dy1"),
            pl.when((pl.col("month2") > 2) & (pl.col("year2") % 4 == 0)).then(
                pl.col("dy2") - 1).otherwise(pl.col("dy2")).alias("dy2"),
        ).with_columns((pl.col("year1") - 2021).cast(pl.Int8).alias("year1"),
                       (pl.col("year2") - 2021).cast(pl.Int8).alias("year2"), )

        # Добавим квартал, к которому принадлежит неделя
        weeks = weeks.to_pandas()
        weeks['q'] = weeks['d1'].apply(determine_quarter)
        weeks = pl.from_pandas(weeks)

        age = {"1m": 1,
               "2_3m": 3,
               "3_6m": 6,
               "6_12m": 12,
               "1_2y": 24,
               "2_4y": 48,
               "4_8y": 96,
               "8_12y": 144,
               "more_12y": 192,
               # "NULL":np.nan
               }
        ipul = {"ip": 0, "ul": 1, }
        # Уникальные ИНН
        test_inns = inn_test["inn_id"].unique()
        # Уберем лишнюю инфу
        profiles = profiles.filter(profiles['inn_id'].is_in(test_inns)).with_columns(
            pl.col("diff_datopen_report_date_flg").replace(age, default=0),
            pl.col("ipul").replace(ipul).cast(pl.String),
            pl.col("id_region").fill_null("-"),
            pl.col("main_okved_group").fill_null("-"),
        )
        profiles = profiles.sort("report_date").group_by("inn_id").last()
        print('profiles обработан!')

        if log_target:
            # Взятие логарифма от таргета и других сумм
            log_cols = ["target", "cum_sum"] + [col for col in inn.columns
                                                if col.startswith('trns_amount_')]
            for col in log_cols:
                if col in inn.columns:
                    inn = inn.with_columns(np.log1p(pl.col(col)))

        # Сортируем
        inn = inn.sort(by=["inn_id", "week"])

        if add_agg_data:
            # Это целевая переменная = 'target' --> нужно удалить
            if "trns_amount_out" in inn.columns:
                inn = inn.drop("trns_amount_out")

            drop_inn_columns = [col for col in inn.columns if col not in ('inn_id', 'week',
                                                                          'target',)]
            print('drop_inn_columns:', drop_inn_columns)

            tst = self._make_agg(inn, delta=0, shift_start=shift_start,
                                 shift_total=shift_total, w_size=w_size)
            tst = tst.drop(drop_inn_columns)

            # Формируем данные для теста
            inn_last = tst.group_by("inn_id").last().drop(["week", "target"])
            inn_test = inn_test.join(inn_last, on="inn_id", how="left")

            trn = self._make_agg(inn, delta=1, shift_start=shift_start,
                                 shift_total=shift_total, w_size=w_size)
            trn = trn.drop(drop_inn_columns)

            del inn
            gc.collect()

            print('Новые признаки построены!')

            self.comment = dict(shift_start=shift_start,
                                shift_total=shift_total,
                                w_size=w_size)
        else:

            # Определение числовых колонок (включая все целочисленные и вещественные типы)
            numeric_cols = [col for col in inn.select(pl.selectors.numeric()).columns
                            if col not in ('inn_id', 'week', 'target',
                                           # 'cum_sum'
                                           )]
            # Применяем shift(1).over("inn_id") для всех числовых колонок
            inn = inn.with_columns(pl.col(numeric_cols).shift(1).over("inn_id")).fill_null(0)

            trn = inn

            # Формируем данные для теста
            inn_last = inn.group_by("inn_id").last().drop(["week", "target"])
            inn_test = inn_test.join(inn_last, on="inn_id", how="left")

            print('trn.columns', trn.columns)
            # print('inn_test.columns', inn_test.columns)
            # print(inn.shape, trn.shape)

        # Объединение с weeks и profiles
        trn = trn.join(weeks, on="week", how="left")
        print('Объединение с weeks!')
        trn = trn.join(profiles, on="inn_id", how="left")
        print('Объединение с profiles!')
        inn_test = inn_test.join(weeks, on="week", how="left").join(profiles, on="inn_id",
                                                                    how="left")
        print('Объединение с weeks и profiles!')

        if add_agg_data:
            # Удаляем временные колонки
            trn = trn.drop(['d1', 'd2', 'report_date'])
            inn_test = inn_test.drop(['d1', 'd2', 'report_date'])
            print('Удалены временные колонки!')
            # Из трейна удалим строки с пропусками:
            for col in ("target",):
                sf = f'{col}_'
                if col not in trn.columns:
                    continue
                shift_col = f"{sf}shift{shift_start + shift_total - 1:02d}"
                print(f'Удаляю пропуски в {shift_col}!', shift_col in trn.columns)
                chunk_size = 500_000  # Размер чанка
                filtered_chunks = []
                for chunk in tqdm(trn.iter_slices(n_rows=chunk_size),
                                  total=len(trn) / chunk_size):
                    filtered_chunks.append(chunk.filter(pl.col(shift_col).is_not_null()))
                trn = pl.concat(filtered_chunks)
                print(f'Удалены пропуски в {shift_col}!')
        else:
            print('trn.columns:', trn.columns)
            print('inn_test.columns:', inn_test.columns)

            # select_cols = ['inn_id', 'week', 'd1', 'target', 'workdays', 'weekends', 'ipul',
            #                'id_region', 'main_okved_group', 'diff_datopen_report_date_flg']
            # if 'q' in trn.columns:
            #     select_cols.insert(3, 'q')
            # trn = trn.select(select_cols + (['sum_in'] if 'sum_in' in trn.columns else []))
            # select_cols[select_cols.index('target')] = 'predict'
            # inn_test = inn_test.select(select_cols)
            #
            # print('trn.columns:', trn.columns)

        # Сортируем
        trn = trn.sort(by=["inn_id", "week"])

        # Уменьшаем типы данных
        trn = memory_compression_pl(trn)
        inn_test = memory_compression_pl(inn_test)

        # преобразуем в пандас датафрейм
        train_df = trn.to_pandas()
        test_df = inn_test.to_pandas()

        print_time(start_time)

        if self.aggregate_path_file:
            save_time = print_msg('Сохраняем агрегированные данные...')
            with open(aggregate_path_file, 'wb') as file:
                joblib.dump((train_df, test_df), file, compress=7)
            print_time(save_time)

        return train_df, test_df

    @staticmethod
    def drop_constant_columns(df):
        # Ищем колонки с константным значением для удаления
        col_to_drop = []
        for col in df.columns:
            if df[col].nunique() == 1:
                col_to_drop.append(col)
        if col_to_drop:
            df.drop(columns=col_to_drop, inplace=True)
        return df


def set_all_seeds(seed=RANDOM_SEED):
    # python's seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def add_info_to_log(prf, max_num, idx_fold, model, valid_scores, info_cols,
                    comment_dict=None, clf_lr=None, log_file=MODELS_LOGS):
    """
    Добавление информации об обучении модели
    :param prf: Префикс файла сабмита
    :param max_num: номер итерации обучения моделей
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param valid_scores: скоры при обучении
    :param info_cols: информативные колонки
    :param comment_dict: комментарии
    :param clf_lr: список из learning_rate моделей
    :param log_file: полный путь к файлу с логами обучения моделей
    :return:
    """
    m_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, score = valid_scores

    model_columns, exclude_columns, cat_columns = info_cols

    if comment_dict is None:
        comment = {}
    else:
        comment = deepcopy(comment_dict)

    model_clf_lr = feature_imp = None
    if model is not None:
        if 'CatBoost' in model.__class__.__name__:
            model_clf_lr = model.get_all_params().get('learning_rate', 0)
            feature_imp = model.feature_importances_

        elif 'LGBM' in model.__class__.__name__:
            model_clf_lr = model.get_params().get('learning_rate', 0)

        elif 'XGB' in model.__class__.__name__:
            model_clf_lr = model.get_params().get('learning_rate', 0)

    if feature_imp is not None:
        try:
            use_cols = [col for col in model_columns if col not in exclude_columns]
            features = pd.DataFrame({'Feature': use_cols,
                                     'Importance': feature_imp}).sort_values('Importance',
                                                                             ascending=False)
            features.to_excel(MODEL_PATH.joinpath(f'features_{prf}{max_num}.xlsx'),
                              index=False)
        except:
            pass

    if model_clf_lr is not None:
        model_clf_lr = round(model_clf_lr, 8)

    if clf_lr is None:
        clf_lr = model_clf_lr

    comment['clf_lr'] = clf_lr
    if model is not None:
        comment.update(model.get_params())

    prf = prf.strip('_')

    with open(log_file, mode='a') as log:
        # log.write('num;mdl;fold;mdl_score;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;'
        #           'f1_wght;tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        log.write(f'{max_num};{prf};{idx_fold};{m_score:.6f};{auc_macro:.6f};{auc_micro:.6f};'
                  f'{auc_wght:.6f};{f1_macro:.6f};{f1_micro:.6f};{f1_wght:.6f};{score:.6f};'
                  f'{model_columns};{exclude_columns};{cat_columns};{comment}\n')


def merge_submits(max_num: list, submit_prefix='cb_', post_fix=''):
    """
    Объединение сабмитов
    :param max_num: список номеров сабмитов или список файлов
    :param submit_prefix: префикс сабмита модели
    :param post_fix: постфикс для регрессии
    :return: None
    """
    submits = pd.DataFrame()
    num_submits = False
    if all(isinstance(z, int) for z in max_num):
        num_submits = True
    # Читаем каждый файл и добавляем его содержимое в список датафреймов
    for idx, num in enumerate(sorted(max_num)):
        if num_submits:
            submit_csv = f'{submit_prefix}submit_{num:03}{LOCAL_FILE}_reg.csv'
        else:
            submit_csv = num
        file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
        df = pd.read_csv(file_submit_csv)
        df.rename(columns={'predict': f'predict_{num}'}, inplace=True)
        if not idx:
            submits = df
        else:
            submits = submits.merge(df, on=['inn_id', 'week'], suffixes=('', f'_{idx}'))

    if num_submits:
        max_num = '-'.join(map(str, max_num))
    else:
        max_num = '-'.join(sorted(re.findall(r'\d{3,}(?:_\d)?', ' '.join(max_num)), key=int))

    # submits.to_excel(WORK_PATH.joinpath(f'{submit_prefix}submit_{max_num}{LOCAL_FILE}.xlsx'))

    # Нахождение среднего по строкам
    predict_columns = [col for col in submits.columns if col.startswith('predict_')]
    submits['predict'] = submits[predict_columns].mean(axis=1)
    # # Нахождение медианы по строкам
    # submits['target_reg'] = submits.median(axis=1)

    submits_csv = f'{submit_prefix}submit_{max_num}{LOCAL_FILE}_reg.csv'
    submits[['inn_id', 'week', 'predict']].to_csv(PREDICTIONS_DIR.joinpath(submits_csv),
                                                  index=False)
    return submits_csv


def make_predict_reg(idx_fold, model, datasets, max_num=0, submit_prefix='cb_',
                     log_target=True):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: roc_auc и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param log_target: Целевая переменная была логарифмирована
    :return: разные roc_auc и F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, model_columns = datasets

    features2drop = ['week', 'predict']

    test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

    # print('X_train.shape', X_train.shape)
    # print('train.shape', train.shape)
    # print('test.shape', test.shape)

    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''

    predict_valid = model.predict(X_valid)
    predict_valid = np.clip(predict_valid, 0, np.inf)
    predict_test = model.predict(test)
    predict_test = np.clip(predict_test, 0, np.inf)

    y_valid = y_valid.values

    print(y_valid[:5])

    if log_target:
        y_valid = np.expm1(np.array(y_valid, dtype=np.float64))
        predict_valid = np.expm1(predict_valid)
        predict_test = np.expm1(predict_test)

        y_valid = np.where(np.isinf(y_valid), 0, y_valid)
        predict_valid = np.where(np.isinf(predict_valid), 0, predict_valid)
        predict_test = np.where(np.isinf(predict_test), 0, predict_test)

    print(y_valid[:5])
    print(predict_valid[:5].round(2))
    print(predict_test[:5].round(2))

    # Сохранение предсказаний в файл
    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}{LOCAL_FILE}_reg.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    submission = test_df[['inn_id', 'week', 'predict']].copy()
    submission['predict'] = predict_test.flatten()
    submission.to_csv(file_submit_csv, index=False)

    t_score = 0

    # start_item = print_msg("Расчет scores...")
    # Root Mean Squared Error: квадратный корень среднеквадратичной ошибки
    # - среднее отклонение предсказанных значений от фактических значений без учета знака
    auc_macro = mean_squared_error(y_valid, predict_valid) ** 0.5
    # Mean Absolute Error: среднее абсолютное отклонение предсказанных значений от фактических
    auc_micro = mean_absolute_error(y_valid, predict_valid)
    # Mean Squared Error: средний квадрат отклонений предсказаний от фактических значений
    auc_wght = mean_squared_error(y_valid, predict_valid)
    # R² Score показывает, какую долю вариации зависимой переменной объясняет модель. Значение
    # R² варьируется от 0 до 1 (или может быть отрицательным, если модель плоха).
    # Значение 1 говорит о том, что модель идеально подходит к данным.
    f1_macro = r2_score(y_valid, predict_valid)
    # Mean Squared Logarithmic Error: измеряет среднюю разницу между логарифмами предсказанных
    # и фактических значений. Это полезно в случаях, когда хотим уменьшить влияние больших
    # ошибок, особенно при работе с экспоненциально растущими данными.
    f1_micro = mean_squared_log_error(y_valid, predict_valid)
    # Explained Variance Score: метрика показывает долю вариации целевой переменной, которую
    # может объяснить модель. Значение равно 1, если модель идеально объясняет данные, и
    # 0, если модель не способна предсказать значение лучше, чем просто использовать среднее
    # значение целевой переменной.
    f1_wght = explained_variance_score(y_valid, predict_valid)
    # print_time(start_item)

    try:
        if 'CatBoost' in model.__class__.__name__:
            eval_metric = model.get_params()['eval_metric']
            model_score = model.best_score_['validation'][eval_metric]
        elif 'LGBM' in model.__class__.__name__:
            model_score = model.best_score_['valid_0']['rmse']
        elif 'XGB' in model.__class__.__name__:
            model_score = model.best_score
        else:
            model_score = auc_macro
    except:
        model_score = 0

    return model_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, t_score


if __name__ == "__main__":
    border_count = 254  # для кетбуста на ГПУ

    # Чтение и предобработка данных
    data_cls = DataTransform(use_catboost=True,
                             category_columns=[],
                             drop_first=False,
                             # samples_paths=300,
                             )

    # data_cls.preprocess_files = None
    # data_cls.aggregate_path_file = None
    # data_cls.aggregate_inn_trn_file = None

    sample = None

    log_target = True

    # calendar, train, profiles, test = data_cls.preprocess_data(remake_file=False,
    #                                                            use_sum_in_file=True,
    #                                                            use_cls_trn=False,
    #                                                            use_cls_vect=True,
    #                                                            sample=sample,
    #                                                            log_target=log_target,
    #                                                            return_pandas=True)
    # inns = train.inn_id.unique()[:10]
    # print(train)
    # print(train.columns)
    # print([col for col in train.columns if col.startswith('cls_')])
    # exit()

    train, test = data_cls.make_agg_data(remake_file=True,
                                         use_sum_in_file=True,
                                         use_cls_trn=False,
                                         use_cls_vect=False,
                                         add_agg_data=True,
                                         shift_total=24,
                                         w_size=48,
                                         log_target=log_target,
                                         # sample=300,
                                         )
    if "cum_sum" in train.columns:
        inns = train[train["cum_sum"] > 0.01]["inn_id"].unique()[:10]
    else:
        inns = train["inn_id"].unique()[:10]
    if 'd1' in train.columns:
        print(train[['inn_id', 'week', 'd1', 'q', 'target']])
    else:
        print(train[['inn_id', 'week', 'q', 'target']])

    train[train.inn_id.isin(inns)].to_excel(WORK_PATH.joinpath('train_sample.xlsx'),
                                            index=False)
    test[test.inn_id.isin(inns)].to_excel(WORK_PATH.joinpath('test_sample.xlsx'),
                                          index=False)
    print(train.columns)
    print(test.columns)

    # print(merge_submits([62, 63, 65]))
