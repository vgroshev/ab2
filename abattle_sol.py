#!/usr/bin/env python
# coding: utf-8

"""
#  Alfa Battle 2.0, дек.2020 - янв 2021 #
"""

# __Импорт библиотек и определение вспомогательных функций__

import pandas as pd
import numpy as np


def process_one_group(dfgrp):
    """Обработка данных по одному клиенту"""
    m_pop = dfgrp['multi_class_target'].value_counts()
    m_pop_class = m_pop.index[0]
    m_pop_qty = m_pop[0]

    min_tstamp = dfgrp['timestamp'].min()

    max_tstamp = dfgrp['timestamp'].max()
    max_act = grp[grp['timestamp'] == max_tstamp]['multi_class_target'].values[0]

    return [m_pop_class, m_pop_qty, min_tstamp, max_tstamp, max_act]


# основная функция для расчета статистики и вычисления значения, выставляемого в качестве решения
def get_weighted_pop(onegrp, ts_check):
    """По данным для одного клиента находим взвешенное популярное:
    находим вес отдельного действия как обратное к дельте по времени до точки прогноза,
    суммируем веса наблюдений одного и того же действия"""
    stat_weights = []
    ts_list = onegrp['timestamp'].values
    act_list = onegrp['multi_class_target'].values
    for t in ts_list:
        tdelta = round((ts_check - t)/np.timedelta64(1, 's'))
        stat_weights.append(1/tdelta)
    top_position = pd.DataFrame({'act':act_list, 'w': stat_weights})\
            .groupby('act')\
            .sum()\
            .sort_values(by=['w'], ascending=False)\
            .index[0]
    return top_position


# __Импорт данных и обработка__


# Выборка для обучения

# client - Идентификатор клиента
# session_id - Идентификатор сессии
# timestamp - Время начала сессии
# target - Целевое действие внутри сессии, multi-class переменная
df = pd.read_csv('alfabattle2_abattle_train_target.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(df.shape)

# подсчет количества сессий для каждого из клиентов
df['client_pin'].value_counts()

# выборка для построения прогноза
# client - Идентификатор клиента
# timestamp - Время начала сессии
dfts = pd.read_csv('alfabattle2_prediction_session_timestamp.csv')
dfts['timestamp'] = pd.to_datetime(dfts['timestamp'])
print(dfts.shape)

# создаем группироовку по клиентам
dfg = df.groupby('client_pin')

# считаем разные статистики по каждой из групп
most_pop_class = {} # самое часто встречаемое действие
most_pop_qty = {}   # кол-во для самого частого
min_ts = {}  # самое ранний момент наблюдения
max_ts = {}  # самый последний момент наблюдения
max_act = {} # самое последнее действие
dfg_size = {} # размер группы (кол-во наблюдений для данного клиента)
dfg_size_list = dfg.size()

# проходим по каждой из групп, вычисляем
for k, grp in dfg:
    grp_get_stat = process_one_group(grp)
    most_pop_class[k] = grp_get_stat[0]
    most_pop_qty[k] = grp_get_stat[1]
    min_ts[k] = grp_get_stat[2]
    max_ts[k] = grp_get_stat[3]
    max_act[k] = grp_get_stat[4]
    dfg_size[k] = dfg_size_list[k]

# сохраняем словарь-результат в датафрейме
rez = pd.DataFrame({'client_pin': most_pop_class.keys(),
                    'group_size': dfg_size.values(),
                    'most_pop_class': most_pop_class.values(),
                    'most_pop_qty': most_pop_qty.values(),
                    'min_ts': min_ts.values(),
                    'max_ts': max_ts.values(),
                    'max_act': max_act.values()})

# объединяем таблицу со статистикой и таблицу с моментами времени для прогноза
rez1 = rez.join(dfts.set_index('client_pin'), on='client_pin')
print(rez1.shape)

# добавляем столбец с разницей во времени (в сек)
# между последним наблюдаемым событием и моментом прогноза
time_delta = rez1['timestamp'] - rez1['max_ts']
rez1['td2'] = (time_delta / np.timedelta64(1, 's')).astype(int)

# проходим по группам наблюдений, относящимся к одному и тому же клиенту, вычисляем, пишем в словарь
act_weighted = {}
for k, grp in dfg:
    ts_check = pd.to_datetime(rez1.loc[rez1['client_pin'] == k, 'timestamp']).values[0]
    act_weighted[k] = get_weighted_pop(grp, ts_check)

# вычисленные значения предсказания пишем в столбец объединенной таблицы
rez1['prediction'] = rez1['client_pin'].apply(lambda x: act_weighted[x])

# сохраняем для отправки решения
rez1[['client_pin', 'prediction']].to_csv('stat_table_20210105.csv', index=None)
