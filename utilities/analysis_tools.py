import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from bisect import bisect, bisect_right, bisect_left
from functools import reduce
from scipy.sparse import coo_matrix

N_FOR_COURSES = 5
N_FOR_ADS = 5
N_FOR_SOQ = 5

def load_google_trends_file(filename, names_have_quotations=False):
    trend_df_dict = pickle.load(open(filename, 'rb'))
    trends = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), list(trend_df_dict.values()))
    trends.index = pd.to_datetime(trends.index, dayfirst=True)
    col_name_list = trends.columns.values
    for col_name in col_name_list:
        current_col_name = col_name
        if ' ' in current_col_name:
            old_col_name = current_col_name
            current_col_name = current_col_name.replace(' ', '-')
            trends[current_col_name] = trends[old_col_name]
            trends = trends.drop(old_col_name, axis=1)
            
        if names_have_quotations:
            old_col_name = current_col_name
            current_col_name = current_col_name.replace('"', '')
            trends[current_col_name] = trends[old_col_name]
            trends = trends.drop(old_col_name, axis=1)
            
        if current_col_name == 'date':
            continue
        if trends[current_col_name].dtype == 'int' or trends[current_col_name].dtype == 'float':
            continue
        trends[current_col_name] = trends[current_col_name].apply(lambda x: int(x) if '<' not in x else 0)
    return trends

def filter_df_by_date(df, min_date, max_date, date_field):
    return df.loc[(df[date_field] >= min_date) & (df[date_field] <= max_date)]

def add_value_at_beginning(df, min_date, tag):
    min_value = df['cumulative_value'].min()-1
    if min_value >= 0:
        df = df.append({'date': min_date, 'tag': tag, 'cumulative_value': min_value}, ignore_index=True).\
                                sort_values(by='cumulative_value')
    return df

def plot_trends_and_tags(trends_df, tag_ad_and_course_df,
                        y1_label = 'Google Trends', date_col_name='date'):
    tags_to_get = [x for x in trends_df.columns.values if x != date_col_name]
    earliest_date = trends_df[date_col_name].min()
    latest_date = trends_df[date_col_name].max()
    tag_first_dates = tag_ad_and_course_df.loc[tag_ad_and_course_df.TagName.apply(lambda x: x in tags_to_get)]
    #current_index = 0
    fig = plt.figure(figsize=(20,int(np.ceil((len(trends_df.columns)-1)/5))*5))
    for current_index in range(len(tags_to_get)):
        ax1 = fig.add_subplot(int(np.ceil((len(trends_df.columns)-1)/5)),5,current_index+1)
#         if current_index >= len(colour_map):
#             break
        #print(trends_df['date'])
        ax1.plot(trends_df[date_col_name], trends_df[tags_to_get[current_index]], 
                 color='blue', 
                label=tags_to_get[current_index])
        current_tag_row = tag_first_dates.loc[tag_first_dates.TagName == tags_to_get[current_index]]
        if 'ad_date' in tag_ad_and_course_df.columns.values:
            # The date of the first advert is marked by a black dotted line
            if not pd.isnull(current_tag_row['ad_date'].values[0]):
                ax1.axvline(x=current_tag_row['ad_date'].values[0], linestyle=':', 
                                            color='black')
        if 'nth_ad_date' in tag_ad_and_course_df.columns.values:
            # The date of the nth advert is marked by a black dashed and dotted line
            if not pd.isnull(current_tag_row['nth_ad_date'].values[0]):
                ax1.axvline(x=current_tag_row['nth_ad_date'].values[0], linestyle='-.', 
                                            color='black')
        if 'course_date' in tag_ad_and_course_df.columns.values:
            # The date of the first course is marked by a green dotted line.
            if not pd.isnull(current_tag_row['course_date'].values[0]):
                ax1.axvline(x=current_tag_row['course_date'].values[0], linestyle=':', 
                                                color='green')
        if 'nth_course_date' in tag_ad_and_course_df.columns.values:
            # The date of the first course is marked by a green dotted line.
            if not pd.isnull(current_tag_row['nth_course_date'].values[0]):
                ax1.axvline(x=current_tag_row['nth_course_date'].values[0], linestyle='-.', 
                                                color='green')
        if 'TagFirstUseDate' in tag_ad_and_course_df.columns.values:
            # The first use date of the tag is marked by a red dotted line.
            if not pd.isnull(current_tag_row['TagFirstUseDate'].values[0]):
                ax1.axvline(x=current_tag_row['TagFirstUseDate'].values[0], linestyle=':', 
                                        color='red')
        if 'TagNthUseDate' in tag_ad_and_course_df.columns.values:
            # The first use date of the tag is marked by a red dashed and dotted line.
            if not pd.isnull(current_tag_row['TagNthUseDate'].values[0]):
                ax1.axvline(x=current_tag_row['TagNthUseDate'].values[0], linestyle='-.', 
                                        color='red')

        if current_index % 5 == 0:
            ax1.set_ylabel(y1_label+' volume', fontsize=10)
    #ax1.set_yticks(np.arange(0, 101, 10))
        #ax1.set_xlabel('Time', fontsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_title(tags_to_get[current_index])
    #plt.legend(fontsize=20)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def load_stackoverflow_trends_with_google_trends(google_trends_df, so_tag_index, so_weeks, so_timeseries,
                                                normalise = False, date_col_name='date'):
    so_timeseries = np.array(so_timeseries.todense())
    tags_to_get = [(x, so_tag_index[x]) for x in google_trends_df.columns if x != date_col_name]
    earliest_date = google_trends_df[date_col_name].min()
    latest_date = google_trends_df[date_col_name].max()
    # Since the 0-th column of the so_timeseries matrix is before the first element of so_weeks, 
    # we need to add 1 to each of the indices to find the column indices for that matrix.
    start_index = bisect_left(so_weeks, earliest_date) + 1 + 1
    end_index = bisect(so_weeks, latest_date) - 1 + 1
    
    result_df = google_trends_df.copy()
    max_value = 0
    for tag_info in tags_to_get:
        current_col = np.array(so_timeseries[tag_info[1], start_index:end_index]).flatten()
        if np.max(current_col) > max_value:
            max_value = np.max(current_col)
        result_df['so_'+tag_info[0]] = pd.Series(current_col)
    
    if normalise:
        for tag_info in tags_to_get:
            result_df['so_'+tag_info[0]] = result_df['so_'+tag_info[0]]*100.0 / max_value
    return result_df, max_value

def load_stack_overflow_trends(tags_to_get, so_tag_index, so_weeks, so_timeseries,
                                                normalise = False, starting_date=None, step_size=1):
    so_timeseries = np.array(so_timeseries.todense())
    tags_to_get = [(x, so_tag_index[x]) for x in tags_to_get]
    earliest_date = so_weeks[1]
    latest_date = so_weeks[-1]
    result_dict = dict()
    index_list=pd.to_datetime(pd.Series([so_weeks[i] for i in range(step_size, len(so_weeks)) 
                                           if i % step_size == 0]))
    result_dict['date'] = index_list
    
    max_value = 0
    for tag_info in tags_to_get:
        current_col = np.array(so_timeseries[tag_info[1], 2:]).flatten()
        if step_size > 1:
            current_col = np.array([sum(current_col[i:i+step_size]) for i in range(len(so_weeks)) 
                                    if i % step_size == 0])
        if np.max(current_col) > max_value:
            max_value = np.max(current_col)
        result_dict[tag_info[0]] = pd.Series(current_col)
        
    result_df = pd.DataFrame.from_dict(result_dict, orient='columns')
    if starting_date is not None:
        result_df = result_df.loc[result_df['date'] > starting_date]
    return result_df.set_index('date')

def drop_google_trends_cols(combined_trends_df, date_col_name='date'):
    columns_to_drop = [x for x in combined_trends_df.columns if x != date_col_name and 'so_' not in x]
    result_df = combined_trends_df.copy().drop(columns_to_drop, axis=1)
    result_df = result_df.rename(columns={x: x.split('_')[1] for x in result_df.columns if x != date_col_name})
    return result_df

def sort_adoption_sequence(data_dict):
    data_dict = {k:np.datetime64(data_dict[k]) for k in data_dict if data_dict[k] is not pd.NaT}
    keys = list(data_dict.keys())
    values = np.array([data_dict[x] for x in keys])
    sorting_indices = np.argsort(values)
    keys_ordered = [keys[i] for i in sorting_indices]
    return '-'.join(keys_ordered)

def calculate_full_individual_adoption_sequence(tag_early_appearances, 
                                so_votes_date=pd.NaT, google_trends_date=pd.NaT):
    data_dict = {'SQ1': tag_early_appearances['TagFirstUseDate'],
                 'SQ'+str(N_FOR_SOQ): tag_early_appearances['TagNthUseDate'],
                 'A1': tag_early_appearances['ad_date'],
                 'C1': tag_early_appearances['course_date'],
                 'A'+str(N_FOR_ADS): tag_early_appearances['nth_ad_date'],
                 'C'+str(N_FOR_COURSES): tag_early_appearances['nth_course_date'],
                 'SV': so_votes_date,
                 'GT': google_trends_date}
    return sort_adoption_sequence(data_dict)

def calculate_short_adoption_sequence(tag_early_appearances, 
                                so_votes_date=pd.NaT, google_trends_date=pd.NaT):
    data_dict = {'SQ': tag_early_appearances['TagFirstUseDate'],
                 'A': tag_early_appearances['ad_date'],
                 'C': tag_early_appearances['course_date'],
                 'SV': so_votes_date,
                 'GT': google_trends_date}
    return sort_adoption_sequence(data_dict)

#def to_days()

def populate_delays_dict(delays_dict, data_dict):
    data_dict = {k:np.datetime64(data_dict[k]) for k in data_dict if data_dict[k] is not pd.NaT}
    delays_dict = {k: (data_dict[k.split('_')[1]] - data_dict[k.split('_')[0]]).\
                               astype('timedelta64[D]') / np.timedelta64(1, 'D')
                   for k in delays_dict 
                   if k.split('_')[0] in data_dict and k.split('_')[1] in data_dict}
    return delays_dict

def calculate_inter_and_intra_platform_delays(tag_early_appearances, 
                                so_votes_date=pd.NaT, google_trends_date=pd.NaT):
    data_dict = {'SQ1': tag_early_appearances['TagFirstUseDate'],
                 'SQ'+str(N_FOR_SOQ): tag_early_appearances['TagNthUseDate'],
                 'A1': tag_early_appearances['ad_date'],
                 'C1': tag_early_appearances['course_date'],
                 'A'+str(N_FOR_ADS): tag_early_appearances['nth_ad_date'],
                 'C'+str(N_FOR_COURSES): tag_early_appearances['nth_course_date'],
                 'SV': so_votes_date,
                 'GT': google_trends_date}
    
    return_dict = {'SQ1_SQ'+str(N_FOR_SOQ): None,
                   'SQ1_SV': None,
                   'SQ1_A1': None,
                   'A1_A'+str(N_FOR_ADS): None,
                   'SQ1_C'+str(N_FOR_COURSES): None,
                   'C1_C'+str(N_FOR_COURSES): None,
                   'SV_A1': None,
                   'GT_A1': None,
                   'SV_C1': None,
                   'GT_C1': None,
                   'A1_C1': None}
    
    return populate_delays_dict(return_dict, data_dict)

def calculate_stackoverflow_votes_date(tag, stackoverflow_votes_df, threshold_rel=0.1, threshold_abs=5):
    # calculates the first time when the time series reached 10% of its median
    sub_df = stackoverflow_votes_df.loc[stackoverflow_votes_df[tag] > 0, tag]
    if sub_df.shape[0] == 0:
        return pd.NaT
    first_nonzero = sub_df.head(1).index.values[0]
    median_value = stackoverflow_votes_df.loc[stackoverflow_votes_df.index >= first_nonzero, tag].median()
    above_threshold = stackoverflow_votes_df.loc[
                        stackoverflow_votes_df[tag] >= max([median_value*threshold_rel, threshold_abs]), 
                                                            tag]
    if above_threshold.shape[0] == 0:
        return pd.NaT
    first_date_above_threshold = above_threshold.head(1).index.values[0]
    return first_date_above_threshold

def calculate_all_adoption_sequences(all_tags_early_appearances, stackoverflow_votes_df=None, google_trends_df=None):
    full_adoption_sequences = all_tags_early_appearances.apply(lambda x: 
             calculate_full_individual_adoption_sequence(x, 
                         so_votes_date=calculate_stackoverflow_votes_date(x['TagName'], stackoverflow_votes_df)), 
                                                               axis=1)
    
    short_adoption_sequences = all_tags_early_appearances.apply(lambda x: 
                             calculate_short_adoption_sequence(x, 
                    so_votes_date=calculate_stackoverflow_votes_date(x['TagName'], stackoverflow_votes_df)), axis=1)
    
    adoption_delays = all_tags_early_appearances.apply(lambda x: 
             calculate_inter_and_intra_platform_delays(x, 
                         so_votes_date=calculate_stackoverflow_votes_date(x['TagName'], stackoverflow_votes_df)), 
                                                               axis=1).apply(pd.Series)
    
    adoption_df = pd.concat([all_tags_early_appearances['TagName'], full_adoption_sequences, 
                             short_adoption_sequences, adoption_delays], axis=1).\
                             rename(columns={0:'full_adoption_seq', 1:'short_adoption_seq'})
    return adoption_df

def get_proportion_of_starting_element(df, seq_col, starting_elem):
    return df.loc[df[seq_col].apply(lambda x: x[0] == starting_elem)].TagName.count() / df.TagName.count()

def get_all_proportions(df, seq_col, starting_elems=['S','A','C']):
    return tuple([get_proportion_of_starting_element(df, seq_col, starting_elem) for starting_elem in starting_elems] + [df.shape[0]]) 

def calculate_ordering_proportion(df, seq_col, first_elem, second_elem):
    return (df.loc[df[seq_col].apply(lambda x: x.index(first_elem) != -1 and x.index(second_elem) != -1 and
        x.index(first_elem) < x.index(second_elem))].TagName.sum() / \
        df.loc[df[seq_col].apply(lambda x: x.index(first_elem) != -1 and x.index(second_elem) != -1)].TagName.sum(),
        df.loc[df[seq_col].apply(lambda x: x.index(first_elem) != -1 and x.index(second_elem) != -1)].TagName.sum())


def get_topic_and_type(combined_tag_type):
    result = dict()
    if 'deprecated' in combined_tag_type or 'company' in combined_tag_type or 'person' in combined_tag_type:
        result['topic'] = None
        result['type'] = None
    else:
        split_by_dashes = combined_tag_type.split('-')
        if 'language' in combined_tag_type:
            if split_by_dashes[0] == 'language':
                result['topic'] = 'generalpurpose'
                result['type'] = 'language'
            else:
                result['topic'] = split_by_dashes[0]
                result['type'] = 'language'
        else:
            if len(split_by_dashes) == 1:
                result['topic'] = split_by_dashes[0]
                result['type'] = None
            elif split_by_dashes[0] == 'lang':
                result['topic'] = split_by_dashes[1]
                result['type'] = split_by_dashes[-1]
            else:
                result['topic'] = split_by_dashes[0]
                result['type'] = split_by_dashes[-1]
    return result

def get_subset_of_tags(df, tags_list):
    return df.loc[df.TagName.apply(lambda x: x in tags_list)]
