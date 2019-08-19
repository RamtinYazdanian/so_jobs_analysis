# coding=utf-8
import os
import errno

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST and exception.errno != errno.EPERM:
            raise

def invert_dict(d):
    return {d[k]: k for k in d}

def get_field(row, field_name):
    search_str = ' '+field_name+'=\"'
    start_point = row.find(search_str, 0)
    if start_point == -1:
        return None
    start_point += len(search_str)
    end_point = row.find('\"', start_point)
    return row[start_point:end_point].encode('utf8')