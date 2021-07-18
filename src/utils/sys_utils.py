import os
import re
import shutil
from datetime import datetime

def dir_name_ext(path):
    folder, base = os.path.split(path)
    if os.path.isdir(path):
        return folder, base,""
    name, ext = os.path.splitext(base)
    return folder, name, ext

def clear_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

def make_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def clear_path(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def one_up(s):
    pat = re.compile(r'(\d+)')
    match = re.match(pat, s[::-1])
    if not match:
        return s+'0'
    prev_num = match.group(1)[::-1]
    new_num = int(prev_num) + 1
    res = s[:(-match.end())] + str(new_num)
    return res

def get_avail_path(path):
    while os.path.exists(path):
        folder,name,ext = dir_name_ext(path)
        new_name = one_up(name)
        path = os.path.join(folder, new_name+ext)
    return path

def get_time_path():
    return datetime.now().strftime("%m-%d-%H-%M")

def path_to_linux(path):
    parts = re.split(r'\\', path)
    if len(parts) == 1: return path
    right_parts = ['/mnt']
    for p in parts:
        if p=='C:':
            p = 'c'

        right_parts.append(p)
    return r'/'.join(right_parts)

def path_to_windows(path):
    parts = re.split(r'/', path)
    if len(parts) == 1: return path
    right_parts = []
    for p in parts[2:]:
        if p=='c':
            p = 'C:'
        right_parts.append(p)
    return '\\'.join(right_parts)

def path_to_current_os(path):
    if os.name == 'nt':
        return path_to_windows(path)
    elif os.name == "posix":
        return path_to_linux(path)
    return path

def und_title(string):
    return ('_'+string+'_') if string else ""

def lund(string):
    return f'_{string}' if string else ""

def rund(string):
    return f'{string}_' if string else ""

def out_dir():
    cwd = os.getcwd()
    van_ind = cwd.rfind('VAN_ex')
    base_path = cwd[:van_ind+len('VAN_ex')]
    res_dir = os.path.join(base_path, 'out')
    return res_dir

def sort_dict_keys(dict):
    return sorted(dict.keys())
