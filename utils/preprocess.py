# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/06/30 20:49


import numpy as np
import random

def gen_ind(fnum, ex_fnum=300, avg = True):
    if ex_fnum > fnum:
        cp_time = ex_fnum // fnum

        rest_num = ex_fnum - fnum * cp_time
        if rest_num > 0:
            cp_inter = fnum // rest_num

        f_inds = []

        for i in range(fnum):
            for _ in range(cp_time):
                f_inds.append(i)
                if rest_num > 0 and i % cp_inter == 0:
                    f_inds.append(i)
                    rest_num -= 1

    elif ex_fnum < fnum:
        f_inds = []
        rm_times = fnum // ex_fnum

        if avg:
            rand_int = 0
        else:
            rand_int = random.randint(0, rm_times - 1)

        for i in range(fnum):
            if i % rm_times == rand_int:
                f_inds.append(i)
            if (i + 1) % rm_times == 0 and not avg:
                rand_int = random.randint(0, rm_times - 1)

        n_fnum = len(f_inds)
        rest_num = n_fnum - ex_fnum

        if rest_num > 0:
            if avg:
                rm_inter = n_fnum // rest_num
                i = n_fnum - 1
                c = 0
                while i >= 0:
                    del f_inds[i]
                    # print(i)
                    i -= rm_inter
                    c += 1
                    if i < 0 or c == rest_num:
                        break
            else:
                de_inds = random.sample(list(range(n_fnum)), rest_num)
                de_inds = sorted(de_inds, reverse=True)
                for ind in de_inds:
                    del f_inds[ind]

    else:
        f_inds = list(range(fnum))

    return f_inds

def resize_frames(frames, ex_fnum = 30, avg = True):
    fnum = len(frames)
    a_inds = gen_ind(fnum=fnum,ex_fnum=ex_fnum, avg = avg)
    new_frames = [frames[ind] for ind in a_inds]
    if isinstance(frames, np.ndarray):
        new_frames = np.stack(new_frames, axis=0)
    return new_frames
