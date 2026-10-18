def match_values(a_dict, b_dict, impute=True):
    a_list = []
    b_list = []
    if impute:
        for series_id in b_dict:
            if series_id in a_dict:
                a_list.append(a_dict[series_id])
            else:
                a_list.append(1e7)
            b_list.append(b_dict[series_id])
    else:
        for series_id in b_dict:
            if series_id in a_dict:
                a_list.append(a_dict[series_id])
                b_list.append(b_dict[series_id])
    return a_list, b_list
