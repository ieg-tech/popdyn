"""
Supplementary module to solve Chronic Wasting Disease simulations

Devin Cairns, 2019
"""
import csv
from collections import defaultdict

class CWDError(Exception):
    pass


def rec_dd():
    """Recursively update defaultdicts to avoid key errors"""
    return defaultdict(rec_dd)


def read_input(f):
    """
    Read a .csv with a matrix of direct transmission correlations

    :param str f: Input file path
    :return: dict of age group-sex direct transmission correlations
    """
    with open(f) as csvfile:
        data = [row for row in csv.reader(csvfile)]

    # Search for the direct transmission data
    cols = None
    for rows, row in enumerate(data):
        try:
            cols = [_j for _j, e in enumerate(row) if e == 'FROM'][0]
            break
        except IndexError:
            pass

    if cols is None:
        raise CWDError('Unable to parse Direct Transmission data in the CWD input spreadsheet')

    group_row = rows + 1
    sex_row = rows + 2
    rows += 3

    group_col = cols - 2
    sex_col = cols - 1

    # Parse direct transmission data
    direct_transmission = rec_dd()
    for j in range(cols, len(data[group_row])):
        from_gp = data[group_row][j].lower()
        if len(from_gp) == 0:
            break
        from_sex = data[sex_row][j].lower()
        for i in range(rows, len(data)):
            to_gp = data[i][group_col].lower()
            if len(to_gp) == 0:
                break
            to_sex = data[i][sex_col].lower()
            try:
                value = float(data[i][j])
            except ValueError:
                raise CWDError('Unable to read the value from {}-{} to {}-{}'.format(from_gp, from_sex, to_gp, to_sex))
            direct_transmission[from_gp][from_sex][to_gp][to_sex] = value

    # Search for environmental transmission data
    cols = None
    for rows, row in enumerate(data):
        try:
            cols = [_j for _j, e in enumerate(row) if e == 'C'][0]
            break
        except IndexError:
            pass

    if cols is None:
        raise CWDError('Unable to parse Env. Transmission data in the CWD input spreadsheet')

    group_row = rows - 2
    sex_row = rows - 1
    cols += 1

    # Parse environmental transmission data
    C = defaultdict(dict)
    for j in range(cols, len(data[group_row])):
        gp = data[group_row][j].lower()
        if len(gp) == 0:
            break
        sex = data[sex_row][j].lower()
        try:
            value = float(data[rows][j])
        except ValueError:
            raise CWDError('Unable to read the env. transmission value for {}-{}'.format(gp, sex))
        C[gp][sex] = value

    # Change into a regular dictionary
    def update(key, val, d):
        if isinstance(val, defaultdict):
            d[key] = {}
            for _key, _val in val.items():
                update(_key, _val, d[key])
        else:
            d[key] = val

    out_direct_transmission = {}
    for key, val in direct_transmission.items():
        update(key, val, out_direct_transmission)

    out_C = {}
    for key, val in C.items():
        update(key, val, out_C)

    return out_direct_transmission, out_C
