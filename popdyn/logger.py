"""
Populations dynamics logging tools

Devin Cairns 2018
"""

import time as profile_time
import string
import os
import numpy as np
import xlsxwriter
import summary
from string import punctuation
from datetime import datetime
from dateutil.tz import tzlocal


class LoggerError(Exception):
    pass


def time_this(func):
    """Decorator for profiling methods in classes"""
    def inner(*args, **kwargs):
        instance = args[0]
        start = profile_time.time()
        execution = func(*args, **kwargs)
        if hasattr(instance, 'D'):
            try:
                instance.D.profiler[func.__name__] += profile_time.time() - start
            except KeyError:
                instance.D.profiler[func.__name__] = profile_time.time() - start
            return execution
        else:
            try:
                instance.profiler[func.__name__] += profile_time.time() - start
            except KeyError:
                instance.profiler[func.__name__] = profile_time.time() - start
            return execution
    return inner


def name_key(name):
    """Map a given name to a stripped alphanumeric hash"""
    # Remove white space and make lower-case
    name = name.strip().replace(' ', '').lower()

    try:
        # String
        return name.translate(None, punctuation)
    except:
        # Unicode
        return name.translate(dict.fromkeys(punctuation))


def visualize(domain, output_file, time=None):
    """Visualize this domain tree"""
    try:
        import graphviz as gv
    except ImportError:
        raise LoggerError('Cannot visualize the domain because graphviz is not installed')

    def build_key(one_up, key):
        if one_up == 'Domain':
            return key
        else:
            return '{}\n{}'.format(one_up, key)

    def new_node(d, one_up):
        for key, val in d.items():
            if isinstance(val, dict):
                node_key = build_key(one_up, key)
                new_node(val, node_key)
            else:
                key = val.group_key
            if key is None:
                continue
            node_key = build_key(one_up, key)
            graph.node(node_key)
            graph.edge(one_up, node_key)

    graph = gv.Graph(format='png')
    new_node(domain.species, 'Domain')

    # Add mortality, carrying capacity and population if time is specified
    if time is not None:
        raise NotImplementedError('Have yet to complete this visualization tool')

    # Save the output graph
    if output_file.split('.')[-1].lower() == 'png':
        output_file = output_file[:-4]
    graph.render(filename=output_file)

# Pasted from simulation:
# self.formalLog = {'Parameterization': {'Domain size': str(self.shape),
#                                            'Cell size (x)': self.csx,
#                                            'Cell size (y)': self.csy,
#                                            'Start time': self.start_time,
#                                            'Time step': self.time_step,
#                                            'Age groups': str(zip(self.ages, self.durations))},
#                       'Habitat': {},
#                       'Population': {},
#                       'Natality': {},
#                       'Mortality': {},
#                       'Time': [],
#                       'Solver': [datetime.now(tzlocal()).strftime('%A, %B %d, %Y %I:%M%p %Z')]}


# def log_item(self, message, period):
#     message = (message[:250] + '...') if len(message) > 250 else message  # truncate really long messages
#     ScenarioLog.objects.create(
#         scenario_run=self.scenario_run,
#         message=message,
#         period=period)

def write_xlsx(domain, output_directory):
    """
    Write summarized information from a domain into an excel worksheet
    :param domain: Domain instance
    :param output_directory: Path to a directory that houses a new directory that the .xlsx files are saved to
    :return: Output directory
    """
    # Create the output directory based on time
    if not os.path.isdir(output_directory):
        raise LoggerError('The output directory must be a directory')

    strftime = datetime.now(tzlocal()).strftime('%d%B%Y_%I%M%p')

    path = os.path.join(output_directory, 'popdyn_{}'.format(strftime))

    # In case two model summaries occur within the same minute...
    second = 0
    while os.path.isdir(path):
        second += 1
        path = path[:-2] + '0{}'.format(second) + path[-2:]

    os.mkdir(path)

    # Collect the summarize information from the domain
    ms = summary.ModelSummary(domain)
    ms.model_summary()
    domain_dict = ms.summary

    def wb_write(row, col, val):
        try:
            val = float(val)
            if np.isinf(val) or np.isnan(val):
                val = ''
            if val == 0:
                tb.write(row, col, val, grey)
            else:
                tb.write(row, col, val, normal_num)
        except ValueError:
            tb.write(row, col, val)

    def index_to_char(index):
        if index < 26:
            return string.ascii_uppercase[index]
        else:
            return (string.ascii_uppercase[(index - 26) / 26] +
                    string.ascii_uppercase[(index - 26) % 26])

    for species_key, file_dict in domain_dict.items():

        species_name = list(file_dict['Population'].keys())[0]

        species_path = os.path.join(path, '{}_{}.xlsx'.format(species_key, strftime))

        wb = xlsxwriter.Workbook(species_path)
        bold = wb.add_format({'bold': True})
        grey = wb.add_format()
        grey.set_pattern(1)  # This is optional when using a solid fill.
        grey.set_bg_color('#D3D3D3')
        grey.set_num_format('#,##0.0000')
        normal_num = wb.add_format()
        normal_num.set_num_format('#,##0.0000')

        male_age_groups = file_dict['Parameterization']['Age Groups']['male']
        female_age_groups = file_dict['Parameterization']['Age Groups']['female']

        col_width = 0
        col_count = None

        chartRowIndex = {}
        for tab_key, species_dict in file_dict.items():
            if tab_key == 'Time':
                continue
            tb = wb.add_worksheet(tab_key)
            if tab_key == 'Parameterization':
                _row = -1
                col_count = len(species_dict)
                for col_1, col_2 in species_dict.items():
                    _row += 1
                    tb.write(_row, 0, col_1, bold)
                    tb.write(_row, 1, str(col_2))
                tb.set_column(0, 0, max(len(str(key)) for key in species_dict.keys()))
                tb.set_column(1, 1, max(len(str(key)) for key in species_dict.values()))
                continue
            elif tab_key == 'Habitat':
                tb.write(0, 0, 'Species', bold)
                tb.write(0, 1, 'Carrying capacity', bold)
                for col, _time in enumerate(map(str, file_dict['Time'])):
                    tb.write(0, col + 2, float(_time), bold)
                _row = 0
                col_count = len(species_dict)
                for species, cc_dict in species_dict.items():
                    for ds_type, itemList in cc_dict.items():
                        _row += 1
                        tb.write(_row, 0, species)
                        tb.write(_row, 1, ds_type)
                        for _col, item in enumerate(itemList):
                            wb_write(_row, _col + 2, str(item))
                tb.set_column(0, 0, max([len(sp) for sp in species_dict.keys()] + [len('Species')]))
                tb.set_column(1, 1, len('Carrying Capacity'))
                tb.set_column('{}:{}'.format(index_to_char(3), index_to_char(col_count + 3)), col_width)
                continue
            elif tab_key == 'Solver':
                tb.write(0, 0, 'Run Date', bold)
                tb.write(0, 1, species_dict[0])
                tb.write(2, 0, 'Function', bold)
                tb.write(2, 1, 'Cumulative time (seconds)', bold)
                _row = 2
                col_count = len(species_dict)
                for col_1 in species_dict[1:]:
                    _row += 1
                    items = col_1.split(',')
                    tb.write(_row, 0, items[0])
                    wb_write(_row, 1, items[1])
                tb.set_column(0, 0, max([len(c.split(',')[0]) for c in species_dict[1:]] + [len('Function')]))
                tb.set_column(1, 1, max([len(c.split(',')[1]) for c in species_dict[1:]] + [len(species_dict[0])]))
                continue
            tb.write(0, 0, 'Species', bold)
            tb.write(0, 1, 'Group', bold)
            tb.write(0, 2, 'Item', bold)
            for col, _time in enumerate(map(str, file_dict['Time'])):
                tb.write(0, col + 3, float(_time), bold)
            row = 0
            spLen, gpLen, nameLen = len('Species'), len('Group'), len('Item')
            for species, group_dict in species_dict.items():
                if len(species) > spLen:
                    spLen = len(species)
                for group, name_dict in group_dict.items():
                    if len(group) > gpLen:
                        gpLen = len(group)
                    for name, itemList in name_dict.items():
                        if len(name) > nameLen:
                            nameLen = len(name)
                        row += 1
                        if group != 'NA':
                            chartName = '{} {}'.format(group, name)
                        else:
                            chartName = name
                        try:
                            chartRowIndex[species][chartName] = (row + 1, len(itemList) + 2)
                        except KeyError:
                            chartRowIndex[species] = {chartName: (row + 1, len(itemList) + 2)}
                        tb.write(row, 0, str(species))
                        tb.write(row, 1, str(group))
                        tb.write(row, 2, str(name))
                        col_count = len(itemList)
                        for col, item in enumerate(itemList):
                            str_item = str(item)
                            wb_write(row, col + 3, str(str_item))
                            if len(str_item) > col_width:
                                col_width = len(str_item)
            tb.set_column(0, 0, spLen)
            tb.set_column(1, 1, gpLen)
            tb.set_column(2, 2, nameLen)
            tb.set_column('{}:{}'.format(index_to_char(3), index_to_char(col_count + 3)), col_width)

        try:
            # Add charts
            chartItems = []
            for species, chartRows in chartRowIndex.items():
                chartItems += [[('Total Population', '=Population!$D${}:${}${}'.format(
                                    chartRows['Total Population'][0],
                                    index_to_char(chartRows['Total Population'][1]),
                                    chartRows['Total Population'][0]),
                                    '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Population'][1])),
                                    'Population', '{} Total Population'.format(species)),
                               ('Total Females', '=Population!$D${}:${}${}'.format(
                                    chartRows['Total Females'][0],
                                    index_to_char(chartRows['Total Females'][1]),
                                    chartRows['Total Females'][0]),
                                    '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Females'][1])),
                                    'Population', '{} Total Population'.format(species)),
                               ('Total Males', '=Population!$D${}:${}${}'.format(
                                    chartRows['Total Males'][0],
                                    index_to_char(chartRows['Total Males'][1]),
                                    chartRows['Total Males'][0]),
                                    '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Males'][1])),
                                    'Population', '{} Total Population'.format(species))],
                              [('Total Population', '=Population!$D${}:${}${}'.format(
                                    chartRows['Total Population'][0],
                                    index_to_char(chartRows['Total Population'][1]),
                                    chartRows['Total Population'][0]),
                                    '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Population'][1])),
                                    'Population', '{} Total Population By Age Class'.format(species))] +
                               [('{} Total'.format(gp), '=Population!$D${}:${}${}'.format(
                                    chartRows['{} Total'.format(gp)][0],
                                    index_to_char(chartRows['{} Total'.format(gp)][1]),
                                    chartRows['{} Total'.format(gp)][0]),
                                 '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Population'][1])),
                                 'Population', '{} Total Population By Age Class'.format(species))
                                for gp in np.unique(male_age_groups + female_age_groups)],
                              [('Total Females', '=Population!$D${}:${}${}'.format(
                                    chartRows['Total Females'][0],
                                    index_to_char(chartRows['Total Females'][1]),
                                    chartRows['Total Females'][0]),
                                    '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Females'][1])),
                                    'Population', '{} Total Females By Age Class'.format(species))] +
                              [('{} Females'.format(gp), '=Population!$D${}:${}${}'.format(
                                  chartRows['{} Females'.format(gp)][0],
                                  index_to_char(chartRows['{} Females'.format(gp)][1]),
                                  chartRows['{} Females'.format(gp)][0]),
                                '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Population'][1])),
                                'Population', '{} Total Females By Age Class'.format(species))
                               for gp in female_age_groups],
                              [('Total Males', '=Population!$D${}:${}${}'.format(
                                    chartRows['Total Males'][0],
                                    index_to_char(chartRows['Total Males'][1]),
                                    chartRows['Total Males'][0]),
                                    '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Males'][1])),
                                    'Population', '{} Total Males By Age Class'.format(species))] +
                              [('{} Males'.format(gp), '=Population!$D${}:${}${}'.format(
                                  chartRows['{} Males'.format(gp)][0],
                                  index_to_char(chartRows['{} Males'.format(gp)][1]),
                                  chartRows['{} Males'.format(gp)][0]),
                                '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Population'][1])),
                                'Population', '{} Total Males By Age Class'.format(species)) for gp in male_age_groups],
                               [('Average Age (all)', '=Population!$D${}:${}${}'.format(
                                   chartRows['Average Age'][0],
                                   index_to_char(chartRows['Average Age'][1]),
                                   chartRows['Average Age'][0]),
                                 '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Average Age'][1])),
                                 'Average Age', '{} Average Age'.format(species)),
                                ('Average Female Age', '=Population!$D${}:${}${}'.format(
                                    chartRows['Average Female Age'][0],
                                    index_to_char(chartRows['Average Female Age'][1]),
                                    chartRows['Average Female Age'][0]),
                                 '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Average Female Age'][1])),
                                 'Average Age', '{} Average Age'.format(species)),
                                ('Average Male Age', '=Population!$D${}:${}${}'.format(
                                    chartRows['Average Male Age'][0],
                                    index_to_char(chartRows['Average Male Age'][1]),
                                    chartRows['Average Male Age'][0]),
                                 '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Average Male Age'][1])),
                                 'Average Age', '{} Average Age'.format(species))],
                              [('Total New Offspring', '=Natality!$D${}:${}${}'.format(
                                  chartRows['Total new offspring'][0],
                                  index_to_char(chartRows['Total new offspring'][1]),
                                  chartRows['Total new offspring'][0]),
                                '=Natality!$D$1:${}$1'.format(index_to_char(chartRows['Total new offspring'][1])),
                                'New Population', '{} New Offspring from each Female Age Class'.format(species))] +
                              [('{} New Offspring'.format(gp), '=Natality!$D${}:${}${}'.format(
                                  chartRows['{} Total offspring'.format(gp)][0],
                                  index_to_char(chartRows['{} Total offspring'.format(gp)][1]),
                                  chartRows['{} Total offspring'.format(gp)][0]),
                                '=Natality!$D$1:${}$1'.format(index_to_char(chartRows['Total new offspring'][1])),
                                'New Population', '{} New Offspring from each Female Age Class'.format(species))
                               for gp in np.unique(male_age_groups + female_age_groups)],
                              [('Total Deaths', '=Mortality!$D${}:${}${}'.format(
                                  chartRows['All deaths'][0],
                                  index_to_char(chartRows['All deaths'][1]),
                                  chartRows['All deaths'][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                'Deaths', '{} Total Mortality by Gender Class'.format(species)),
                               ('Total Female Deaths', '=Mortality!$D${}:${}${}'.format(
                                   chartRows['Total female deaths'][0],
                                   index_to_char(chartRows['Total female deaths'][1]),
                                   chartRows['Total female deaths'][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['Total female deaths'][1])),
                                'Deaths', '{} Total Mortality by Gender Class'.format(species)),
                               ('Total Male Deaths', '=Mortality!$D${}:${}${}'.format(
                                   chartRows['Total male deaths'][0],
                                   index_to_char(chartRows['Total male deaths'][1]),
                                   chartRows['Total male deaths'][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['Total male deaths'][1])),
                                'Deaths', '{} Total Mortality by Gender Class'.format(species))],
                              [('Total Deaths', '=Mortality!$D${}:${}${}'.format(
                                  chartRows['All deaths'][0],
                                  index_to_char(chartRows['All deaths'][1]),
                                  chartRows['All deaths'][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                'Deaths', '{} Total Mortality by Age Class'.format(species))] +
                              [('{} Total Deaths'.format(gp), '=Mortality!$D${}:${}${}'.format(
                                  chartRows['{} Total deaths'.format(gp)][0],
                                  index_to_char(chartRows['{} Total deaths'.format(gp)][1]),
                                  chartRows['{} Total deaths'.format(gp)][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                'Deaths', '{} Total Mortality by Age Class'.format(species))
                               for gp in np.unique(male_age_groups + female_age_groups)],
                              [('Total Deaths', '=Mortality!$D${}:${}${}'.format(
                                  chartRows['All deaths'][0],
                                  index_to_char(chartRows['All deaths'][1]),
                                  chartRows['All deaths'][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                'Deaths', '{} Total Male Mortality by Age Class'.format(species))] +
                              [('{} Male Deaths'.format(gp), '=Mortality!$D${}:${}${}'.format(
                                  chartRows['{} Male deaths'.format(gp)][0],
                                  index_to_char(chartRows['{} Male deaths'.format(gp)][1]),
                                  chartRows['{} Male deaths'.format(gp)][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                'Deaths', '{} Total Male Mortality by Age Class'.format(species))
                               for gp in male_age_groups],
                              [('Total Deaths', '=Mortality!$D${}:${}${}'.format(
                                  chartRows['All deaths'][0],
                                  index_to_char(chartRows['All deaths'][1]),
                                  chartRows['All deaths'][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                'Deaths', '{} Total Female Mortality by Age Class'.format(species))] +
                              [('{} Female Deaths'.format(gp), '=Mortality!$D${}:${}${}'.format(
                                  chartRows['{} Female deaths'.format(gp)][0],
                                  index_to_char(chartRows['{} Female deaths'.format(gp)][1]),
                                  chartRows['{} Female deaths'.format(gp)][0]),
                                '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                'Deaths', '{} Total Female Mortality by Age Class'.format(species))
                               for gp in female_age_groups],
                               [('Total Deaths', '=Mortality!$D${}:${}${}'.format(
                                   chartRows['All deaths'][0],
                                   index_to_char(chartRows['All deaths'][1]),
                                   chartRows['All deaths'][0]),
                                 '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                 'Deaths', '{} Total Mortality by Mortality Type (anthro, natural)'.format(species))] +
                               [('Total {} Deaths'.format(mm), '=Mortality!$D${}:${}${}'.format(
                                   chartRows['Total deaths from {}'.format(mm)][0],
                                   index_to_char(chartRows['Total deaths from {}'.format(mm)][1]),
                                   chartRows['Total deaths from {}'.format(mm)][0]),
                                 '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                 'Deaths', '{} Total Mortality by Mortality Type (anthro, natural)'.format(species))
                                for mm in summary.list_mortality_types(
                                   domain, name_key(species), None
                               ).tolist()],
                               [('Total Male Deaths', '=Mortality!$D${}:${}${}'.format(
                                   chartRows['Total male deaths'][0],
                                   index_to_char(chartRows['Total male deaths'][1]),
                                   chartRows['Total male deaths'][0]),
                                 '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                 'Deaths', '{} Total Male Mortality by Mortality Type'.format(species))] +
                               [('Male {} Deaths'.format(mm), '=Mortality!$D${}:${}${}'.format(
                                   chartRows['Male deaths from {}'.format(mm)][0],
                                   index_to_char(chartRows['Male deaths from {}'.format(mm)][1]),
                                   chartRows['Male deaths from {}'.format(mm)][0]),
                                 '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                 'Deaths', '{} Total Male Mortality by Mortality Type'.format(species))
                                for mm in summary.list_mortality_types(
                                   domain, name_key(species), None, 'male'
                               ).tolist()],
                               [('Total Female Deaths', '=Mortality!$D${}:${}${}'.format(
                                   chartRows['Total female deaths'][0],
                                   index_to_char(chartRows['Total female deaths'][1]),
                                   chartRows['Total female deaths'][0]),
                                 '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                 'Deaths', '{} Total Female Mortality by Mortality Type'.format(species))] +
                               [('Female {} Deaths'.format(mm), '=Mortality!$D${}:${}${}'.format(
                                   chartRows['Female deaths from {}'.format(mm)][0],
                                   index_to_char(chartRows['Female deaths from {}'.format(mm)][1]),
                                   chartRows['Female deaths from {}'.format(mm)][0]),
                                 '=Mortality!$D$1:${}$1'.format(index_to_char(chartRows['All deaths'][1])),
                                 'Deaths', '{} Total Female Mortality by Mortality Type'.format(species))\
                                for mm in summary.list_mortality_types(
                                   domain, name_key(species), None, 'female'
                               ).tolist()],
                               # Lambda
                               [('Total', '=Population!$D${}:${}${}'.format(
                                   chartRows['Total Population Lambda'][0],
                                   index_to_char(chartRows['Total Population Lambda'][1]),
                                   chartRows['Total Population Lambda'][0]),
                                 '=Population!$D$1:${}$1'.format(index_to_char(chartRows['Total Population Lambda'][1])),
                                 'Lambda (t+1/t)', '{} Population Lambda'.format(species))] +
                               [('{}'.format(gp), '=Population!$D${}:${}${}'.format(
                                   chartRows['{} Lambda'.format(gp)][0],
                                   index_to_char(chartRows['{} Lambda'.format(gp)][1]),
                                   chartRows['{} Lambda'.format(gp)][0]),
                                 '=Population!$D$1:${}$1'.format(index_to_char(chartRows['{} Lambda'.format(gp)][1])),
                                 'Lambda (t+1/t)', '{} Population Lambda'.format(species))
                                for gp in np.unique(male_age_groups + female_age_groups)],
                               # Sex Ratio
                               [('Total', '=Natality!$D${}:${}${}'.format(
                                   chartRows['F:M Ratio'][0],
                                   index_to_char(chartRows['F:M Ratio'][1]),
                                   chartRows['F:M Ratio'][0]),
                                 '=Natality!$D$1:${}$1'.format(
                                     index_to_char(chartRows['F:M Ratio'][1])),
                                 'Sex Ratio', '{} F:M Sex Ratio'.format(species))] +
                               [('{}'.format(gp), '=Natality!$D${}:${}${}'.format(
                                   chartRows['{} F:M Ratio'.format(gp)][0],
                                   index_to_char(chartRows['{} F:M Ratio'.format(gp)][1]),
                                   chartRows['{} F:M Ratio'.format(gp)][0]),
                                 '=Natality!$D$1:${}$1'.format(index_to_char(chartRows['{} F:M Ratio'.format(gp)][1])),
                                 'Sex Ratio', '{} F:M Sex Ratio'.format(species))
                                for gp in np.unique(male_age_groups + female_age_groups)],
                               # Fecundity
                               [('Total', '=Natality!$D${}:${}${}'.format(
                                   chartRows['Total offspring per female'][0],
                                   index_to_char(chartRows['Total offspring per female'][1]),
                                   chartRows['Total offspring per female'][0]),
                                 '=Natality!$D$1:${}$1'.format(
                                     index_to_char(chartRows['Total offspring per female'][1])),
                                 'Offspring/Female', '{} Number of offspring per female'.format(species))] +
                               [('{}'.format(gp), '=Natality!$D${}:${}${}'.format(
                                   chartRows['{} offspring per female'.format(gp)][0],
                                   index_to_char(chartRows['{} offspring per female'.format(gp)][1]),
                                   chartRows['{} offspring per female'.format(gp)][0]),
                                 '=Natality!$D$1:${}$1'.format(index_to_char(chartRows['{} offspring per female'.format(gp)][1])),
                                 'Offspring/Female', '{} Number of offspring per female'.format(species))
                                for gp in np.unique(male_age_groups + female_age_groups)]
                              ]

            positions = []
            for i in range(2, 3000, 23):
                for letter in ['B', 'N']:
                    positions.append('{}{}'.format(letter, i))
            tb = wb.add_worksheet('Charts')
            for i, items in enumerate(chartItems):
                chart = wb.add_chart({'type': 'scatter', 'subtype': 'smooth'})
                chart.set_size({'x_scale': 1.5, 'y_scale': 1.5})
                for name, values, cats, yAxis, title in items:
                    chart.add_series({'name': name, 'values': values, 'categories': cats,
                                      'marker': {'type': 'circle', 'size': 3}})
                chart.set_x_axis({'name': 'Time'})
                chart.set_y_axis({'name': yAxis})  # y-axis/title get the last value in __iter__
                chart.set_title({'name': title})
                tb.insert_chart(positions[i], chart)

        except Exception as e:
            tb = wb.add_worksheet('Charts')
            tb.write(0, 0, 'Error while creating charts:')
            tb.write(1, 0, str(e))
            i = 2
            for key, val in file_dict.items():
                if type(val) == dict:
                    for _key, _val in val.items():
                        i += 1
                        tb.write(i, 0, key)
                        tb.write(i, 1, _key)
                        tb.write(i, 2, str(file_dict[key]))

        # Hard-coded worksheet for AFW
        # --------------------------------
        afw = wb.add_worksheet('Harvest')

        # Styling
        black = wb.add_format({'bg_color': '#000000', 'font_color': 'white'})
        purple = wb.add_format({'bg_color': '#800080', 'font_color': 'white'})
        purple.set_num_format('#,##0.0')
        green = wb.add_format({'bg_color': '#98FB98', 'align': 'right'})
        green.set_num_format('#,##0.0')
        green_sr = wb.add_format({'bg_color': '#98FB98', 'align': 'right'})
        green_sr.set_num_format('#,##0.00')
        orange = wb.add_format({'bg_color': '#FFDEAD', 'align': 'right'})
        yellow = wb.add_format({'bg_color': '#FFFF00'})
        yellow.set_num_format('#,##0.0')
        brown = wb.add_format({'bg_color': '#CD853F', 'align': 'right'})
        brown.set_num_format('#,##0.0')
        bold_orange = wb.add_format({'bg_color': '#FFDEAD', 'bold': True})

        afw.write(0, 3, 'Computing "Harvest Permits" in Alces Online Population Dynamics', bold)

        # Add female age groups to males to be inclusive for the remainder of writing
        male_age_groups = list(np.unique(male_age_groups + female_age_groups))

        # Block 1
        afw.write(3, 3, 'Current Pre-Hunting Season Population Size and Composition', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Total', 'Sex Ratio']):
            afw.write(row + 4, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(4, col + 4, gp, black)
            # Collect the initial populations
            afw.write(5, col + 4, round(file_dict['Population'][species_name][gp]['Males'][0], 1), purple)
            afw.write(6, col + 4, round(file_dict['Population'][species_name][gp]['Females'][0], 1), purple)
            afw.write(7, col + 4, '={0}6+{0}7'.format(index_to_char(col + 4)), green)
            afw.write(8, col + 4, '={0}7/{0}6'.format(index_to_char(col + 4)), green_sr)
        afw.write(4, len(male_age_groups) + 4, 'Total', black)
        for row in range(5, 8):
            afw.write(row, len(male_age_groups) + 4, '=SUM({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)
        afw.write(8, len(male_age_groups) + 4, '={0}7/{0}6'.format(index_to_char(len(male_age_groups) + 4)), green_sr)

        # Block 2
        afw.write(10, 3, 'Simulated Annual Harvest Fraction (% of cohort) for Permitted Harvest', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 11, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(11, col + 4, gp, black)
            # Fraction of population
            try:
                if file_dict['Population'][species_name][gp]['Males'][0] > 0:
                    male_rate = file_dict['Mortality'][species_name][gp]['male License Harvest deaths'][1] / \
                                file_dict['Population'][species_name][gp]['Males'][0]
                else:
                    male_rate = 0
            except TypeError:
                male_rate = 0
            try:
                if file_dict['Population'][species_name][gp]['Females'][0] > 0:
                    female_rate = file_dict['Mortality'][species_name][gp]['female License Harvest deaths'][1] / \
                                  file_dict['Population'][species_name][gp]['Females'][0]
                else:
                    female_rate = 0
            except TypeError:
                female_rate = 0

            female_rate = round(female_rate, 1)
            male_rate = round(male_rate, 1)

            afw.write(12, col + 4, male_rate, purple)
            afw.write(13, col + 4, female_rate, purple)
            afw.write(14, col + 4, '=AVERAGE({0}13:{0}14)'.format(index_to_char(col + 4)), green)
        afw.write(11, len(male_age_groups) + 4, 'Average', black)
        for row in range(12, 15):
            afw.write(row, len(male_age_groups) + 4, '=SUMPRODUCT({0}{3}:{1}{3}, {0}{4}:{1}{4})/{2}{4}'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), index_to_char(len(male_age_groups) + 4),
                row + 1, row - 6
            ), green)

        # Block 3
        afw.write(16, 3, 'Simulated "Total" Permit Harvest Number', bold)
        afw.write(16, 6, 'Note: These values will include non-permitted mortality if not address in AO PD explicitly')
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Total']):
            afw.write(row + 17, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(17, col + 4, gp, black)
            afw.write(18, col + 4, '={0}6*{0}13'.format(index_to_char(col + 4)), green)
            afw.write(19, col + 4, '={0}7*{0}14'.format(index_to_char(col + 4)), green)
            afw.write(20, col + 4, '=SUM({0}19:{0}20)'.format(index_to_char(col + 4)), green)
        afw.write(17, len(male_age_groups) + 4, 'Total', black)
        afw.write(18, len(male_age_groups) + 4, '=SUM({}19:{}19)'.format(
            index_to_char(4), index_to_char(len(male_age_groups) + 3)
        ), green)
        afw.write(19, len(male_age_groups) + 4, '=SUM({}20:{}20)'.format(
            index_to_char(4), index_to_char(len(male_age_groups) + 3)
        ), green)
        afw.write(20, len(male_age_groups) + 4, '=SUM({}21:{}21)'.format(
            index_to_char(4), index_to_char(len(male_age_groups) + 3)
        ), green)

        # Block 4
        afw.write(22, 1, 'Factoring in "Non-Permitted" Mortality. Units are # animals killed/year', bold_orange)
        afw.write(22, 2, '', orange)
        afw.write(22, 3, '', orange)

        afw.write(23, 3, 'Gender/Stage', black)
        for col, gp in enumerate(male_age_groups + ['Total']):
            afw.write(23, col + 4, gp, black)

        categories = ['Expected Offtake from Outfitters', 'Expected Offtake from Indigenous Communities',
                      'Expected Offtake from Landowners', 'Expected Offtake from Archers',
                      'Expected Offtake from Vehicular Mortality', 'Expected Offtake from Wounding',
                      'Expected Offtake from Poachers', 'Total Non-Permit Offtake',
                      'Total Available for "Permit Harvest"']

        for row, category in enumerate(categories):
            cur_row = ((row + (row * 3)) + 24)

            if row == len(categories) - 2:
                # Second last group
                bg = brown
                afw.write(cur_row - 1, 1, '', orange)
                afw.write(cur_row - 1, 2, '', orange)
                for col in range(len(male_age_groups) + 1):
                    afw.write(cur_row, col + 4, '={}'.format(
                        '+'.join(['{}{}'.format(
                            index_to_char(col + 4), ((_row + (_row * 3)) + 24) + 1
                        ) for _row in range(len(categories) - 2)])
                    ), brown)
                    afw.write(cur_row + 1, col + 4, '={}'.format(
                        '+'.join(['{}{}'.format(
                            index_to_char(col + 4), ((_row + (_row * 3)) + 24) + 2
                        ) for _row in range(len(categories) - 2)])
                    ), brown)
                    afw.write(cur_row + 2, col + 4, '=SUM({0}{1}:{0}{2})'.format(
                        index_to_char(col + 4), cur_row + 1, cur_row + 2
                    ), brown)

            elif row == len(categories) - 1:
                # Last group
                bg = green
                for col in range(len(male_age_groups) + 1):
                    afw.write(cur_row, col + 4, '=MAX(0,{0}19-{0}{1})'.format(index_to_char(col + 4), cur_row - 3), green)
                    afw.write(cur_row + 1, col + 4, '=MAX(0,{0}20-{0}{1})'.format(index_to_char(col + 4), cur_row - 2), green)
                    afw.write(cur_row + 2, col + 4, '=MAX(0,{0}21-{0}{1})'.format(index_to_char(col + 4), cur_row - 1), green)

            else:
                bg = orange
                afw.write(cur_row - 1, 1, '', orange)
                afw.write(cur_row - 1, 2, '', orange)
                for col in range(len(male_age_groups) + 1):
                    afw.write(cur_row, col + 4, 0, yellow)
                    afw.write(cur_row + 1, col + 4, 0, yellow)
                    afw.write(cur_row + 2, col + 4, '=SUM({0}{1}:{0}{2})'.format(index_to_char(col + 4),
                                                                                 cur_row + 1, cur_row + 2), green)
                afw.write(cur_row, len(male_age_groups) + 4, '=SUM({0}{2}:{1}{2})'.format(
                    index_to_char(4), index_to_char(len(male_age_groups) + 3), cur_row + 1
                ), yellow)
                afw.write(cur_row + 1, len(male_age_groups) + 4, '=SUM({0}{2}:{1}{2})'.format(
                    index_to_char(4), index_to_char(len(male_age_groups) + 3), cur_row + 2
                ), yellow)
                afw.write(cur_row + 2, len(male_age_groups) + 4, '=SUM({0}{2}:{1}{2})'.format(
                    index_to_char(4), index_to_char(len(male_age_groups) + 3), cur_row + 3), green)

            afw.write(cur_row, 2, category, bg)
            afw.write(cur_row, 1, '', bg)
            afw.write(cur_row + 1, 1, '', bg)
            afw.write(cur_row + 1, 2, '', bg)
            afw.write(cur_row + 2, 1, '', bg)
            afw.write(cur_row + 2, 2, '', bg)
            afw.write(cur_row, 3, 'Males', black)
            afw.write(cur_row + 1, 3, 'Females', black)
            afw.write(cur_row + 2, 3, 'Total', black)

        # Block 5
        afw.write(60, 3, 'Average Quota Success (fraction) of Permit Hunter', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 61, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(61, col + 4, gp, black)
            afw.write(62, col + 4, 0, yellow)
            afw.write(63, col + 4, 0, yellow)
            afw.write(64, col + 4, '=AVERAGE({0}63:{0}64)'.format(index_to_char(col + 4)), green)
        afw.write(61, len(male_age_groups) + 4, 'Average', black)
        for row in range(62, 65):
            afw.write(row, len(male_age_groups) + 4, '=AVERAGE({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 6
        afw.write(66, 3, 'Number of Tags to Issue to Permit Hunters', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Total']):
            afw.write(row + 67, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(67, col + 4, gp, black)
            afw.write(68, col + 4, '={0}57/{0}63'.format(index_to_char(col + 4)), green)
            afw.write(69, col + 4, '={0}58/{0}64'.format(index_to_char(col + 4)), green)
            afw.write(70, col + 4, '=SUM({0}69:{0}70)'.format(index_to_char(col + 4)), green)
        afw.write(67, len(male_age_groups) + 4, 'Total', black)
        afw.write(68, len(male_age_groups) + 4, '={0}57/{0}63'.format(index_to_char(len(male_age_groups) + 4)), green)
        afw.write(69, len(male_age_groups) + 4, '={0}58/{0}64'.format(index_to_char(len(male_age_groups) + 4)), green)
        afw.write(70, len(male_age_groups) + 4, '={0}59/{0}65'.format(index_to_char(len(male_age_groups) + 4)), green)

        # Block 7
        afw.write(72, 3, 'Average Number of Permit Hunter Days Per Animal Harvested', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 73, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(73, col + 4, gp, black)
            afw.write(74, col + 4, 0, yellow)
            afw.write(75, col + 4, 0, yellow)
            afw.write(76, col + 4, '=AVERAGE({0}75:{0}76)'.format(index_to_char(col + 4)), green)
        afw.write(73, len(male_age_groups) + 4, 'Average', black)
        for row in range(74, 77):
            afw.write(row, len(male_age_groups) + 4, '=AVERAGE({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 8
        afw.write(78, 3, 'Total Number of "Permitted" Hunting Days', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Total']):
            afw.write(row + 79, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(79, col + 4, gp, black)
            afw.write(80, col + 4, '={0}69*{0}75'.format(index_to_char(col + 4)), green)
            afw.write(81, col + 4, '={0}70*{0}76'.format(index_to_char(col + 4)), green)
            afw.write(82, col + 4, '=SUM({0}81:{0}82)'.format(index_to_char(col + 4)), green)
        afw.write(79, len(male_age_groups) + 4, 'Total', black)
        afw.write(80, len(male_age_groups) + 4, '=SUM({0}81:{1}81)'.format(
            index_to_char(4), index_to_char(len(male_age_groups) + 3)), green
                  )
        afw.write(81, len(male_age_groups) + 4, '=SUM({0}82:{1}82)'.format(
            index_to_char(4), index_to_char(len(male_age_groups) + 3)), green
                  )
        afw.write(82, len(male_age_groups) + 4, '=SUM({0}83:{1}83)'.format(
            index_to_char(4), index_to_char(len(male_age_groups) + 3)), green
                  )

        # Block 9
        afw.write(90, 3, 'Morphometric-based inputs', bold)

        afw.write(92, 3, 'Liveweight (Kilograms)', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 93, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(93, col + 4, gp, black)
            afw.write(94, col + 4, 0, yellow)
            afw.write(95, col + 4, 0, yellow)
            afw.write(96, col + 4, '=AVERAGE({0}95:{0}96)'.format(index_to_char(col + 4)), green)
        afw.write(93, len(male_age_groups) + 4, 'Average', black)
        for row in range(94, 97):
            afw.write(row, len(male_age_groups) + 4, '=AVERAGE({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 10
        afw.write(98, 3, 'Carcass Weight (Kilograms)', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 99, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(99, col + 4, gp, black)
            afw.write(100, col + 4, 0, yellow)
            afw.write(101, col + 4, 0, yellow)
            afw.write(102, col + 4, '=AVERAGE({0}101:{0}102)'.format(index_to_char(col + 4)), green)
        afw.write(99, len(male_age_groups) + 4, 'Average', black)
        for row in range(100, 103):
            afw.write(row, len(male_age_groups) + 4, '=AVERAGE({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 11
        afw.write(104, 3, 'Trophy Metrics (Relative Scale)', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 105, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(105, col + 4, gp, black)
            afw.write(106, col + 4, 0, yellow)
            afw.write(107, col + 4, 0, yellow)
            afw.write(108, col + 4, '=AVERAGE({0}107:{0}108)'.format(index_to_char(col + 4)), green)
        afw.write(105, len(male_age_groups) + 4, 'Average', black)
        for row in range(106, 109):
            afw.write(row, len(male_age_groups) + 4, '=AVERAGE({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 12
        afw.write(111, 3, 'Economic-based inputs', bold)

        afw.write(113, 3, 'Cost of General License ($/license)', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 114, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(114, col + 4, gp, black)
            afw.write(115, col + 4, 0, yellow)
            afw.write(116, col + 4, 0, yellow)
            afw.write(117, col + 4, '=AVERAGE({0}116:{0}117)'.format(index_to_char(col + 4)), green)
        afw.write(114, len(male_age_groups) + 4, 'Average', black)
        for row in range(115, 118):
            afw.write(row, len(male_age_groups) + 4, '=AVERAGE({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 13
        afw.write(119, 3, 'Cost of Outifitter License ($/license)', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Average']):
            afw.write(row + 120, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(120, col + 4, gp, black)
            afw.write(121, col + 4, 0, yellow)
            afw.write(122, col + 4, 0, yellow)
            afw.write(123, col + 4, '=AVERAGE({0}122:{0}123)'.format(index_to_char(col + 4)), green)
        afw.write(120, len(male_age_groups) + 4, 'Average', black)
        for row in range(121, 124):
            afw.write(row, len(male_age_groups) + 4, '=AVERAGE({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 14
        afw.write(125, 3, 'General License Revenue ($)', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Sum']):
            afw.write(row + 126, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(126, col + 4, gp, black)
            afw.write(127, col + 4, 0, yellow)
            afw.write(128, col + 4, 0, yellow)
            afw.write(129, col + 4, '=SUM({0}128:{0}129)'.format(index_to_char(col + 4)), green)
        afw.write(126, len(male_age_groups) + 4, 'Sum', black)
        for row in range(127, 130):
            afw.write(row, len(male_age_groups) + 4, '=SUM({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        # Block 15
        afw.write(131, 3, 'Outfitter License Revenue ($)', bold)
        for row, heading in enumerate(['Gender/Stage', 'Males', 'Females', 'Sum']):
            afw.write(row + 132, 3, heading, black)
        for col, gp in enumerate(male_age_groups):
            afw.write(132, col + 4, gp, black)
            afw.write(133, col + 4, 0, yellow)
            afw.write(134, col + 4, 0, yellow)
            afw.write(135, col + 4, '=SUM({0}134:{0}135)'.format(index_to_char(col + 4)), green)
        afw.write(132, len(male_age_groups) + 4, 'Sum', black)
        for row in range(133, 136):
            afw.write(row, len(male_age_groups) + 4, '=SUM({0}{2}:{1}{2})'.format(
                index_to_char(4), index_to_char(len(male_age_groups) + 3), row + 1
            ), green)

        afw.set_column(2, 2, 40)
        afw.set_column(3, 3, 13)
        for col in range(len(male_age_groups)):
            afw.set_column(col + 4, col + 4, 13)

        tb.activate()

        wb.close()

    return path
