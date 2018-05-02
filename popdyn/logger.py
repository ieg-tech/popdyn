"""
Populations dynamics logging tools

Devin Cairns 2018
"""

import time as profile_time
import string
import os
import numpy as np
import xlsxwriter
from popdyn.summary import summarize_species
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
    path = os.path.join(output_directory, datetime.now(tzlocal()).strftime('popdyn_%d%B%Y_%I%M%p'))

    # Collect the summarize information from the domain
    domain_dict = summarize_species(domain)

    def wb_write(row, col, val):
        try:
            val = float(val)
            if val == 0:
                tb.write(row, col, val, grey)
            else:
                tb.write(row, col, val)
        except:
            if np.isinf(val) or np.isnan(val):
                val = ''
            tb.write(row, col, val)

    def index_to_char(index):
        if index < 26:
            return string.ascii_uppercase[index]
        else:
            return (string.ascii_uppercase[(index - 26) / 26] +
                    string.ascii_uppercase[(index - 26) % 26])

    for species, file_dict in domain_dict.items():
        species_path = os.path.join(path, '{}.xlsx'.format(species))

        wb = xlsxwriter.Workbook(species_path)
        bold = wb.add_format({'bold': True})
        grey = wb.add_format()
        grey.set_pattern(1)  # This is optional when using a solid fill.
        grey.set_bg_color('#D3D3D3')

        chartRowIndex = {}
        for tab_key, species_dict in file_dict.items():
            if tab_key == 'Time':
                continue
            tb = wb.add_worksheet(tab_key)
            if tab_key == 'Parameterization':
                _row = -1
                for col_1, col_2 in species_dict.items():
                    _row += 1
                    tb.write(_row, 0, col_1, bold)
                    tb.write(_row, 1, col_2)
                tb.set_column(0, 0, max(len(str(key)) for key in species_dict.keys()))
                tb.set_column(1, 1, max(len(str(key)) for key in species_dict.values()))
                continue
            elif tab_key == 'Habitat':
                tb.write(0, 0, 'Species', bold)
                tb.write(0, 1, 'Carrying capacity', bold)
                for col, _time in enumerate(map(str, file_dict['Time'])):
                    tb.write(0, col + 2, float(_time), bold)
                _row = 0
                for species, itemList in species_dict.items():
                    _row += 1
                    tb.write(_row, 0, species)
                    tb.write(_row, 1, 'Total')
                    for _col, item in enumerate(itemList):
                        wb_write(_row, _col + 2, str(item))
                tb.set_column(0, 0, max([len(sp) for sp in species_dict.keys()] + [len('Species')]))
                tb.set_column(1, 1, len('Carrying Capacity'))
                continue
            elif tab_key == 'Solver':
                tb.write(0, 0, 'Run Date', bold)
                tb.write(0, 1, species_dict[0])
                tb.write(2, 0, 'Function', bold)
                tb.write(2, 1, 'Cumulative time (seconds)', bold)
                _row = 2
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
                        for col, item in enumerate(itemList):
                            wb_write(row, col + 3, str(item))
            tb.set_column(0, 0, spLen)
            tb.set_column(1, 1, gpLen)
            tb.set_column(2, 2, nameLen)

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
                                 'Population', '{} Total Population By Age Class'.format(species)) for gp in self.age_groups],
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
                                'Population', '{} Total Females By Age Class'.format(species)) for gp in self.age_groups],
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
                                'Population', '{} Total Males By Age Class'.format(species)) for gp in self.age_groups],
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
                                'New Population', '{} New Offspring from each Female Age Class'.format(species)) for gp in self.age_groups],
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
                                'Deaths', '{} Total Mortality by Age Class'.format(species)) for gp in self.age_groups],
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
                                'Deaths', '{} Total Male Mortality by Age Class'.format(species)) for gp in self.age_groups],
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
                                'Deaths', '{} Total Female Mortality by Age Class'.format(species)) for gp in self.age_groups],
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
                                 'Deaths', '{} Total Mortality by Mortality Type (anthro, natural)'.format(species))\
                                for mm in np.unique(self.mortalityNames).tolist() + ['Old Age', 'Density Dependent']],
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
                                 'Deaths', '{} Total Male Mortality by Mortality Type'.format(species))\
                                for mm in np.unique(self.mortalityNames).tolist() + ['Old Age', 'Density Dependent']],
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
                                for mm in np.unique(self.mortalityNames).tolist() + ['Old Age', 'Density Dependent']]
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
            tb.activate()
        except KeyError as e:
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

        wb.close()
