"""
Summarize data from a popdyn simulation

ALCES 2018
"""


def total_population(self, species, time, sex=None, age_gp=None,
                     migrated=False, file_instance=None):
    """
    Retrieve the total population in the domain at a given point in time
    """

    def get_tot(f, age_gp):
        try:
            f['%s/%s' % (species, time)].keys()
        except:
            # No time slice in domain, which means no population exists
            for _sex in ['male', 'female']:
                f.create_group('%s/%s/%s' % (species, time, _sex))
            return numpy.zeros(shape=self.shape, dtype='float32')
        local_dict = {}
        if sex is None:
            sexiter = ['male', 'female']
        else:
            sexiter = [sex]
        for _sex in sexiter:
            if migrated:

                # Use the migrated population
                prefix = 'migrations/%s' % (_sex)
            else:
                prefix = '%s/%s/%s' % (species, time, _sex)
            try:
                keys = f[prefix].keys()
            except KeyError:
                # Sex does not exist- create group to avoid error
                f.create_group(prefix)
                keys = []
            for key in keys:
                try:
                    age = int(key)
                except ValueError:
                    continue
                if age_gp is not None:
                    if ((dur_from <= age <= dur_to) or
                            (age > self.durations[-1][1])):
                        d_key = '%s/%s' % (prefix, key)
                        local_dict['%s%i' % (_sex, age)] = f[d_key]
                else:
                    d_key = '%s/%s' % (prefix, key)
                    local_dict['%s%i' % (_sex, age)] = f[d_key]

        if len(local_dict) == 0:
            # No population yet in this time slice
            total = numpy.zeros(shape=self.shape, dtype='float32')
        elif len(local_dict) == 1:
            total = local_dict.values()[0][:]
        elif len(local_dict) > 31:
            # Need to process in chunks due to numexpr limitations
            total = numpy.zeros(shape=self.shape, dtype='float32')
            keys = local_dict.keys()
            keychunks = range(0, len(local_dict) + 31, 31)
            keychunks = zip(keychunks[:-1],
                            keychunks[1:-1] + [len(local_dict)])
            for ch in keychunks:
                new_local = {k: local_dict[k] for k in keys[ch[0]: ch[1]]}
                total += ne.evaluate(
                    '+'.join(new_local.keys()),
                    local_dict=new_local
                )
        else:
            total = ne.evaluate(
                '+'.join(local_dict.keys()),
                local_dict=local_dict
            )
        return total

    if age_gp is not None:
        dur_from, dur_to = [self.durations[i]
                            for i, ag in enumerate(self.ages)
                            if age_gp == ag][0]

    if file_instance is not None:
        return get_tot(file_instance, age_gp)
    with self.file as f:
        return get_tot(f, age_gp)


def total_mortality(self, species, time, sex=None, age_gp=None, mortality_name=None):
    """
    Collect the sum of mortality arrays for a given query set
    :param species: species name
    :param time: time slice
    :param sex: male or female
    :param age_gp: specific age group
    :param mortality_name: mortality name
    :return:
    """
    reverse_mortality = {}
    for mortName in set(self.mortality_names.values()):
        for mortKey, matchName in self.mortality_names.iteritems():
            if mortName == matchName:
                try:
                    reverse_mortality[mortName].append(mortKey)
                except KeyError:
                    reverse_mortality[mortName] = [mortKey]
    if age_gp is not None:
        dur_from, dur_to = [self.durations[i]
                            for i, ag in enumerate(self.ages)
                            if age_gp == ag][0]
    with self.file as f:
        try:
            f['%s/%s' % (species, time)].keys()
        except:
            raise PopdynError('No time slice "%s" in model domain' % time)

        local_dict = {}
        if sex is None:
            sexiter = ['male', 'female']
        else:
            sexiter = [sex]
        local_dict_enum = 0
        for _sex in sexiter:
            prefix = '%s/%s/%s/mortality' % (species, time, _sex)
            try:
                keys = f[prefix].keys()
            except KeyError:
                # Mortality/sex does not exist- create group to avoid error
                f.create_group(prefix)
                keys = []
            for key in keys:
                try:
                    age = int(key)
                except ValueError:
                    continue
                if age_gp is not None:
                    if ((dur_from <= age <= dur_to) or
                            (age > self.durations[-1][1])):
                        d_key = '%s/%s' % (prefix, key)
                        for mort_key in f[d_key].keys():
                            if mortality_name is not None:
                                try:
                                    if mort_key not in reverse_mortality[str(mortality_name)]:
                                        continue
                                except KeyError:
                                    if mortality_name != mort_key:
                                        continue
                            full_mort_key = '{}/{}'.format(d_key, mort_key)
                            local_dict_enum += 1
                            local_dict['_{}'.format(local_dict_enum)] = f[full_mort_key]
                else:
                    d_key = '%s/%s' % (prefix, key)
                    for mort_key in f[d_key].keys():
                        if mortality_name is not None:
                            try:
                                if mort_key not in reverse_mortality[str(mortality_name)]:
                                    continue
                            except KeyError:
                                if mortality_name != mort_key:
                                    continue
                        full_mort_key = '{}/{}'.format(d_key, mort_key)
                        local_dict_enum += 1
                        local_dict['_{}'.format(local_dict_enum)] = f[full_mort_key]

        if len(local_dict) == 0:
            # No mortality in this time slice
            print "Warning: No mortality to sum using:\n{}".format(
                '\n'.join(map(str, [species, time, sex, age_gp, mortality_name])))
            total = numpy.zeros(shape=self.shape, dtype='float32')
        elif len(local_dict) == 1:
            total = local_dict.values()[0][:]
        elif len(local_dict) > 31:
            # Need to process in chunks due to numexpr limitations
            total = numpy.zeros(shape=self.shape, dtype='float32')
            keys = local_dict.keys()
            keychunks = range(0, len(local_dict) + 31, 31)
            keychunks = zip(keychunks[:-1],
                            keychunks[1:-1] + [len(local_dict)])
            for ch in keychunks:
                new_local = {k: local_dict[k] for k in keys[ch[0]: ch[1]]}
                total += ne.evaluate(
                    '+'.join(new_local.keys()),
                    local_dict=new_local
                )
        else:
            total = ne.evaluate(
                '+'.join(local_dict.keys()),
                local_dict=local_dict
            )
        return total


def get_births(self, species, time, sex=None):
    """
    Collect births at a time step for a species
    :param species:
    :param time:
    :param sex:
    :return:
    """
    if sex is None:
        sexiter = ['male', 'female']
    else:
        sexiter = [sex]
    births = numpy.zeros(shape=self.shape, dtype='float32')
    with self.file as f:
        try:
            test = f['%s/%s' % (species, time + self.time_step)]
        except KeyError:
            raise PopdynError('Births were not simulated at time slice %s' % time)
        for _sex in sexiter:
            try:
                births += f['%s/%s/%s/0' % (species, time + self.time_step, _sex)]
            except KeyError:
                # Sex or age does not exist- return zeros
                pass
    return births


def group_from_age(self, age):
    """
    Get the age group name from an absolute age
    :param age: absolute age
    :return: group name
    """
    maxAge = numpy.max(numpy.concatenate([range(dur[0], dur[1] + 1) for dur in self.durations]))
    for gp, dur in zip(self.ages, self.durations):
        durRange = range(dur[0], dur[1] + 1)
        if age in durRange or (durRange[-1] == maxAge and age > maxAge):
            return gp
    raise PopDynError('Query age is not within any groups')

def average_age(self, species, time, sex=None):
    """
    Compute the reduced average species population age at a given time step or for a given sex
    :param species: Input species name
    :param time: Time step value
    :param sex: 'male' or 'female'
    :return: age (float scalar)
    """
    with self.file as f:
        try:
            f['%s/%s' % (species, time)].keys()
        except:
            # No time slice in domain, which means no population exists
            return 0.
        if sex is None:
            sexiter = ['male', 'female']
        else:
            sexiter = [sex]

        a, n = 0., 0.
        for _sex in sexiter:
            prefix = '%s/%s/%s' % (species, time, _sex)
            try:
                keys = f[prefix].keys()
            except KeyError:
                # Sex does not exist- create group to avoid error
                f.create_group(prefix)
                keys = []
            for key in keys:
                try:
                    age = int(key)
                except ValueError:
                    continue
                pop = numpy.sum(f['%s/%s' % (prefix, key)])
                n += pop
                a += pop * age
    if n > 0:
        return a / n
    else:
        return 0.
