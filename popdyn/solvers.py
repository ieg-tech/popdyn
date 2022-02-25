"""
Population dynamics numerical solvers

Devin Cairns, 2018
"""
from numpy.lib.function_base import disp
from popdyn import *
import dask.array as da

# import dafake as da


class SolverError(Exception):
    pass


# Carrying Capacity is 0
INF = 0


def error_check(domain):
    """
    Ensure there are no data conflicts within a domain prior to calling a solver. Current checks include:

    - Make sure at least one Species is included
    - Make sure age (stage) groups do not have overlapping ages
    - Make sure species defined in inter-species relationships are included

    :param Domain domain: Domain instance
    """

    def test(sex_d):
        """Collect all discrete ages from a sex"""
        ages = []
        for age_group in sex_d.keys():
            if age_group is None:
                continue

            instance = sex_d[age_group]
            if any([isinstance(instance, obj) for obj in [Species, Sex, AgeGroup]]):
                ages += range(instance.min_age, instance.max_age + 1)

        if np.any(np.diff(np.sort(ages)) != 1):
            raise SolverError(
                "The species (key) {} {}s have group age ranges that overlap or have gaps".format(
                    species_key, sex
                )
            )

    # Make sure at least one species exists in the domain
    species_keys = domain.species.keys()
    if len(species_keys) == 0:
        raise SolverError("At least one species must be placed into the domain")

    # Iterate species and ensure there are no overlapping age groups
    for species_key in species_keys:
        # Iterate sexes
        for sex in domain.species[species_key].keys():
            # Collect the ages associated with this sex if possible
            if sex is None:
                continue
            test(domain.species[species_key][sex])

    # Check that species with relationships are both included
    recipient_ranges = {}

    def get_species(d):
        for key, val in d.items():
            if isinstance(val, dict):
                get_species(val)
            elif isinstance(val, tuple):
                if val[0].species is not None:
                    species.append(val[0].species.name_key)
                if (
                    hasattr(val[0], "recipient_species")
                    and val[0].recipient_species is not None
                ):
                    species.append(val[0].recipient_species.name_key)

    for ds_type in ["carrying_capacity", "mortality"]:
        species = []
        get_species(getattr(domain, ds_type))
        if not np.all(np.in1d(np.unique(species), species_keys)):
            # Flesh out the species the hard way for a convenient traceback
            for sp in species:
                if sp not in species_keys:
                    raise SolverError(
                        "The domain requires the species (key) {} to calculate {}".format(
                            sp, ds_type.replace("_", " ")
                        )
                    )

    def all_times(domain):
        """duplicate of summary.all_times"""

        def _next(gp_cnt):
            group, cnt = gp_cnt
            # time is always 4 nodes down
            if cnt == 4:
                times.append(int(group.name.split("/")[-1]))
            else:
                for key in group.keys():
                    _next((group[key], cnt + 1))

        times = []

        for key in domain.file.keys():
            _next((domain.file[key], 1))

        return np.unique(times)


def inherit(domain, start_time, end_time):
    """
    Divide population and carrying capacity among children of species if they exist, and apply max age truncation to
    all children species.

    :param domain: Domain instance
    :param int start_time: A simulation start time that coincides with that of the specified solver time range
    :param int end_time: A simulation stop time that coincides with that of the specified solver time range
    """
    # Iterate all species and times in the domain
    for species in domain.species.keys():
        # Check on the age truncation
        live_past_max = False
        sex_keys = [key for key in domain.species[species].keys() if key is not None]
        for time in range(start_time, end_time + 1):
            # Collect the toplevel species datasets if they exist, and distribute them among the sexes
            population_ds = cc_ds = None
            species_population_key = domain.get_population(species, time)
            # Multiple carrying capacity datasets may exist
            carrying_capacity = domain.get_carrying_capacity(
                species, time, snap_to_time=False
            )

            # Population
            if species_population_key is not None and len(sex_keys) > 0:
                # Read the species-level population data and remove from the domain
                population_ds = domain[species_population_key][:] / len(sex_keys)
                domain.remove_dataset(
                    "population", species, None, None, time, None
                )  # Group, sex and age are None
            # Carrying capacity
            if len(carrying_capacity) > 0 is not None and len(sex_keys) > 0:
                # Read the species-level carrying capacity data and remove from the domain
                cc_ds = []
                for cc in carrying_capacity:
                    if cc[1] is not None:
                        cc_ds.append((cc[0], domain[cc[1]][:] / len(sex_keys)))
                    else:
                        cc_ds.append((cc[0], None))
                    # Remove each dataset under the name key
                    domain.remove_dataset(
                        "carrying_capacity", species, None, None, time, cc[0].name_key
                    )

            if None in domain.species[species].keys():
                if None in domain.species[species][None].keys() and isinstance(
                    domain.species[species][None][None], Species
                ):
                    live_past_max = domain.species[species][None][None].live_past_max

            # Iterate any sexes in the domain
            for sex in sex_keys:
                if live_past_max and None in domain.species[species][sex].keys():
                    instance = domain.species[species][sex][None]
                    if isinstance(instance, Species):
                        print(
                            "{} {} inheriting live_past_max from {}".format(
                                species, sex, species
                            )
                        )
                        instance.live_past_max = True

                # Collect groups
                group_keys = [
                    key
                    for key in domain.species[species][sex].keys()
                    if key is not None
                ]
                if len(group_keys) == 0:
                    # Apply any species-level data to the sex dataset
                    if population_ds is not None:
                        instance = domain.species[species][sex][None]

                        print("Cascading population {} --> {}".format(species, sex))
                        domain.add_population(
                            instance,
                            population_ds,
                            time,
                            distribute=False,
                            overwrite="add",
                        )
                    if cc_ds is not None:
                        instance = domain.species[species][sex][None]
                        for cc in cc_ds:
                            print(
                                "Cascading carrying capacity {} {} --> {}".format(
                                    cc[0].name, species, sex
                                )
                            )
                            if cc[1] is not None:
                                domain.add_carrying_capacity(
                                    instance,
                                    cc[0],
                                    time,
                                    cc[1],
                                    distribute=False,
                                    overwrite="add",
                                )
                            else:
                                domain.add_carrying_capacity(
                                    instance,
                                    cc[0],
                                    time,
                                    distribute=False,
                                    overwrite="add",
                                )
                else:
                    # Combine the species-level and sex-level datasets for population
                    # ---------------------------------------------------------------
                    sex_population_key = domain.get_population(species, time, sex)
                    pop_to_groups = True
                    if sex_population_key is None:
                        if population_ds is None:
                            # No populations at species or sex level
                            pop_to_groups = False
                        else:
                            # Use only the species-level key divided among groups
                            sp_sex_prefix = "{} --> [{}] --> ".format(species, sex)
                            next_ds = population_ds / len(group_keys)
                    else:
                        # Collect the sex-level data and remove them from the Domain
                        next_ds = domain[sex_population_key][:]
                        domain.remove_dataset(
                            "population", species, sex, None, time, None
                        )  # Group and age are None

                        # Add the species-level data if necessary and divide among groups
                        if population_ds is not None:
                            sp_sex_prefix = "{} --> {} --> ".format(species, sex)
                            next_ds += population_ds
                        else:
                            sp_sex_prefix = "[{}] --> {} --> ".format(species, sex)
                        next_ds /= len(group_keys)

                    # Combine the species-level and sex-level datasets for carrying capacity
                    # ----------------------------------------------------------------------
                    # All carrying capacity datasets are appended, and not added together
                    next_cc_ds = []
                    if cc_ds is not None:
                        # Use only the species-level keys divided among groups
                        for cc in cc_ds:
                            if cc[1] is None:
                                next_cc_ds.append(cc)
                            else:
                                next_cc_ds.append((cc[0], cc[1] / len(group_keys)))

                    sex_carrying_capacity = domain.get_carrying_capacity(
                        species, time, sex, snap_to_time=False
                    )
                    # Collect the sex-level data, append them to groups, and remove them from the Domain
                    for cc in sex_carrying_capacity:
                        if cc[1] is None:
                            next_cc_ds.append(cc)
                        else:
                            next_cc_ds.append(
                                (cc[0], domain[cc[1]][:] / len(group_keys))
                            )
                        # Remove each dataset under the name key
                        domain.remove_dataset(
                            "carrying_capacity",
                            species,
                            sex,
                            None,
                            time,
                            cc[0].name_key,
                        )

                    # Apply populations and carrying capacity to groups
                    for group in group_keys:
                        instance = domain.species[species][sex][group]
                        if live_past_max:
                            # Apply the sex-level live-past-max attribute to the child
                            print(
                                "{} {} {} inheriting live_past_max from {} {}".format(
                                    species, sex, group, species, sex
                                )
                            )
                            instance.live_past_max = True

                        if pop_to_groups:
                            print(
                                "Cascading population {}{}".format(sp_sex_prefix, group)
                            )
                            domain.add_population(
                                instance,
                                next_ds,
                                time,
                                distribute=False,
                                overwrite="add",
                            )
                        for cc in next_cc_ds:
                            print(
                                "Cascading carrying capacity {} -- > [{}] --> {}".format(
                                    cc[0].name, sex, group
                                )
                            )
                            if cc[1] is None:
                                domain.add_carrying_capacity(
                                    instance,
                                    cc[0],
                                    time,
                                    distribute=False,
                                    overwrite="add",
                                )
                            else:
                                domain.add_carrying_capacity(
                                    instance,
                                    cc[0],
                                    time,
                                    cc[1],
                                    distribute=False,
                                    overwrite="add",
                                )


def compute_domain(
    solver, domain, first_datasets, delayed_datasets, first_aux=None, delayed_aux=None
):
    """
    Takes dask arrays and dataset pointers and computes/writes to the file

    NOTE: dask.store optimization will not allow multiple writes of the same output from the graph

    :param solver: Instance of a solver class
    :param Domain domain: Instance of domain class
    :param dict first_datasets: dataset pointers (keys) and respective dask arrays
    :param dict delayed_datasets:
    :param dict first_aux: delayed auxiliary data
    :param dict delayed_aux: delayed auxiliary data
    """
    for datasets, aux in [(first_datasets, first_aux), (delayed_datasets, delayed_aux)]:
        if len(datasets) == 0:
            continue
        if aux is None:
            aux = {}

        domain.timer.start("cast all data")
        # Force all data to float32, and place in a delayed location first
        sources = list(ds.astype(np.float32) for ds in datasets.values()) + list(
            aux.values()
        )
        domain.timer.stop("cast all data")

        domain.timer.start("create target datasets")
        # Create all necessary datasets in the file, including counter objects
        if h5py.__name__ == "h5py":
            kwargs = {"compression": "lzf"}
        else:
            # h5fake
            kwargs = {
                "sr": domain.projection,
                "gt": (domain.left, domain.csx, 0, domain.top, 0, domain.csy * -1),
                "nd": [0],
            }

        targets = [
            domain.file.require_dataset(
                dp, shape=domain.shape, dtype=np.float32, chunks=domain.chunks, **kwargs
            )
            for dp in datasets.keys()
        ] + [Counter(solver, aux_key) for aux_key in aux.keys()]
        domain.timer.stop("create target datasets")

        domain.timer.start("compute and store")
        store(sources, targets)
        domain.timer.stop("compute and store")

        for key in list(datasets.keys()):
            # Add population keys to domain
            species, sex, group, time, age = domain._deconstruct_key(key)[:5]
            if age not in [
                "params",
                "flux",
            ]:  # Avoid offspring, carrying capacity, and mortality
                domain.population[species][sex][group][time][age] = key

        # Flush buffers to the disk
        if hasattr(domain.file, "flush"):
            domain.file.flush()


class discrete_explicit(object):
    """
    Solve the domain from a start time to an end time on a discrete time step associated with the Domain.

    The total populations from the previous year are used to calculate derived
    parameters (density, fecundity, inter-species relationships, etc.).

    Parameters used during execution are stored in the domain under the ``params`` key, while magnitude
    related to mortality and fecundity are stored under the ``flux`` key at each time step.

    To run the simulation, use .execute()
    """

    def __init__(self, domain, start_time, end_time, **kwargs):
        """
        :param Domain domain: Domain instance populated with at least one species
        :param int start_time: Start time of the simulation
        :param int end_time: End time of the simulation

        :Keyword Arguments:
            **total_density** (*bool*) --
                Use the total species population for density calculations (as opposed to
                populations of sexes or groups) (Default: True)
        """
        # Prepare time
        start_time = Domain._get_time_input(start_time)
        end_time = Domain._get_time_input(end_time)
        self.simulation_range = range(start_time + 1, end_time + 1)

        self.D = domain
        self.prepare_domain(start_time, end_time)

        # A counter is used to track consecutive years of non-zero populations
        self.counter = {}

        # Extra toggles
        # -------------
        self.total_density = kwargs.get("total_density", True)

    def prepare_domain(self, start_time, end_time):
        """Error check and apply inheritance to domain"""
        error_check(self.D)
        inherit(self.D, start_time, end_time)

    def execute(self):
        """Run the simulation"""
        self.D.timer.start("execute")
        all_species = self.D.species.keys()

        # Iterate time. The first time step cannot be solved and serves to provide initial parameters
        for time in self.simulation_range:
            print("Solving {}".format(time))
            self.current_time = time

            # Dask pointers to HDF5 datasets are tracked to avoid redundant I/O
            self.dsts = {}
            self.population_arrays = {sp: {} for sp in all_species}
            self.carrying_capacity_arrays = {sp: {} for sp in all_species}
            output, _delayed, counter_update, delayed_counter_update = {}, {}, {}, {}

            # All births are recorded from each sex and are added to the age 0 slot
            # Age zero populations are updated with offspring calculated in self.propagate()
            self.age_zero_population = {}
            for species in all_species:
                self.age_zero_population[species] = {}
                sex_keys = self.D.species[species].keys()
                if any([fm in sex_keys for fm in ["male", "female"]]):
                    sex_keys = [key for key in sex_keys if key is not None]
                for _sex in sex_keys:
                    self.age_zero_population[species][_sex] = da_zeros(
                        self.D.shape, self.D.chunks
                    )

            # Hierarchically traverse [species --> sex --> groups --> age] and solve children populations
            # Each are solved independently, deriving relationships from baseline data
            # -------------------------------------------------------------------------------------------
            # Collect totals for entire species-based calculations, and dicts to store all output and offspring
            self.totals(all_species, time)

            for species in all_species:

                # Look for a minimum viable population value in a Species-level instance, and apply it to the total
                #   population in advance of propagating the species
                mvp = 0
                self.mvp_coeff = 1.0
                for _sex in self.D.species[species].keys():
                    if _sex is None:
                        for _gp in self.D.species[species][_sex].keys():
                            if _gp is None:
                                if isinstance(
                                    self.D.species[species][_sex][_gp], Species
                                ):
                                    species_instance = self.D.species[species][_sex][
                                        _gp
                                    ]
                                    mvp = species_instance.minimum_viable_population
                if mvp > 0:
                    study_area = (
                        da.sum(self.carrying_capacity_arrays[species]["total"] > 0)
                        * self.D.csx
                        * self.D.csy
                    )

                    self.mvp_coeff = dispersal.minimum_viable_population(
                        self.population_arrays[species]["total {}".format(time)],
                        species_instance.minimum_viable_population,
                        species_instance.minimum_viable_area,
                        self.D.csx,
                        self.D.csy,
                        study_area,
                    )

                # Collect sex keys
                sex_keys = self.D.species[species].keys()
                # Sex may not exist and will be None, but if one of males or females are included,
                # do not propagate Nonetypes
                if any([fm in sex_keys for fm in ["male", "female"]]):
                    sex_keys = [key for key in sex_keys if key is not None]

                # Iterate sexes and groups and calculate each individually
                for sex in sex_keys:
                    for group in self.D.species[species][
                        sex
                    ].keys():  # Group - may not exist and will be None
                        (
                            _output,
                            __delayed,
                            _counter_update,
                            _delayed_counter_update,
                        ) = self.propagate(species, sex, group, time)
                        output.update(_output)
                        _delayed.update(__delayed)
                        counter_update.update(_counter_update)
                        delayed_counter_update.update(_delayed_counter_update)

                # Update the output with the age 0 populations
                for _sex in self.age_zero_population[species].keys():

                    zero_group = self.D.group_from_age(species, _sex, 0)
                    if zero_group is None:

                        # Apply the offspring to the youngest group or do not record an age
                        zero_group = self.D.youngest_group(species, _sex)
                        if zero_group is not None:
                            if time == self.simulation_range[0]:
                                print(
                                    "Warning: no group with age 0 exists for {} {}s.\n"
                                    "Offspring will be added to the group {}.".format(
                                        zero_group.name, _sex, zero_group.group_name
                                    )
                                )

                            age = zero_group.min_age
                            zero_group = zero_group.group_key

                        else:
                            # Age is not recorded
                            age = None
                    else:
                        age = 0
                    key = "{}/{}/{}/{}/{}".format(species, _sex, zero_group, time, age)
                    existing_age = self.D.get_population(
                        species, self.current_time, _sex, zero_group, age
                    )

                    if existing_age is not None:
                        try:
                            self.age_zero_population[species][_sex] += self.dsts[
                                existing_age
                            ]
                        except KeyError:
                            self.dsts[existing_age] = da.from_array(
                                self.D[existing_age], self.D.chunks
                            )
                            self.age_zero_population[species][_sex] += self.dsts[
                                existing_age
                            ]

                    try:
                        # Add them together if the key exists in the output
                        output[key] += self.age_zero_population[species][_sex]
                    except KeyError:
                        output[key] = self.age_zero_population[species][_sex]
                    counter_update[key] = da.any(output[key] > 0)

            # Compute this time slice and write to the domain file
            compute_domain(
                self, self.D, output, _delayed, counter_update, delayed_counter_update
            )

            # Clean up dataset pointers
            del self.carrying_capacity_arrays
            del self.age_zero_population
            del output, _delayed
            del self.dsts
            del self.population_arrays

        self.D.timer.stop("execute")

    def totals(self, all_species, time):
        """
        Collect all population and carrying capacity datasets, and calculate species totals if necessary.
        Dynamic calculations of carrying capacity are made by calling :func:`collect_parameter`, which may also
        update the ``carrying_capacity`` or ``population`` array ``dicts``.

        :param list all_species: All species in the domain
        :param int time: Time slice to collect totals
        """
        # Accumulate the global population
        global_population = []
        global_carrying_capacity = {}

        for species in all_species:
            # Collect the total population for fecundity and density

            # Population keys
            population = self.D.all_population(species, time - 1)
            # Carrying capacity keys
            cc = self.D.all_carrying_capacity(species, time)

            global_population += population.tolist()
            try:
                global_carrying_capacity[species] += cc
            except KeyError:
                global_carrying_capacity[species] = cc

            if self.total_density:
                # Total population of contributing groups
                self.population_arrays[species][
                    "total {}".format(time)
                ] = self.population_total(population)

                # Carrying Capacity
                self.carrying_capacity_arrays[species][
                    "total"
                ] = self.carrying_capacity_total(species, cc)

            # Calculate species-wide male and female populations that reproduce
            # Note: Species that do not contribute to density contribute to the reproduction totals
            # -----------------------------------------------------------------
            # Males
            self.population_arrays[species][
                "Reproducing Males {}".format(time)
            ] = self.population_total(
                self.D.all_population(species, time - 1, "male"), False, True
            )

            # Females
            self.population_arrays[species][
                "Reproducing Females {}".format(time)
            ] = self.population_total(
                self.D.all_population(species, time - 1, "female"), False, True
            )

            # Calculate Density
            self.population_arrays[species][
                "Female to Male Ratio {}".format(time)
            ] = da_where(
                self.population_arrays[species]["Reproducing Males {}".format(time)]
                > 0,
                self.population_arrays[species]["Reproducing Females {}".format(time)]
                / self.population_arrays[species]["Reproducing Males {}".format(time)],
                INF,
            )

        # The global population includes all species that have the global_population attr set to True
        self.population_arrays["global {}".format(time)] = self.population_total(
            global_population, honor_global=True
        )

        self.carrying_capacity_arrays["global {}".format(time)] = da_zeros(
            self.D.shape, self.D.chunks
        )
        for species, cc_ds in global_carrying_capacity.items():
            self.carrying_capacity_arrays[
                "global {}".format(time)
            ] += self.carrying_capacity_total(species, cc_ds)

    def collect_parameter(self, param, data):
        """
        Collect data associated with a Fecundity, CarryingCapacity, or Mortality instance. If a species is linked,
        data will be None and density of the linked species will be calculated.

        :param param: CarryingCapacity, Fecundity, or Mortality instance
        :param data: :class:`h5py.Dataset` key

        :return: A dask array with the collected data
        """
        kwargs = {}

        # Collect the total species population from the previous time step
        if data is None:
            # There must be a species attached to the parameter instance if there are no data
            # Generate lookup data using the population type of the attached species
            population = self.D.all_population(
                param.species.name_key,
                self.current_time - 1,
                param.species.sex,
                param.species.group_key,
            )
            p = self.population_total(population)

            if param.population_type == "total population":
                population_data = p

            elif param.population_type == "density":
                carrying_capacity = self.D.all_carrying_capacity(
                    param.species.name_key,
                    self.current_time,
                    param.species.sex,
                    param.species.group_key,
                )
                cc = self.carrying_capacity_total(
                    param.species.name_key, carrying_capacity
                )

                population_data = da_where(cc > 0, p / cc, INF)

            elif param.population_type == "global population":
                population_data = self.population_arrays[
                    "global {}".format(self.current_time)
                ]

            elif param.population_type == "global ratio":
                population_data = da_where(
                    self.population_arrays["global {}".format(self.current_time)] > 0,
                    p / self.population_arrays["global {}".format(self.current_time)],
                    INF,
                )
            else:
                raise PopdynError(
                    'Unknown population type argument "{}"'.format(
                        param.population_type
                    )
                )

            kwargs.update(
                {"lookup_data": population_data, "lookup_table": param.species_table}
            )
        else:
            try:
                data = self.dsts[data]
            except KeyError:
                self.dsts[data] = da.from_array(self.D[data], self.D.chunks)
                data = self.dsts[data]

        # Include random parameters (except not if carrying capacity is another species)
        if getattr(param, "random_method", False) and not (
            isinstance(param, CarryingCapacity) and data is None
        ):
            kwargs.update(
                {
                    "random_method": param.random_method,
                    "random_args": param.random_args,
                    "apply_using_mean": param.apply_using_mean,
                }
            )

        kwargs.update({"chunks": self.D.chunks})
        return dynamic.collect(data, **kwargs)

    def interspecies_carrying_capacity(self, parent_species, param):
        """
        Because inter-species dependencies may be nested or circular, the carrying capacity of all connected
        species is calculated herein and added to the ``self.carrying_capacity_arrays`` object. Future calls of any
        of the connected species will have data available, avoiding redundant calculations.

        :param Species parent_species: One species that is tied to others
        :param CarryingCapacity param: Input Carrying Capacity
        :return: coefficient array for the given carrying capacity parameter
        """

        def _population(species):
            """Population at the previous time"""
            population = self.D.all_population(
                species.name_key, self.current_time - 1, species.sex, species.group_key
            )
            return self.population_total(population)

        def next_species_node(param):
            """Prepare the next node in the species graph"""
            # Condition that terminates recursion
            if not param.species in density:
                density[param.species] = {
                    "p": _population(param.species),
                    "d": da_zeros(self.D.shape, self.D.chunks),
                }
                cc = self.D.all_carrying_capacity(
                    param.species.name_key,
                    self.current_time,
                    param.species.sex,
                    param.species.group_key,
                )
                for cc_instance, key in cc:
                    if key is None:
                        next_species_node(cc_instance)
                    else:
                        if (
                            cc_instance
                            not in self.carrying_capacity_arrays[param.species.name_key]
                        ):
                            d = self.collect_parameter(cc_instance, key)
                            self.carrying_capacity_arrays[param.species.name_key][
                                cc_instance
                            ] = d
                        else:
                            d = self.carrying_capacity_arrays[param.species.name_key][
                                cc_instance
                            ]
                        density[param.species]["d"] += d

        # Collect the sum of all density parameters without inter-species perturbation
        density = {}  # Global in func next_species_node
        next_species_node(param)

        # Calculate the coefficient for each carrying capacity instance
        def next_species_edge(param, parent_species):
            """Traverse the edges and calculate coefficients"""
            if param not in complete:
                complete.append(param)
                p, d = density[param.species]["p"], density[param.species]["d"]
                kwargs = {
                    "lookup_data": None,
                    "lookup_table": param.species_table,
                    "chunks": self.D.chunks,
                }

                if param.population_type == "total population":
                    kwargs["lookup_data"] = p

                elif param.population_type == "density":
                    kwargs["lookup_data"] = da_where(d > 0, p / d, INF)

                elif param.population_type == "global population":
                    kwargs["lookup_data"] = self.population_arrays[
                        "global {}".format(self.current_time)
                    ]

                elif param.population_type == "global ratio":
                    kwargs["lookup_data"] = da_where(
                        self.population_arrays["global {}".format(self.current_time)]
                        > 0,
                        p
                        / self.population_arrays["global {}".format(self.current_time)],
                        INF,
                    )
                else:
                    raise PopdynError(
                        'Unknown population type argument "{}"'.format(
                            param.population_type
                        )
                    )

                self.carrying_capacity_arrays[parent_species][param] = dynamic.collect(
                    None, **kwargs
                )

                cc = self.D.all_carrying_capacity(
                    param.species.name_key,
                    self.current_time,
                    param.species.sex,
                    param.species.group_key,
                )
                for cc_instance, key in cc:
                    if key is None:
                        next_species_edge(cc_instance, param.species.name_key)

        complete = []
        next_species_edge(param, parent_species)

    def population_total(
        self, datasets, honor_density=True, honor_reproduction=False, honor_global=False
    ):
        """
        Calculate the total population for a species with the provided datasets, usually a result of a
        :func:`Domain.all_population` call. This updates ``self.population_arrays`` to avoid repetitive IO

        :param species: Species key
        :param list datasets: population dataset keys, retrieved from popdyn.Domain.all_population()
        :param bool honor_density: Use the density contribution attribute from a species
        :param bool honor_reproduction: Use the density contribution to fecundity attribute from a species
        :return: Dask array of total population
        """
        pop_arrays = []

        for key in datasets:
            instance = self.D._instance_from_key(key)

            if honor_density and not instance.contributes_to_density:
                continue

            if honor_global and not instance.global_population:
                continue

            if honor_reproduction:
                # Collect fecundity to see if it is present
                species_key, sex, group_key, time = self.D._deconstruct_key(key)[:4]

                if len(self.D.get_fecundity(species_key, time, sex, group_key)) == 0:
                    continue

            try:
                pop_arrays.append(self.dsts[key])
            except KeyError:
                self.dsts[key] = da.from_array(self.D[key], self.D.chunks)
                pop_arrays.append(self.dsts[key])

        if len(pop_arrays) > 0:
            out = dsum(pop_arrays)
        else:
            out = da_zeros(self.D.shape, self.D.chunks)

        return out

    def carrying_capacity_total(self, species, datasets):
        """
        Calculate the total carrying capacity for a species with the provided datasets, usually a result of a
        :func:`Domain.get_carrying_capacity` call. This updates ``self.carrying_capacity_arrays`` to avoid excessive or
        infinite recursion depth when collecting carrying capacity that may be circular.

        :param datasets: list of tuples with instance - :class:`h5py.Dataset` pairs
        :param list dependent: Tracks the species used in species-dependent carrying capacity calculations
        :return: dask array of total carrying capacity
        """
        cc_coeff = []
        cc_arrays = []
        for cc in datasets:
            # Check if this has already been computed. If any interspecies data are included, they will
            #  all be computed at the first occurrence and added to the arrays.
            try:
                if cc[1] is None:
                    cc_coeff.append(self.carrying_capacity_arrays[species][cc[0]])
                else:
                    cc_arrays.append(self.carrying_capacity_arrays[species][cc[0]])
            except KeyError:
                # The dataset must be calculated or added to the arrays dict
                if cc[1] is None:
                    # Circular relationships are solved here
                    self.interspecies_carrying_capacity(species, cc[0])
                    cc_coeff.append(self.carrying_capacity_arrays[species][cc[0]])
                else:
                    ds = self.collect_parameter(cc[0], cc[1])
                    self.carrying_capacity_arrays[species][cc[0]] = ds
                    cc_arrays.append(ds)

        if len(cc_arrays) == 0:
            array = da_zeros(self.D.shape, self.D.chunks)
        else:
            array = dsum(cc_arrays)

        if len(cc_coeff) == 0:
            coeff = 1.0
        else:
            # Use the mean of the coefficients to avoid compounding perturbation
            coeff = dmean(cc_coeff)

        return da_where(array > 0, array * coeff, 0)

    def get_counter(self, species, group, sex, age, time):
        """
        Collect the consecutive number of years a non-zero population has existed

        :param str species: Species name key
        :param str group: Group name key
        :param str sex: Sex name key
        :param int age: Age of the population
        :param int time: Current model time
        :return: Integer count
        """
        key = "{}/{}/{}".format(species, sex, group)

        # If no data exists or the counter is 0, no timeline exists yet
        try:
            if self.counter["{}/{}/{}".format(key, time, age)]:
                counter = 0
            else:
                return None
        except KeyError:
            return None

        # Accumulate the counter using time and age datasets
        while True:
            time -= 1
            if age is not None:
                age -= 1
            try:
                _cnt = self.counter["{}/{}/{}".format(key, time, age)]
                if _cnt:
                    counter += 1
                else:
                    return counter
            except KeyError:
                return counter

    def propagate(self, species, sex, group, time):
        """
        Create a graph of calculations to propagate populations and record parameters.
        A dict of key pointers - dask array pairs are created and returned

        .. Note::
            - If age gaps exist, they will be filled by empty groups and will propagate
            - Currently, female:male ratio uses the total species population even if densities are calculated at the
              age group level

        :param str species: Species key (``Species.name_key``)
        :param str sex: Sex (``'male'``, ``'female'``, or ``None``)
        :param str group: Group key (``AgeGroup.group_key``)
        :param int time: Time slice
        """
        # Collect some meta on the species
        # -----------------------------------------------------------------
        species_instance = self.D.species[species][sex][group]
        ages = species_instance.age_range
        max_age = self.D.discrete_ages(species, sex)[-1]
        # Output datasets are stored in "flux" and "params" directories,
        # which are for populations changes and model parameters, respectively
        flux_prefix = "{}/{}/{}/{}/flux".format(species, sex, group, time)
        param_prefix = "{}/{}/{}/{}/params".format(species, sex, group, time)

        # Ensure there are populations to propagate
        # -----------------------------------------------------------------
        if all(
            [
                self.D.get_population(species, time - 1, sex, group, age) is None
                for age in ages
            ]
        ):
            return {}, {}, {}, {}

        # Collect parameters from the current time step.
        # All dynamic modifications are also applied in this step
        # -----------------------------------------------------------------
        params = self.calculate_parameters(species, sex, group, time)

        _mortality = self.D.get_mortality(species, time, sex, group)
        # Collect mortality instances
        mort_types = [
            mort_type[0]
            for mort_type in _mortality
            if mort_type[0].recipient_species is None
        ]
        # Collect conversion instances
        conversion_types = [
            mort_type[0]
            for mort_type in _mortality
            if mort_type[0].recipient_species is not None
        ]

        # All outputs are written at once, so create a dict to do so (pre-populated with mortality types as zeros)
        # output, _delayed, counter_update, delayed_counter_update = NanDict(), NanDict(), NanDict(), NanDict()
        output, _delayed, counter_update, delayed_counter_update = {}, {}, {}, {}
        for mort_type in mort_types + conversion_types:
            mort_name = mort_type.name
            output["{}/mortality/{}".format(flux_prefix, mort_name)] = da_zeros(
                self.D.shape, self.D.chunks
            )
            # Write the output mortality parameters
            if not isinstance(params[mort_name], dict):
                output["{}/mortality/{}".format(param_prefix, mort_name)] = params[
                    mort_name
                ]
        for mort_name in ["Old Age", "Density Dependent"]:
            output["{}/mortality/{}".format(flux_prefix, mort_name)] = da_zeros(
                self.D.shape, self.D.chunks
            )

        # Add carrying capacity to the output datasets
        if self.carrying_capacity_total:
            # For the total species population only
            total_prefix = "{}/{}/{}/{}/params".format(species, None, None, time)
            output[
                "{}/carrying capacity/{}".format(total_prefix, "Carrying Capacity")
            ] = params["Carrying Capacity"]
        else:
            # Specific to the species
            output[
                "{}/carrying capacity/{}".format(param_prefix, "Carrying Capacity")
            ] = params["Carrying Capacity"]

        if sex is not None:
            # Check if this species is capable of offspring
            # Calculate births using the effective fecundity
            # The total population of this group is used, as fecundity will be 0
            # within groups that do not reproduce
            output["{}/fecundity/{}".format(param_prefix, "Fecundity")] = params[
                "Fecundity"
            ]
            output[
                "{}/fecundity/{}".format(
                    param_prefix, "Density-Based Fecundity Reduction Rate"
                )
            ] = params["Density-Based Fecundity Reduction Rate"]
            births = params["Population"] * params["Fecundity"]
            male_births = births * params.get("Birth Ratio", 0.5)
            female_births = births - male_births

            # Check if this is a recipient species, as offspring will need to be added to the contributing species
            cont_species = self._contributing_species(species)

            # Add new offspring to males or females
            for _sex in self.age_zero_population[species].keys():
                if _sex == "female":
                    # Add female portion of births
                    if len(cont_species) == 0:
                        self.age_zero_population[species][_sex] += female_births
                    else:
                        for cont_sp in cont_species:
                            self.age_zero_population[cont_sp][
                                _sex
                            ] += female_births / len(cont_species)
                    output[
                        "{}/offspring/{}".format(flux_prefix, "Female Offspring")
                    ] = female_births
                else:
                    if len(cont_species) == 0:
                        self.age_zero_population[species][_sex] += male_births
                    else:
                        for cont_sp in cont_species:
                            self.age_zero_population[cont_sp][
                                _sex
                            ] += male_births / len(cont_species)
                    output[
                        "{}/offspring/{}".format(flux_prefix, "Male Offspring")
                    ] = male_births

        # Density-dependent mortality
        # ---------------------------
        # Constants that scale density-dependent mortality
        density_threshold = species_instance.density_threshold
        density_scale = species_instance.density_scale
        linear_range = max(0.0, 1.0 - density_threshold)

        # Calculate a density-dependent mortality rate if this species contributes to density
        # The density parameter is either the total species or group value, depending on the flag
        if species_instance.contributes_to_density:
            # Make the density 1 or the threshold if it is inf
            # Density dependent rate
            dd_range = da.maximum(0.0, params["Density"] - density_threshold)
            # Apply the scale
            if linear_range == 0:
                linear_proportion = 1.0
            else:
                linear_proportion = da.minimum(1.0, dd_range / linear_range)
            ddm_rate = da_where(
                (params["Density"] > 0) & ~da.isinf(params["Density"]),
                (dd_range * (density_scale * linear_proportion)) / params["Density"],
                0,
            )
            # Make the rate 1 where there is an infinite density
            ddm_rate = da_where(da.isinf(params["Density"]), 1.0, ddm_rate)

        for age in ages:
            # Collect the population of the age from the previous time step
            population = self.D.get_population(species, time - 1, sex, group, age)

            # Use population total to avoid re-read from disk
            population = self.population_total([population], False)

            # Aggregate mortality and collect time-based mortality if necessary
            mortality_data = []
            conversion_data = []
            conv_coeff = self._prepare_conversion(
                species,
                group,
                sex,
                age,
                time,
                param_prefix,
                conversion_types,
                conversion_data,
                params,
                output,
            )
            # Conversion is applied first, so mortality scaling must include conversion calculations
            mort_coeff = self._prepare_mortality(
                species,
                group,
                sex,
                age,
                time,
                param_prefix,
                mort_types,
                mortality_data,
                params,
                output,
                conversion_data,
            )

            # Apply conversion
            et = dt = False
            if hasattr(species_instance, "direct_transmission"):
                dt = True
            if hasattr(species_instance, "environmental_transmission"):
                et = True
            population = self._apply_conversion(
                conversion_types,
                conv_coeff,
                conversion_data,
                population,
                flux_prefix,
                time,
                age,
                output,
                _delayed,
                delayed_counter_update,
                dt,
                et,
            )

            # All mortality types are applied to each age to record sub-populations for each group
            population = self._apply_mortality(
                mort_types,
                mort_coeff,
                mortality_data,
                population,
                flux_prefix,
                output,
                _delayed,
            )

            # Apply old age mortality if necessary to avoid unnecessary dispersal calculations
            if (
                not species_instance.live_past_max
                and max_age is not None
                and age == max_age
            ):
                output["{}/mortality/{}".format(flux_prefix, "Old Age")] += population
                # All done with this population
                continue

            # Apply dispersal
            # TODO: Do children receive dispersal of any parent species classes?

            static_population = population.copy()

            # Collect species and time-based dispersal methods
            dispersal_methods = self.D.get_dispersal(species, time - 1, sex, group, age)

            # Collect all mask keys so they can be queried for dispersal arguments
            mask_query = self.D.get_mask(species, self.current_time, sex, group)
            mask_keys = (
                [mk.split("/")[-1] for mk in mask_query]
                if mask_query is not None
                else None
            )

            for dispersal_method, args in dispersal_methods:
                args = args + (self.D.csx, self.D.csy)

                disp_kwargs = {}
                if mask_keys is not None:
                    kwarg_keys = [
                        mask_key
                        for mask_key in mask_keys
                        if "dispersal__{}".format(dispersal_method) in mask_key
                    ]
                    for mask_key in kwarg_keys:
                        mask_ds = self.D.get_mask(
                            species, self.current_time, sex, group, mask_key
                        )
                        if mask_ds is not None:
                            try:
                                mask_ds = self.dsts[mask_ds]
                            except KeyError:
                                self.dsts[mask_ds] = da.from_array(
                                    self.D[mask_ds], self.D.chunks
                                )
                                mask_ds = self.dsts[mask_ds]
                            disp_kwargs[mask_key.split("__")[-1]] = mask_ds

                population = dispersal.apply(
                    population,
                    self.population_arrays[species]["total {}".format(time)],
                    self.carrying_capacity_arrays[species]["total"],
                    dispersal_method,
                    *args,
                    **disp_kwargs
                )

            if species_instance.contributes_to_density:
                # Avoid density-dependent mortality when dispersal has occurred
                # NOTE: This may still leave a higher density than the threshold if dispersal has exceeded the
                #  carrying capacity, as the DDM rate does not increase (only decreases to ensure dispersal
                #  results in populations being allowed to live where their density has become lower).
                #  Increasing the rate to account for dispersed populations may result in excessive
                #  population reduction to below the carrying capacity. Leave those to the next time step.
                mask = population < static_population
                yes = ddm_rate - (1 - (population / static_population))
                no = ddm_rate
                new_ddm_rate = da_where(mask, yes, no)

                ddm = da_where(new_ddm_rate > 0.0, population * new_ddm_rate, 0)

                output[
                    "{}/mortality/{}".format(param_prefix, "Density Dependent Rate")
                ] = new_ddm_rate
                output[
                    "{}/mortality/{}".format(flux_prefix, "Density Dependent")
                ] += ddm

                population -= ddm

            # Propagate age by one increment and _save to the current time step.
            # If the input age is None, the population will simply propagate
            if age is not None:
                new_age = age + 1
            else:
                new_age = age
            new_group = self.D.group_from_age(species, sex, new_age)
            if new_group is None:
                # No group past this point. If there is a legitimate age, this means that live_past_max is True
                if max_age is not None and age == max_age:
                    # Need to add an age sequentially to the current group
                    self.D.species[species][sex][group].max_age += 1
                    new_group = group
                else:
                    new_age = None

            # Lastly, apply minimum viable population calculations if necessary
            if species_instance.minimum_viable_population > 0:
                study_area = (
                    da.sum(params["Carrying Capacity"] > 0) * self.D.csx * self.D.csy
                )

                mvp_coeff = dispersal.minimum_viable_population(
                    population,
                    species_instance.minimum_viable_population,
                    species_instance.minimum_viable_area,
                    self.D.csx,
                    self.D.csy,
                    study_area,
                )
                population *= mvp_coeff

            key = "{}/{}/{}/{}/{}".format(species, sex, new_group, time, new_age)
            existing_age = self.D.get_population(
                species, self.current_time, sex, new_group, new_age
            )

            # Apply species-level MVP
            population *= self.mvp_coeff

            if existing_age is not None:
                # This population exists, and serves to implement immigration/emigration
                try:
                    ds = self.dsts[existing_age]
                except KeyError:
                    self.dsts[existing_age] = da.from_array(
                        self.D[existing_age], chunks=self.D.chunks
                    )
                    ds = self.dsts[existing_age]
                # Used the delayed keys to make sure calculations take place first
                _delayed[key] = ds + population
                # Do not allow negative populations
                _delayed[key] = da_where(_delayed[key] < 0, 0, _delayed[key])
                delayed_counter_update[key] = da.any(_delayed[key] > 0)

            else:
                # Do not allow negative populations
                output[key] = da_where(population < 0, 0, population)
                counter_update[key] = da.any(output[key] > 0)

        # Return output so that it may be included in the final compute call
        return output, _delayed, counter_update, delayed_counter_update

    def calculate_parameters(self, species, sex, group, time):
        """Collect all required parameters at a time step"""
        parameters = {}

        # Mortality
        # ---------------------
        # Collect Mortality instances and/or data from the domain at the current time step
        for instance, key in self.D.get_mortality(species, time, sex, group):
            # Mortality types remain separated by name to track numbers associated with types
            if hasattr(instance, "time_based_rates"):
                # Data are not collected, as the age timer dictates the rate
                parameters[instance.name] = {
                    "time_based_rates": instance.time_based_rates,
                    "time_based_std": instance.time_based_std,
                }
            else:
                # Collect non-zero mortality datasets
                parameters[instance.name] = self.collect_parameter(instance, key)

                # Ensure no negatives
                parameters[instance.name] = da_where(
                    parameters[instance.name] < 0, 0, parameters[instance.name]
                )

                # Check if a multiplicative modifier mask exists (used to explicitly spatially
                #   constrain inter-species mortality)
                mort_mask = self.D.get_mask(species, time, sex, group, instance.name)
                if mort_mask is not None:
                    # Enforce a maximum of 1
                    parameters[instance.name] *= da_where(
                        mort_mask > 1.0, 1.0, mort_mask
                    )

        # Carrying Capacity, Population, & Density
        # ----------------------------------------
        # Total Population from previous time step (all population, without regard to contribution filtering)
        population_dsts = self.D.all_population(species, time - 1, sex, group)
        parameters["Population"] = self.population_total(population_dsts, False)

        # Collect CarryingCapacity instances and/or data from the domain at the current time step and
        # the total population used for density
        if self.total_density:
            global_cc = self.carrying_capacity_arrays[species]["total"]
            parameters["Carrying Capacity"] = global_cc
            population = self.population_arrays[species]["total {}".format(time)]
        else:
            carrying_capacity = self.D.get_carrying_capacity(species, time, sex, group)
            global_cc = self.carrying_capacity_total(species, carrying_capacity)
            parameters["Carrying Capacity"] = global_cc
            population = self.population_total(population_dsts)

        # Species-level toggle to use global density
        species_instance = self.D.species[species][sex][group]
        if species_instance.use_global_density:
            global_cc = self.carrying_capacity_arrays["global {}".format(time)]
            population = self.population_arrays["global {}".format(time)]

        parameters["Density"] = da_where(global_cc > 0, population / global_cc, INF)

        # Collect fecundity parameters
        # These may be collected for any species, whether it be male or female. If fecundity exists for
        # both males and females, the female rate will override the male rate.
        # ---------------------
        # Collect the fecundity instances - HDF5[or None] pairs
        fecundity = self.D.get_fecundity(species, time, sex, group)

        # Calculate the fecundity values
        # Fecundity is collected then aggregated
        fec_arrays = []
        avg_mod = 0
        for instance, key in fecundity:
            # First, check if the species multiplies
            if getattr(instance, "multiplies"):
                # Filtered through its density lookup table
                fec = self.collect_parameter(instance, key)

                # Apply random perturbation first
                # Apply randomness
                if getattr(instance, "random_method", False):
                    fec = dynamic.collect(
                        fec,
                        random_method=instance.random_method,
                        random_args=instance.random_args,
                        apply_using_mean=instance.apply_using_mean,
                        **{"chunks": self.D.chunks}
                    )

                fec *= dynamic.collect(
                    None,
                    lookup_data=self.population_arrays[species][
                        "Female to Male Ratio {}".format(time)
                    ],
                    lookup_table=instance.fecundity_lookup,
                    **{"chunks": self.D.chunks}
                )

                # Fecundity scales from a low threshold to high and is reduced linearly using a specified rate
                density_fecundity_threshold = getattr(
                    instance, "density_fecundity_threshold"
                )
                density_fecundity_max = getattr(instance, "density_fecundity_max")
                fecundity_reduction_rate = min(
                    1.0, getattr(instance, "fecundity_reduction_rate")
                )

                input_range = max(
                    0.0, density_fecundity_max - density_fecundity_threshold
                )
                density_range = parameters["Density"] - density_fecundity_threshold

                if input_range == 0:
                    fec_mod = da_where(density_range > 0, fecundity_reduction_rate, 0.0)
                else:
                    fec_mod = da_where(
                        density_range > 0,
                        da.minimum(1.0, (density_range / input_range))
                        * fecundity_reduction_rate,
                        0.0,
                    )

                fec -= fec * fec_mod

                avg_mod += fec_mod

                fec = da_where(fec > 0, fec, 0)

                # If multiple fecundity instances are specified for this species, only the last birth ratio will
                #  be used
                birth_ratio = instance.birth_ratio

                if birth_ratio == "random":
                    parameters["Birth Ratio"] = np.random.random()
                else:
                    parameters["Birth Ratio"] = birth_ratio

                # Add to stack
                fec_arrays.append(fec)

        if len(fec_arrays) == 0:
            # Defaults to 0
            parameters["Fecundity"] = da_zeros(self.D.shape, self.D.chunks)
            parameters["Density-Based Fecundity Reduction Rate"] = da_zeros(
                self.D.shape, self.D.chunks
            )
        else:
            parameters["Fecundity"] = dsum(fec_arrays)
            parameters["Density-Based Fecundity Reduction Rate"] = avg_mod / len(
                fec_arrays
            )

        return parameters

    def _contributing_species(self, species_name):
        """
        Determine if the species is the recipient of another, and report the contributing species
        :param str species_name:
        :return: List of contributing species
        """
        cont_species = []
        for mort in self.D.mortality_instances:
            if (
                mort.recipient_species is not None
                and mort.recipient_species.name_key == species_name
            ):
                sps = self.D.all_species_with_mortality(mort)
                for sp in sps:
                    if sp != species_name:
                        cont_species.append(sp)

        return np.unique(cont_species)

    def _prepare_conversion(
        self,
        species,
        group,
        sex,
        age,
        time,
        param_prefix,
        mort_types,
        mortality_data,
        params,
        output,
    ):
        """
        Prepare a coefficient that dictates conversion from one species to another as a result of mortality.
        This functionality was originally developed to approach Chronic Wasting Disease simulations, although may
        be applicable in other situations.

        :param species:
        :param group:
        :param sex:
        :param age:
        :param time:
        :param param_prefix:
        :param mort_types:
        :param mortality_data:
        :param params:
        :param output:
        :param pre_existing_rate:
        :return:
        """
        for mort_type in mort_types:
            species_instance = self.D.species[species][sex][group]
            # Calculate Direct Transmission if it is provided
            if hasattr(species_instance, "direct_transmission") or hasattr(
                species_instance, "environmental_transmission"
            ):
                transmission = da_zeros(self.D.shape, self.D.chunks)
                if hasattr(species_instance, "direct_transmission"):
                    # Total population
                    dsets = self.D.all_population(species, time - 1)
                    if not hasattr(self, "direct_transmission_total"):
                        self.direct_transmission_total = {
                            time - 1: self.population_total(dsets)
                        }
                    else:
                        try:
                            self.direct_transmission_total[
                                time - 1
                            ] = self.population_total(dsets)
                        except KeyError:
                            self.direct_transmission_total = {
                                time - 1: self.population_total(dsets)
                            }

                    # Calculate the proportion of infected species (Pi)
                    dsets = self.D.all_population(
                        mort_type.recipient_species.name_key, time - 1, sex, group
                    )
                    # Where there is no population the rate is 0
                    Pi = da.where(
                        self.direct_transmission_total[time - 1] > 0,
                        self.population_total(dsets)
                        / self.direct_transmission_total[time - 1],
                        0,
                    )

                    # Accumulate FOI using the direct transmission relationships
                    FOI = da_zeros(self.D.shape, self.D.chunks)
                    for from_gp in species_instance.direct_transmission.keys():
                        for from_sex in species_instance.direct_transmission[
                            from_gp
                        ].keys():
                            FOI += (
                                species_instance.direct_transmission[from_gp][from_sex][
                                    str(group).lower()
                                ][str(sex).lower()]
                                * Pi
                            )

                    transmission += FOI
                    output[
                        "{}/mortality/Direct Transmission effective rate".format(
                            param_prefix
                        )
                    ] = transmission

                if hasattr(species_instance, "environmental_transmission"):
                    env_rate = (
                        species_instance.environmental_transmission["C"][
                            str(group).lower()
                        ][str(sex).lower()]
                        * species_instance.environmental_transmission["E"][time]
                    )
                    transmission += env_rate
                    output[
                        "{}/mortality/Env. Transmission effective rate".format(
                            param_prefix
                        )
                    ] = da.from_array(env_rate, self.D.chunks)

                mortality_data.append(transmission)

            else:
                if isinstance(params[mort_type.name], dict):
                    # This is time-based mortality
                    time_based_rate = params[mort_type.name]["time_based_rates"]
                    time_based_index = self.get_counter(
                        species, group, sex, age, time - 1
                    )
                    if time_based_index is None:
                        # There has never been a population of this nature
                        continue
                    try:
                        time_based_mortality = time_based_rate[time_based_index]
                    except IndexError:
                        # The timer has surpassed the range of rates
                        continue
                    if params[mort_type.name]["time_based_std"] is not None:
                        time_based_mortality = np.random.normal(
                            time_based_mortality,
                            params[mort_type.name]["time_based_std"],
                        )
                    time_based_mortality = da.broadcast_to(
                        min(1, max(0, time_based_mortality)),
                        self.D.shape,
                        self.D.chunks,
                    )
                    mortality_data.append(time_based_mortality)
                    output[
                        "{}/mortality/{}".format(param_prefix, mort_type.name)
                    ] = time_based_mortality
                else:
                    mortality_data.append(params[mort_type.name])

        if len(mortality_data) > 0:
            mort_coeff = dsum(mortality_data)
            mort_coeff = da_where(
                (mort_coeff > 1) & (mort_coeff > 0), 1 / mort_coeff, 1
            )
        else:
            mort_coeff = 1

        return mort_coeff

    def _prepare_mortality(
        self,
        species,
        group,
        sex,
        age,
        time,
        param_prefix,
        mort_types,
        mortality_data,
        params,
        output,
        pre_existing_rate=None,
    ):
        """
        Accumulate an iterable of mortality

        :param str species:
        :param str group:
        :param str sex:
        :param int age:
        :param int time:
        :param str param_prefix:
        :param list mort_types:
        :param list mortality_data:
        :param dict params:
        :param dict output:
        :param list pre_existing_rate:
        """
        for mort_type in mort_types:
            if isinstance(params[mort_type.name], dict):
                # This is time-based mortality
                time_based_rate = params[mort_type.name]["time_based_rates"]
                time_based_index = self.get_counter(species, group, sex, age, time - 1)
                if time_based_index is None:
                    # There has never been a population of this nature
                    continue
                try:
                    time_based_mortality = time_based_rate[time_based_index]
                except IndexError:
                    # The timer has surpassed the range of rates
                    continue
                if params[mort_type.name]["time_based_std"] is not None:
                    time_based_mortality = np.random.normal(
                        time_based_mortality, params[mort_type.name]["time_based_std"]
                    )
                time_based_mortality = da.broadcast_to(
                    min(1, max(0, time_based_mortality)), self.D.shape, self.D.chunks
                )
                mortality_data.append(time_based_mortality)
                output[
                    "{}/mortality/{}".format(param_prefix, mort_type.name)
                ] = time_based_mortality
            else:
                mortality_data.append(params[mort_type.name])

        # Mortality must be scaled to not exceed 1
        if len(mortality_data) > 0:
            if pre_existing_rate is not None and len(pre_existing_rate) > 0:
                # The mortality coefficient must account for conversion since it is applied first
                pre_agg_mort = dsum(pre_existing_rate)
                agg_mort = dsum(mortality_data)
                max_mort = da_where(
                    (pre_agg_mort + agg_mort) > 1.0,
                    da.minimum(agg_mort, da.maximum(0.0, 1.0 - pre_agg_mort)),
                    1.0,
                )
                mort_coeff = da_where(
                    agg_mort > max_mort, 1.0 - ((agg_mort - max_mort) / agg_mort), 1.0
                )
            else:
                agg_mort = dsum(mortality_data)
                mort_coeff = da_where(
                    agg_mort > 1.0, 1.0 - ((agg_mort - 1.0) / agg_mort), 1.0
                )
        else:
            mort_coeff = 1

        return mort_coeff

    def _apply_conversion(
        self,
        mort_types,
        mort_coeff,
        mortality_data,
        population,
        flux_prefix,
        time,
        age,
        output,
        _delayed,
        delayed_counter_update,
        dt,
        et,
    ):
        """

        :param mort_types:
        :param mort_coeff:
        :param mortality_data:
        :param population:
        :param flux_prefix:
        :param time:
        :param age:
        :param output:
        :param _delayed:
        :param delayed_counter_update:
        :return:
        """
        mort_fluxes = []
        for mort_type, mort_data in zip(mort_types, mortality_data):
            if dt or et:
                # The coefficient is the mortality rate as a result of direct/environmental transmission calculations
                mort_fluxes.append(mort_coeff * population)
            else:
                mort_fluxes.append(mort_data * mort_coeff * population)
            output[
                "{}/mortality/{}".format(flux_prefix, mort_type.name)
            ] += mort_fluxes[-1]

            # Apply mortality to any recipients (a conversion that serves to simulate disease)
            # The population is added to the model domain for the incremented age and group to
            # use the correct output recipient population time slot
            other_species = mort_type.recipient_species

            if age is not None:
                recipient_age = age + 1
            else:
                recipient_age = None
            recipient_group = self.D.group_from_age(
                other_species.name_key, other_species.sex, recipient_age
            )
            if recipient_group is None:
                max_age = self.D.discrete_ages(
                    other_species.name_key, other_species.sex
                )[-1]
                # No group past this point. If there is a legitimate age, this means that live_past_max is True
                if max_age is not None and age == max_age:
                    # Need to add an age sequentially to the current group
                    self.D.species[other_species.name_key][other_species.sex][
                        other_species.group_key
                    ].max_age += 1
                    recipient_group = other_species.group_key
                else:
                    recipient_age = None

            recipient_instance = self.D.species[other_species.name_key][
                other_species.sex
            ][recipient_group].age_range
            if recipient_age not in recipient_instance:
                raise PopdynError(
                    "Could not apply conversion to recipient species {} because the age {} "
                    "does not exist for the group {} (with range {})".format(
                        other_species.name,
                        recipient_age,
                        recipient_group,
                        other_species.age_range,
                    )
                )

            try:
                output[
                    "{}/mortality/Converted to {}".format(
                        flux_prefix, other_species.name
                    )
                ] += mort_fluxes[-1]
            except KeyError:
                output[
                    "{}/mortality/Converted to {}".format(
                        flux_prefix, other_species.name
                    )
                ] = mort_fluxes[-1]

            other_species_key = "{}/{}/{}/{}/{}".format(
                other_species.name_key,
                other_species.sex,
                recipient_group,
                time,
                recipient_age,
            )

            # This key may not exist yet
            if h5py.__name__ == "h5py":
                kwargs = {"compression": "lzf"}
            else:
                # h5fake
                kwargs = {
                    "sr": self.D.projection,
                    "gt": (self.D.left, self.D.csx, 0, self.D.top, 0, self.D.csy * -1),
                    "nd": [0],
                }

            ds = self.D.file.require_dataset(
                other_species_key,
                shape=self.D.shape,
                dtype=np.float32,
                chunks=self.D.chunks,
                **kwargs
            )
            other_species_data = da.from_array(ds, self.D.chunks) + mort_fluxes[-1]

            # If multiple species are contributing to the recipient, they must be added in a delayed fashion
            try:
                _delayed[other_species_key] += other_species_data
            except KeyError:
                _delayed[other_species_key] = other_species_data
            delayed_counter_update[other_species_key] = da.any(
                _delayed[other_species_key] > 0
            )

        # Reduce the population by the mortality
        for mort_flux in mort_fluxes:
            population -= mort_flux
        return da.maximum(0, population)

    def _apply_mortality(
        self,
        mort_types,
        mort_coeff,
        mortality_data,
        population,
        flux_prefix,
        output,
        _delayed,
    ):
        """
        :param mort_types:
        :param mort_coeff:
        :param mortality_data:
        :param population:
        :param flux_prefix:
        :param time:
        :param age:
        :param output:
        :param _delayed:
        :param delayed_counter_update:
        :return:
        """
        mort_fluxes = []
        for mort_type, mort_data in zip(mort_types, mortality_data):
            mort_fluxes.append(mort_data * mort_coeff * population)
            output[
                "{}/mortality/{}".format(flux_prefix, mort_type.name)
            ] += mort_fluxes[-1]

        # Reduce the population by the mortality
        for mort_flux in mort_fluxes:
            population -= mort_flux
        return da.maximum(0, population)


class Counter(object):
    """Used to update population counters with dask computes"""

    def __init__(self, solver, key):
        self.S = solver
        self.key = key

    def __setitem__(self, _, value):
        """
        Used during computation to update counter values
        """
        self.S.counter[self.key] = np.squeeze(value)


class NanDict(dict):
    """
    Debugging class to tease out nan's
    """

    def __init__(self):
        super(NanDict, self).__init__()

    def __setitem__(self, key, val):
        if np.any(np.isnan(val.compute())):
            raise Exception("The key {} has nans".format(key))

        super(NanDict, self).__setitem__(key, val)
