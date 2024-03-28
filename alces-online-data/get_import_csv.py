import psycopg2
import boto3

conn = psycopg2.connect(
    database="alces",
    host="alces-production.crwj7evtykeu.us-west-2.rds.amazonaws.com",
    user="aluf_user",
    password="Et1GY83yhR",
    port="15567"
)

# # Development scenarios:
# # CBGC High Development Scenario
# # RCP 8.5 induced land cover shift 1.0%
# indicator_names = [
#     'Barren lands',
#     'Sub-polar or polar barren-lichen-moss',
#     'Sub-polar or polar shrubland-lichen-moss',
#     'Temperate or sub-polar shrubland',
#     'Mixed forest',
#     'Sub-polar taiga needleleaf forest',
#     'Temperate or sub-polar broadleaf deciduous forest',
#     'Temperate or sub-polar needleleaf forest',
#     'Waterbodies adjusted',
#     'Watercourse',
#     'Wetland',
#     'CBGC Forest Age',
#     'Moving window 10 km Linear Footprint',
#     'Moving window 10 km Polygonal Footprint',

#     'Snow and Ice',
#     'Sub-polar or polar grassland-lichen-moss',
#     'Temperate or sub-polar grassland',
# ]

# # Barren lands
# # Mixed forest
# # Sub-polar or polar barren-lichen-moss
# # Sub-polar or polar shrubland-lichen-moss
# # Sub-polar taiga needleleaf forest
# # Temperate or sub-polar broadleaf deciduous forest
# # Temperate or sub-polar needleleaf forest
# # Temperate or sub-polar shrubland
# # Watercourse
# # Wetland


# indicators = [
#     {
#         'indicator_name': name,
#         'scenarios': [
#             'CBGC High Development Scenario',
#             # 'historic - empirical or loaded from outside data'
#         ],
#     } for name in indicator_names
# ]

# Climate scenarios:
# CanESM2 RCP 8.5
indicator_names = [
    'BA Calving Slope z',
    'BA Calving Aspect z',
    'BA Summer Min Elev z',
    'BA Summer Slope z',
    'BA Summer Aspect z',
    'BA Summer Min Temp z',
    'BA Summer Evaporation z',
    'BA Summer Precipitation z',
    'BA Fall Min Elev z',
    'BA Fall Slope z',
    'BA Fall Aspect z',
    'BA Fall Average Temperature z',
    'BA Fall Evaporation z',
    'BA Fall Precipitation z',
    'BA Winter Slope z',
    'BA Winter Aspect z',
    'BA Winter Maximum Temperature z',
]
indicators = [
    {
        'indicator_name': name,
        'scenarios': [
            'CanESM2 RCP 8.5',
            'historic - empirical or loaded from outside data'
        ],
    } for name in indicator_names
]

# indicators = [
#     {
#         'indicator_name': name,
#         'scenarios': [
#             'CBGC High Development Scenario',
#             # 'historic - empirical or loaded from outside data'
#         ],
#     } for name in [
#         'BA RSF - Spring Migration',
#         'BA RSF - Calving',
#         'BA expRSF linear stretch - Calving',
#         'BA RSF - Summer',
#         'BA RSF - Fall',
#         'BA RSF - Winter',
#     ]
# ]

s3_client = boto3.Session(profile_name='ao').client('s3')

with open('import.csv', 'w') as f:
    f.write("name,start_date,url\n")
    for indicator in indicators:

        cursor = conn.cursor()
        indicator_name = indicator['indicator_name']

        if 'scenarios' in indicator:
            scenarios_in_term = ', '.join([f"'{s}'" for s in indicator['scenarios']])
            cursor.execute(
                f"""
                SELECT map_rasterfileindicator.year,map_superregion.db_code,map_rasterfileindicator.filename
                    FROM map_indicator
                    INNER JOIN map_superregion ON map_superregion.id=map_indicator.super_region_id
                    INNER JOIN map_rasterfileindicator ON map_rasterfileindicator.indicator_id = map_indicator.id
                    INNER JOIN mapper_scenario ON mapper_scenario.id=map_rasterfileindicator.scenario_id
                    WHERE map_indicator.name = '{indicator_name}' AND mapper_scenario.name IN ({scenarios_in_term})
                    ORDER BY map_rasterfileindicator.year;
                """,
            )
        else:
            cursor.execute(
                f"""
                SELECT map_rasterfileindicator.year,map_superregion.db_code,map_rasterfileindicator.filename
                    FROM map_indicator
                    INNER JOIN map_superregion ON map_superregion.id=map_indicator.super_region_id
                    INNER JOIN map_rasterfileindicator ON map_rasterfileindicator.indicator_id = map_indicator.id
                    WHERE map_indicator.name = '{indicator_name}'
                    ORDER BY map_rasterfileindicator.year;
                """,
            )

        flow_name = indicator.get('flow_name', indicator['indicator_name'])
        print(f'Query results for "{flow_name}": {cursor.rowcount}')
        for (year, db_code, filename) in cursor.fetchall():
            resolution = 1000
            # if flow_name == "Waterbodies adjusted":
            #     resolution = 100
            presigned_get = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': 'rasters', 'Key': f'indicators_{db_code}_{resolution}_{filename}'},
                ExpiresIn=72*3600
            )
            f.write(f"{flow_name},{year},{presigned_get}\n")
