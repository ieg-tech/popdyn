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
# indicators = [
#     {
#         'indicator_name': name,
#         'scenarios': [
#             'CBGC High Development Scenario',
#             'historic - empirical or loaded from outside data'
#         ],
#     } for name in indicator_names
# ]

# # Climate scenarios:
# # CanESM2 RCP 8.5
# indicator_names = [
#     # # BA
#     # 'BA Calving Slope z',
#     # 'BA Calving Aspect z',
#     # 'BA Summer Min Elev z',
#     # 'BA Summer Slope z',
#     # 'BA Summer Aspect z',
#     # 'BA Summer Min Temp z',
#     # 'BA Summer Evaporation z',
#     # 'BA Summer Precipitation z',
#     # 'BA Fall Min Elev z',
#     # 'BA Fall Slope z',
#     # 'BA Fall Aspect z',
#     # 'BA Fall Average Temperature z',
#     # 'BA Fall Evaporation z',
#     # 'BA Fall Precipitation z',
#     # 'BA Winter Slope z',
#     # 'BA Winter Aspect z',
#     # 'BA Winter Maximum Temperature z',

#     # # BNE
#     # 'BNE Calving Aspect z',
#     # 'BNE Calving Evaporation z',
#     # 'BNE Calving Max Temp z',
#     # 'BNE Calving Min Elev z',
#     # 'BNE Calving Precipitation z',
#     # 'BNE Calving Slope z',
#     # 'BNE Fall Aspect z',
#     # 'BNE Fall Evaporation z',
#     # 'BNE Fall Mean Temp z',
#     # 'BNE Fall Min Elev z',
#     # 'BNE Fall Precipitation z',
#     # 'BNE Fall Slope z',
#     # 'BNE Spr_Mig Aspect z',
#     # 'BNE Spr_Mig Evaporation z',
#     # 'BNE Spr_Mig MeanTemp z',
#     # 'BNE Spr_Mig Min Elev z',
#     # 'BNE Spr_Mig Precipitation z',
#     # 'BNE Spr_Mig Slope z',
#     # 'BNE Summer Aspect z',
#     # 'BNE Summer Evaporation z',
#     # 'BNE Summer Max Elev z',
#     # 'BNE Summer Max Temp z',
#     # 'BNE Summer Precipitation z',
#     # 'BNE Summer Slope z',
#     # 'BNE Winter Aspect z',
#     # 'BNE Winter Mean Elev z',
#     # 'BNE Winter Mean Temp z',
#     # 'BNE Winter Precipitation z',
#     # 'BNE Winter Slope z',

#     # # BNW
#     # 'BNW Calving Evaporation z',
#     # 'BNW Calving Max Temp z',
#     # 'BNW Calving Mean Temp z',
#     # 'BNW Calving Precipitation z',
#     # 'BNW Fall Aspect z',
#     # 'BNW Fall Evaporation z',
#     # 'BNW Fall Max Temp z',
#     # 'BNW Fall Min Elev z',
#     # 'BNW Fall Precipitation z',
#     # 'BNW Fall Slope z',
#     # 'BNW Winter Aspect z',
#     # 'BNW Winter Max Elev z',
#     # 'BNW Winter Min Temp z',
#     # 'BNW Winter Precipitation z',
#     # 'BNW Winter Slope z',

#     # # CBA
#     # 'CBA Calving Mean Temp z',
#     # 'CBA Fall Aspect z',
#     # 'CBA Fall Max Temp z',
#     # 'CBA Fall Precipitation z',
#     # 'CBA Fall Slope z',
#     # 'CBA Spr_Mig Aspect z',
#     # 'CBA Spr_Mig Max Temp z',
#     # 'CBA Spr_Mig Slope z',
#     # 'CBA Summer Max Temp z',
#     # 'CBA Winter Aspect z',
#     # 'CBA Winter Mean Temp z',
#     # 'CBA Winter Precipitation z',
#     # 'CBA Winter Slope z',

#     # TP
#     'TP Calving Aspect z',
#     'TP Calving Evaporation z',
#     'TP Calving Max Elev z',
#     'TP Calving Mean Temp z',
#     'TP Calving Precipitation z',
#     'TP Calving Slope z',
#     'TP Fall Evaporation z',
#     'TP Fall Max Temp z',
#     'TP Fall Precipitation z',
#     'TP Spr_Mig Evaporation z',
#     'TP Spr_Mig Max Temp z',
#     'TP Spr_Mig Precipitation z',
#     'TP Summer Aspect z',
#     'TP Summer Evaporation z',
#     'TP Summer Max Elev z',
#     'TP Summer Max Temp z',
#     'TP Summer Precipitation z',
#     'TP Summer Slope z',
#     'TP Winter Aspect z',
#     'TP Winter Mean Temp z',
#     'TP Winter Precipitation z',
#     'TP Winter Slope z',
# ]
# indicators = [
#     {
#         'indicator_name': name,
#         'scenarios': [
#             'CanESM2 RCP 8.5',
#             'historic - empirical or loaded from outside data'
#         ],
#     } for name in indicator_names
# ]

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
# # ]


# # Development scenarios:
# # CBGC High Development Scenario
# # RCP 8.5 induced land cover shift 1.0%
# indicator_names = [
#     # 'BA initial spring migration noncalf population',
#     # 'BNE initial spring migration adult population'
#     # 'BNW initial spring migration adult population'
#     # 'CBA initial spring migration adult population'
#     'TP initial spring migration noncalf population'
# ]
# indicators = [
#     {
#         'indicator_name': name,
#     } for name in indicator_names
# ]


indicator_names = [
    'Change in fall snow depth relative to 2010s',
    'Change in June average temperature relative to 2010s',
    'Change in June precipitation relative to 2010s',
    'Change in September temperature relative to 2010s',
    'Change in October temperature relative to 2010s',
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
            if 'initial spring migration' in flow_name:
                resolution = 10000
            # if flow_name == "Waterbodies adjusted":
            #     resolution = 100
            presigned_get = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': 'rasters', 'Key': f'indicators_{db_code}_{resolution}_{filename}'},
                ExpiresIn=7*24*3600
            )
            f.write(f"{flow_name},{year},{presigned_get}\n")
