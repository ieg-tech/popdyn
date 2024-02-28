import psycopg2
import boto3

conn = psycopg2.connect(
    database="alces",
    host="alces-production.crwj7evtykeu.us-west-2.rds.amazonaws.com",
    user="aluf_user",
    password="Et1GY83yhR",
    port="15567"
)


indicator_category = 'BA RSF'

indicators = [
    {
        'indicator_name': 'BAH seasonal cow mortality with climate change impact',
    },
    {
        'indicator_name': 'BAH fecundity with climate change',
    },
    {
        'indicator_name': 'BAH 300 adult harvest',
    },
    {
        'indicator_name': 'BAH 300 bull harvest',
    },
    {
        'indicator_name': 'BA initial spring migration noncalf population',
    },
]
for season in ['Calving', 'Fall', 'Spring Migration', 'Summer', 'Winter']:
    for scenario in ['CBGC High Development Scenario', 'RCP 8.5 induced land cover shift 1.0%']:
        indicators.append({
            'indicator_name': f'BA expRSF linear stretch - {season}',
            'scenarios': ['historic - empirical or loaded from outside data', scenario],  # need to include "historic - empirical or loaded from outside data"
            'flow_name': f'BA expRSF linear stretch - {season} - {scenario}',
        })

    # indicators.append({
    #     'indicator_name': f'BA expRSF linear stretch - {season}',
    #     'flow_name': f'BA expRSF linear stretch - {season} - {scenario}',
    # })

# indicator_names = [
#     'BA expRSF linear stretch - Calving##historic - empirical or loaded from outside data;CBGC High Development Scenario',
#     'BA expRSF linear stretch - Calving##historic - empirical or loaded from outside data;RCP 8.5 induced land cover shift 1.0%',
#     'BA expRSF linear stretch - Fall',
#     'BA expRSF linear stretch - Spring Migration',
#     'BA expRSF linear stretch - Summer',
#     'BA expRSF linear stretch - Winter',
#     'BA expRSF linear stretch - Calving constant',
#     'BA expRSF linear stretch - Fall constant',
#     'BA expRSF linear stretch - Spring Migration constant',
#     'BA expRSF linear stretch - Summer constant',
#     'BA expRSF linear stretch - Winter constant',
#     'BAH seasonal cow mortality with climate change impact',
#     'BAH fecundity with climate change',
#     'BAH 300 adult harvest',
#     'BAH 300 bull harvest',
#     'BA initial spring migration noncalf population',
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
            presigned_get = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': 'rasters', 'Key': f'indicators_{db_code}_10000_{filename}'},
                ExpiresIn=72*3600
            )
            f.write(f"{flow_name},{year},{presigned_get}\n")
