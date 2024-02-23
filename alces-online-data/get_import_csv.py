import psycopg2
import boto3

conn = psycopg2.connect(
    database="alces",
    host="alces-production.crwj7evtykeu.us-west-2.rds.amazonaws.com",
    user="aluf_user",
    password="Et1GY83yhR",
    port="15567"
)

cursor = conn.cursor()
indicator_category = 'BA RSF'
indicator_name = 'BA expRSF linear stretch - Calving'

cursor.execute(
    f"""
    SELECT map_rasterfileindicator.year,map_superregion.db_code,map_rasterfileindicator.filename
        FROM map_indicatorcategory
        INNER JOIN map_indicator ON map_indicator.category_id = map_indicatorcategory.id
        INNER JOIN map_superregion ON map_superregion.id=map_indicator.super_region_id
        INNER JOIN map_rasterfileindicator ON map_rasterfileindicator.indicator_id = map_indicator.id
        WHERE map_indicatorcategory.name = '{indicator_category}' AND map_indicator.name = '{indicator_name}'
        ORDER BY map_rasterfileindicator.year;
    """,
)

s3_client = boto3.Session(profile_name='ao').client('s3')

with open('import.csv', 'w') as f:
    for (year, db_code, filename) in cursor.fetchall():
        presigned_get = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'rasters', 'Key': f'indicators_{db_code}_10000_{filename}'},
            ExpiresIn=72*3600
        )
        f.write(f"{year},{presigned_get}\n")
