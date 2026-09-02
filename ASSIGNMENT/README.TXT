# ================================================================
# SMART-CITY TRAFFIC AND TRANSPORTATION ANALYTICS
# USING BIG DATA TECHNOLOGIES
# ================================================================
# Technology: Python + PySpark + Spark SQL + K-Means
# Platform: Google Colab
# ================================================================

# =========================
# 1. INSTALL PYSPARK
# =========================

!pip -q install pyspark

# =========================
# 2. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    hour,
    dayofweek,
    avg,
    count,
    when,
    round as spark_round
)

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# =========================
# 3. START SPARK
# =========================

spark = SparkSession.builder \
    .appName("SmartCityTrafficAnalytics") \
    .master("local[*]") \
    .getOrCreate()

print("=" * 70)
print(" SMART-CITY TRAFFIC AND TRANSPORTATION ANALYTICS")
print(" USING BIG DATA TECHNOLOGIES")
print("=" * 70)

print("Spark Version:", spark.version)
print("Spark Session Started Successfully")

# =========================
# 4. CREATE TRAFFIC DATASET
# =========================

np.random.seed(42)

n = 50000

locations = [
    "Anna Nagar",
    "T Nagar",
    "Guindy",
    "Adyar",
    "Velachery",
    "Tambaram",
    "Porur",
    "OMR",
    "Mylapore",
    "Egmore"
]

road_types = [
    "Highway",
    "Main Road",
    "Local Road"
]

weather_conditions = [
    "Sunny",
    "Cloudy",
    "Rainy"
]

signal_statuses = [
    "Green",
    "Red",
    "Yellow"
]

incident_statuses = [
    "Yes",
    "No"
]

traffic_pd = pd.DataFrame({

    "timestamp": pd.date_range(
        start="2026-01-01",
        periods=n,
        freq="15min"
    ),

    "location": np.random.choice(
        locations,
        n
    ),

    "vehicle_count": np.random.poisson(
        100,
        n
    ),

    "avg_speed": np.clip(
        np.random.normal(
            40,
            10,
            n
        ),
        10,
        80
    ),

    "capacity": np.random.choice(
        [100, 120, 150, 180, 200],
        n
    ),

    "traffic_density": np.random.uniform(
        0.30,
        1.20,
        n
    ),

    "travel_time": np.random.uniform(
        5,
        40,
        n
    ),

    "weather_condition": np.random.choice(
        weather_conditions,
        n,
        p=[0.60, 0.25, 0.15]
    ),

    "incident_status": np.random.choice(
        incident_statuses,
        n,
        p=[0.10, 0.90]
    ),

    "road_type": np.random.choice(
        road_types,
        n
    ),

    "signal_status": np.random.choice(
        signal_statuses,
        n
    ),

    "public_transport_count": np.random.poisson(
        15,
        n
    ),

    "occupancy_rate": np.random.uniform(
        0.20,
        1.00,
        n
    )
})

# =========================
# 5. ADD DAY OF WEEK
# =========================

traffic_pd["day_of_week"] = (
    traffic_pd["timestamp"]
    .dt.day_name()
)

# =========================
# 6. ADD PEAK-HOUR TRAFFIC
# =========================

hours = traffic_pd["timestamp"].dt.hour

peak_period = hours.isin([
    7, 8, 9,
    17, 18, 19
])

traffic_pd.loc[
    peak_period,
    "vehicle_count"
] += np.random.poisson(
    60,
    peak_period.sum()
)

traffic_pd.loc[
    peak_period,
    "avg_speed"
] -= np.random.normal(
    8,
    3,
    peak_period.sum()
)

traffic_pd["avg_speed"] = (
    traffic_pd["avg_speed"]
    .clip(8, 80)
)

# Recalculate traffic density
traffic_pd["traffic_density"] = (
    traffic_pd["vehicle_count"] /
    traffic_pd["capacity"]
)

traffic_pd["traffic_density"] = (
    traffic_pd["traffic_density"]
    .clip(0, 1.5)
)

traffic_pd["occupancy_rate"] = (
    traffic_pd["traffic_density"]
    .clip(0, 1)
)

# =========================
# 7. ADD MISSING VALUES
# =========================

missing_indices = np.random.choice(
    traffic_pd.index,
    size=500,
    replace=False
)

traffic_pd.loc[
    missing_indices[:250],
    "avg_speed"
] = np.nan

traffic_pd.loc[
    missing_indices[250:],
    "weather_condition"
] = np.nan

print("\nDataset Generated Successfully")
print("Total Records:", len(traffic_pd))

# =========================
# 8. CREATE PYSPARK DATAFRAME
# =========================

traffic_df = spark.createDataFrame(
    traffic_pd
)

print("\nPySpark DataFrame Created")

# =========================
# 9. DISPLAY SCHEMA
# =========================

print("\nDATASET SCHEMA")
print("-" * 70)

traffic_df.printSchema()

# =========================
# 10. DATASET INFORMATION
# =========================

print("\nDATASET INFORMATION")
print("-" * 70)

print(
    "Number of Records:",
    traffic_df.count()
)

print(
    "Number of Columns:",
    len(traffic_df.columns)
)

# =========================
# 11. SHOW SAMPLE DATA
# =========================

print("\nSAMPLE TRAFFIC DATA")
print("-" * 70)

traffic_df.show(
    10,
    truncate=False
)

# =========================
# 12. MISSING VALUE ANALYSIS
# =========================

print("\nMISSING VALUE ANALYSIS")
print("-" * 70)

for column in traffic_df.columns:

    missing = traffic_df.filter(
        col(column).isNull()
    ).count()

    print(
        f"{column}: {missing}"
    )

# =========================
# 13. REMOVE DUPLICATES
# =========================

before_duplicates = traffic_df.count()

traffic_df = traffic_df.dropDuplicates()

after_duplicates = traffic_df.count()

print("\nDUPLICATE REMOVAL")
print("-" * 70)

print(
    "Before:",
    before_duplicates
)

print(
    "After:",
    after_duplicates
)

print(
    "Duplicates Removed:",
    before_duplicates - after_duplicates
)

# =========================
# 14. HANDLE MISSING VALUES
# =========================

traffic_df = traffic_df.fillna({

    "avg_speed": 40.0,

    "travel_time": 15.0,

    "traffic_density": 0.5,

    "occupancy_rate": 0.5,

    "weather_condition": "Unknown",

    "incident_status": "No",

    "road_type": "Main Road",

    "signal_status": "Green"
})

# =========================
# 15. REMOVE INVALID VALUES
# =========================

traffic_df = traffic_df.filter(

    (col("vehicle_count") >= 0) &
    (col("avg_speed") > 0) &
    (col("capacity") > 0) &
    (col("traffic_density") >= 0) &
    (col("travel_time") > 0) &
    (col("occupancy_rate") >= 0)

)

print("\nDATA CLEANING COMPLETED")

print(
    "Clean Records:",
    traffic_df.count()
)

# =========================
# 16. FEATURE ENGINEERING
# =========================

traffic_df = traffic_df.withColumn(
    "hour",
    hour("timestamp")
)

traffic_df = traffic_df.withColumn(
    "day_number",
    dayofweek("timestamp")
)

traffic_df = traffic_df.withColumn(
    "congestion_ratio",
    spark_round(
        col("vehicle_count") /
        col("capacity"),
        3
    )
)

traffic_df = traffic_df.withColumn(

    "congestion_level",

    when(
        col("congestion_ratio") >= 0.91,
        "High"
    )

    .when(
        col("congestion_ratio") >= 0.61,
        "Medium"
    )

    .otherwise(
        "Low"
    )
)

# =========================
# 17. SHOW PROCESSED DATA
# =========================

print("\nPROCESSED TRAFFIC DATA")
print("-" * 70)

traffic_df.select(
    "timestamp",
    "location",
    "vehicle_count",
    "avg_speed",
    "capacity",
    "traffic_density",
    "congestion_ratio",
    "congestion_level"
).show(
    10,
    truncate=False
)

# =========================
# 18. SPARK SQL
# =========================

traffic_df.createOrReplaceTempView(
    "traffic"
)

print("\nSPARK SQL ANALYSIS")
print("-" * 70)

sql_result = spark.sql("""

SELECT

    location,

    COUNT(*) AS total_records,

    ROUND(
        AVG(vehicle_count),
        2
    ) AS average_vehicle_count,

    ROUND(
        AVG(avg_speed),
        2
    ) AS average_speed,

    ROUND(
        AVG(congestion_ratio),
        3
    ) AS average_congestion

FROM traffic

GROUP BY location

ORDER BY average_congestion DESC

""")

sql_result.show(
    truncate=False
)

# =========================
# 19. HOURLY TRAFFIC ANALYSIS
# =========================

hourly_analysis = (

    traffic_df

    .groupBy("hour")

    .agg(

        spark_round(
            avg("vehicle_count"),
            2
        ).alias(
            "average_vehicle_count"
        ),

        spark_round(
            avg("avg_speed"),
            2
        ).alias(
            "average_speed"
        ),

        spark_round(
            avg("congestion_ratio"),
            3
        ).alias(
            "average_congestion"
        )

    )

    .orderBy("hour")
)

print("\nHOURLY TRAFFIC ANALYSIS")
print("-" * 70)

hourly_analysis.show(
    24
)

# =========================
# 20. PEAK TRAFFIC HOURS
# =========================

peak_hours = (

    hourly_analysis

    .orderBy(
        col(
            "average_vehicle_count"
        ).desc()
    )

    .limit(5)
)

print("\nTOP 5 PEAK TRAFFIC HOURS")
print("-" * 70)

peak_hours.show(
    truncate=False
)

# =========================
# 21. LOCATION-WISE ANALYSIS
# =========================

location_analysis = (

    traffic_df

    .groupBy("location")

    .agg(

        count("*").alias(
            "records"
        ),

        spark_round(
            avg("vehicle_count"),
            2
        ).alias(
            "average_vehicles"
        ),

        spark_round(
            avg("avg_speed"),
            2
        ).alias(
            "average_speed"
        ),

        spark_round(
            avg("congestion_ratio"),
            3
        ).alias(
            "average_congestion"
        )

    )

    .orderBy(
        col(
            "average_congestion"
        ).desc()
    )
)

print("\nLOCATION-WISE TRAFFIC ANALYSIS")
print("-" * 70)

location_analysis.show(
    truncate=False
)

# =========================
# 22. CONGESTION ANALYSIS
# =========================

congestion_analysis = (

    traffic_df

    .groupBy(
        "congestion_level"
    )

    .agg(

        count("*").alias(
            "number_of_records"
        ),

        spark_round(
            avg("vehicle_count"),
            2
        ).alias(
            "average_vehicle_count"
        ),

        spark_round(
            avg("avg_speed"),
            2
        ).alias(
            "average_speed"
        )

    )
)

print("\nTRAFFIC CONGESTION CLASSIFICATION")
print("-" * 70)

congestion_analysis.show(
    truncate=False
)

# =========================
# 23. WEATHER ANALYSIS
# =========================

weather_analysis = (

    traffic_df

    .groupBy(
        "weather_condition"
    )

    .agg(

        count("*").alias(
            "records"
        ),

        spark_round(
            avg("vehicle_count"),
            2
        ).alias(
            "average_vehicles"
        ),

        spark_round(
            avg("avg_speed"),
            2
        ).alias(
            "average_speed"
        ),

        spark_round(
            avg("congestion_ratio"),
            3
        ).alias(
            "average_congestion"
        )

    )
)

print("\nWEATHER-WISE TRAFFIC ANALYSIS")
print("-" * 70)

weather_analysis.show(
    truncate=False
)

# =========================
# 24. INCIDENT ANALYSIS
# =========================

incident_analysis = (

    traffic_df

    .groupBy(
        "incident_status"
    )

    .agg(

        count("*").alias(
            "records"
        ),

        spark_round(
            avg("vehicle_count"),
            2
        ).alias(
            "average_vehicles"
        ),

        spark_round(
            avg("avg_speed"),
            2
        ).alias(
            "average_speed"
        ),

        spark_round(
            avg("congestion_ratio"),
            3
        ).alias(
            "average_congestion"
        )

    )
)

print("\nTRAFFIC INCIDENT ANALYSIS")
print("-" * 70)

incident_analysis.show(
    truncate=False
)

# =========================
# 25. ROAD TYPE ANALYSIS
# =========================

road_analysis = (

    traffic_df

    .groupBy(
        "road_type"
    )

    .agg(

        count("*").alias(
            "records"
        ),

        spark_round(
            avg("vehicle_count"),
            2
        ).alias(
            "average_vehicles"
        ),

        spark_round(
            avg("avg_speed"),
            2
        ).alias(
            "average_speed"
        ),

        spark_round(
            avg("congestion_ratio"),
            3
        ).alias(
            "average_congestion"
        )

    )

    .orderBy(
        col(
            "average_congestion"
        ).desc()
    )
)

print("\nROAD TYPE ANALYSIS")
print("-" * 70)

road_analysis.show(
    truncate=False
)

# =========================
# 26. K-MEANS CLUSTERING
# =========================

features = [

    "vehicle_count",

    "avg_speed",

    "traffic_density",

    "travel_time",

    "occupancy_rate"

]

assembler = VectorAssembler(

    inputCols=features,

    outputCol="raw_features"

)

assembled_df = assembler.transform(
    traffic_df
)

scaler = StandardScaler(

    inputCol="raw_features",

    outputCol="scaled_features",

    withStd=True,

    withMean=True

)

scaler_model = scaler.fit(
    assembled_df
)

scaled_df = scaler_model.transform(
    assembled_df
)

print("\nFEATURE SCALING COMPLETED")

# =========================
# 27. TRAIN K-MEANS MODEL
# =========================

kmeans = KMeans(

    k=3,

    seed=42,

    featuresCol="scaled_features",

    predictionCol="cluster"

)

kmeans_model = kmeans.fit(
    scaled_df
)

clustered_df = kmeans_model.transform(
    scaled_df
)

print("K-MEANS CLUSTERING COMPLETED")

# =========================
# 28. CLUSTER ANALYSIS
# =========================

cluster_analysis = (

    clustered_df

    .groupBy("cluster")

    .agg(

        count("*").alias(
            "records"
        ),

        spark_round(
            avg("vehicle_count"),
            2
        ).alias(
            "average_vehicles"
        ),

        spark_round(
            avg("avg_speed"),
            2
        ).alias(
            "average_speed"
        ),

        spark_round(
            avg("traffic_density"),
            3
        ).alias(
            "average_density"
        ),

        spark_round(
            avg("congestion_ratio"),
            3
        ).alias(
            "average_congestion"
        ),

        spark_round(
            avg("travel_time"),
            2
        ).alias(
            "average_travel_time"
        )

    )

    .orderBy("cluster")
)

print("\nTRAFFIC CLUSTER ANALYSIS")
print("-" * 70)

cluster_analysis.show(
    truncate=False
)

# =========================
# 29. CLUSTER INTERPRETATION
# =========================

cluster_pd = (
    cluster_analysis
    .toPandas()
)

print("\nTRAFFIC PATTERN INTERPRETATION")
print("-" * 70)

for _, row in cluster_pd.iterrows():

    if (
        row["average_congestion"] >= 0.91
        and
        row["average_speed"] < 35
    ):

        pattern = (
            "High Traffic / Low Speed"
        )

    elif (
        row["average_congestion"] >= 0.61
    ):

        pattern = (
            "Moderate Traffic"
        )

    else:

        pattern = (
            "Low Traffic / Higher Speed"
        )

    print(
        "Cluster",
        int(row["cluster"]),
        "=>",
        pattern
    )

# =========================
# 30. CONVERT RESULTS
# =========================

hourly_pd = (
    hourly_analysis
    .toPandas()
)

location_pd = (
    location_analysis
    .toPandas()
)

congestion_pd = (
    congestion_analysis
    .toPandas()
)

weather_pd = (
    weather_analysis
    .toPandas()
)

road_pd = (
    road_analysis
    .toPandas()
)

# =========================
# 31. TRAFFIC VOLUME GRAPH
# =========================

plt.figure(figsize=(10, 5))

plt.plot(

    hourly_pd["hour"],

    hourly_pd[
        "average_vehicle_count"
    ],

    marker="o"

)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Average Vehicle Count"
)

plt.title(
    "Average Traffic Volume by Hour"
)

plt.grid(True)

plt.show()

# =========================
# 32. AVERAGE SPEED GRAPH
# =========================

plt.figure(figsize=(10, 5))

plt.plot(

    hourly_pd["hour"],

    hourly_pd[
        "average_speed"
    ],

    marker="o"

)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Average Speed (km/h)"
)

plt.title(
    "Average Vehicle Speed by Hour"
)

plt.grid(True)

plt.show()

# =========================
# 33. CONGESTION GRAPH
# =========================

plt.figure(figsize=(10, 5))

plt.plot(

    hourly_pd["hour"],

    hourly_pd[
        "average_congestion"
    ],

    marker="o"

)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Average Congestion Ratio"
)

plt.title(
    "Traffic Congestion by Hour"
)

plt.grid(True)

plt.show()

# =========================
# 34. LOCATION TRAFFIC GRAPH
# =========================

plt.figure(figsize=(12, 6))

plt.bar(

    location_pd["location"],

    location_pd[
        "average_vehicles"
    ]

)

plt.xlabel(
    "Location"
)

plt.ylabel(
    "Average Vehicle Count"
)

plt.title(
    "Location-Wise Traffic Volume"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()

# =========================
# 35. CONGESTION DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))

plt.bar(

    congestion_pd[
        "congestion_level"
    ],

    congestion_pd[
        "number_of_records"
    ]

)

plt.xlabel(
    "Congestion Level"
)

plt.ylabel(
    "Number of Records"
)

plt.title(
    "Traffic Congestion Distribution"
)

plt.show()

# =========================
# 36. WEATHER VS TRAFFIC
# =========================

plt.figure(figsize=(8, 5))

plt.bar(

    weather_pd[
        "weather_condition"
    ],

    weather_pd[
        "average_vehicles"
    ]

)

plt.xlabel(
    "Weather Condition"
)

plt.ylabel(
    "Average Vehicle Count"
)

plt.title(
    "Weather Condition vs Traffic Volume"
)

plt.show()

# =========================
# 37. ROAD TYPE VS CONGESTION
# =========================

plt.figure(figsize=(8, 5))

plt.bar(

    road_pd[
        "road_type"
    ],

    road_pd[
        "average_congestion"
    ]

)

plt.xlabel(
    "Road Type"
)

plt.ylabel(
    "Average Congestion Ratio"
)

plt.title(
    "Road Type vs Congestion"
)

plt.show()

# =========================
# 38. VEHICLE COUNT VS SPEED
# =========================

sample_pd = (

    traffic_df

    .select(
        "vehicle_count",
        "avg_speed"
    )

    .sample(
        False,
        0.05,
        seed=42
    )

    .toPandas()
)

plt.figure(figsize=(9, 5))

plt.scatter(

    sample_pd[
        "vehicle_count"
    ],

    sample_pd[
        "avg_speed"
    ],

    alpha=0.5

)

plt.xlabel(
    "Vehicle Count"
)

plt.ylabel(
    "Average Speed (km/h)"
)

plt.title(
    "Vehicle Count vs Average Speed"
)

plt.grid(True)

plt.show()

# =========================
# 39. CORRELATION ANALYSIS
# =========================

correlation_data = (

    traffic_df

    .select(

        "vehicle_count",

        "avg_speed",

        "traffic_density",

        "travel_time",

        "occupancy_rate",

        "congestion_ratio"

    )

    .toPandas()
)

correlation_matrix = (
    correlation_data.corr()
)

print("\nCORRELATION MATRIX")
print("-" * 70)

print(
    correlation_matrix.round(3)
)

# =========================
# 40. CORRELATION GRAPH
# =========================

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(

    range(
        len(
            correlation_matrix.columns
        )
    ),

    correlation_matrix.columns,

    rotation=45,

    ha="right"

)

plt.yticks(

    range(
        len(
            correlation_matrix.columns
        )
    ),

    correlation_matrix.columns

)

plt.title(
    "Traffic Feature Correlation Matrix"
)

plt.tight_layout()

plt.show()

# =========================
# 41. MOST CONGESTED LOCATIONS
# =========================

most_congested = (

    location_analysis

    .orderBy(

        col(
            "average_congestion"
        ).desc()

    )

    .limit(5)
)

print("\nTOP 5 MOST CONGESTED LOCATIONS")
print("-" * 70)

most_congested.show(
    truncate=False
)

# =========================
# 42. HIGHEST TRAFFIC LOCATIONS
# =========================

highest_traffic = (

    location_analysis

    .orderBy(

        col(
            "average_vehicles"
        ).desc()

    )

    .limit(5)
)

print("\nTOP 5 HIGHEST TRAFFIC LOCATIONS")
print("-" * 70)

highest_traffic.show(
    truncate=False
)

# =========================
# 43. FASTEST LOCATIONS
# =========================

fastest_locations = (

    location_analysis

    .orderBy(

        col(
            "average_speed"
        ).desc()

    )

    .limit(5)
)

print(
    "\nTOP 5 LOCATIONS WITH HIGHEST AVERAGE SPEED"
)

print("-" * 70)

fastest_locations.show(
    truncate=False
)

# =========================
# 44. AUTOMATIC RECOMMENDATIONS
# =========================

top_location = (

    location_analysis

    .orderBy(

        col(
            "average_congestion"
        ).desc()

    )

    .first()
)

top_hour = (

    hourly_analysis

    .orderBy(

        col(
            "average_vehicle_count"
        ).desc()

    )

    .first()
)

print("\n")
print("=" * 70)
print(" SMART-CITY TRAFFIC RECOMMENDATIONS")
print("=" * 70)

print(
    "\nMost Congested Location:",
    top_location["location"]
)

print(
    "Average Congestion Ratio:",
    top_location["average_congestion"]
)

print(
    "\nPeak Traffic Hour:",
    str(top_hour["hour"]) + ":00"
)

print(
    "Average Vehicle Count:",
    top_hour[
        "average_vehicle_count"
    ]
)

print(
    "\nRECOMMENDATIONS:"
)

print(
    "1. Optimize traffic signals during peak traffic hours."
)

print(
    "2. Increase traffic monitoring at highly congested locations."
)

print(
    "3. Encourage public transportation during peak periods."
)

print(
    "4. Use real-time traffic sensors for continuous monitoring."
)

print(
    "5. Develop dynamic traffic routing for congested roads."
)

print(
    "6. Use historical traffic patterns for transportation planning."
)

print(
    "7. Investigate traffic incidents that contribute to congestion."
)

# =========================
# 45. SAVE CLEANED DATA
# =========================

output_pd = (
    traffic_df
    .toPandas()
)

output_pd.to_csv(

    "/content/cleaned_smart_city_traffic.csv",

    index=False

)

# =========================
# 46. SAVE ANALYTICAL RESULTS
# =========================

hourly_pd.to_csv(
    "/content/hourly_traffic_analysis.csv",
    index=False
)

location_pd.to_csv(
    "/content/location_traffic_analysis.csv",
    index=False
)

congestion_pd.to_csv(
    "/content/congestion_analysis.csv",
    index=False
)

cluster_pd.to_csv(
    "/content/traffic_cluster_analysis.csv",
    index=False
)

# =========================
# 47. FINAL PROJECT SUMMARY
# =========================

print("\n")
print("=" * 70)
print(" FINAL PROJECT SUMMARY")
print("=" * 70)

print(
    "\nProject:",
    "Smart-City Traffic and Transportation Analytics"
)

print(
    "Technology:",
    "PySpark + Spark SQL + K-Means"
)

print(
    "Records Processed:",
    traffic_df.count()
)

print(
    "Locations:",
    traffic_df
    .select("location")
    .distinct()
    .count()
)

print(
    "Traffic Clusters:",
    clustered_df
    .select("cluster")
    .distinct()
    .count()
)

print(
    "\nModules Completed:"
)

print("✓ Data Collection")
print("✓ Data Preprocessing")
print("✓ Data Cleaning")
print("✓ Feature Engineering")
print("✓ Big Data Processing")
print("✓ Spark SQL Analysis")
print("✓ Traffic Analytics")
print("✓ Congestion Analysis")
print("✓ Peak Hour Analysis")
print("✓ Location Analysis")
print("✓ Weather Analysis")
print("✓ Incident Analysis")
print("✓ K-Means Clustering")
print("✓ Data Visualization")
print("✓ Correlation Analysis")
print("✓ Smart-City Recommendations")
print("✓ CSV Export")

print("\nOutput files saved in /content/")
print("1. cleaned_smart_city_traffic.csv")
print("2. hourly_traffic_analysis.csv")
print("3. location_traffic_analysis.csv")
print("4. congestion_analysis.csv")
print("5. traffic_cluster_analysis.csv")

print("\n")
print("=" * 70)
print(" PROJECT EXECUTED SUCCESSFULLY")
print("=" * 70)

# ================================================================
# END OF PROJECT
# ================================================================
