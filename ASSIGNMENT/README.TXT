print("\n")
print("=" * 60)
print(" SMART-CITY TRAFFIC AND TRANSPORTATION ANALYTICS")
print("=" * 60)

print("\nPROJECT SUMMARY")
print("-" * 60)

print(
    "Total Records Processed:",
    traffic_df.count()
)

print(
    "Total Locations:",
    traffic_df.select("location").distinct().count()
)

print(
    "Total Traffic Clusters:",
    clustered_df.select("cluster").distinct().count()
)

print("\nTOP CONGESTED LOCATIONS")
most_congested.show(truncate=False)

print("\nPEAK TRAFFIC HOURS")
peak_hours.show(truncate=False)

print("\nCONGESTION LEVELS")
congestion_analysis.show(truncate=False)

print("\nTRAFFIC CLUSTERS")
cluster_analysis.show(truncate=False)

print("\nPROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
