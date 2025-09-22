import json
import csv

# Open JSON file
with open('results.json') as f:
    results = json.load(f)

# Open CSV for writing
with open('vehicle_count.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['frame', 'vehicle_type', 'id', 'bbox'])  # header

    for frame_data in results:  # iterate frames
        frame_id = frame_data['frame_id']
        for obj in frame_data['objects']:
            vehicle_type = obj['category']
            track_id = obj['id']
            bbox = obj['bbox']
            writer.writerow([frame_id, vehicle_type, track_id, bbox])

print("CSV export completed: vehicle_count.csv")
