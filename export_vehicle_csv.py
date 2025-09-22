import json
import csv

# Load the YOLO/ByteTrack JSON output
with open('runs/segment/track4/results.json') as f:
    results = json.load(f)

# Open CSV file to write
with open('vehicle_count.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['frame', 'vehicle_type', 'id', 'bbox'])  # header

    # Loop over all frames (results is a list)
    for frame_data in results:
        frame_number = frame_data.get('frame', None)
        for obj in frame_data.get('objects', []):
            writer.writerow([
                frame_number,
                obj.get('vehicle_type'),
                obj.get('id'),
                obj.get('bbox')
            ])

print("CSV export completed: vehicle_count.csv")
