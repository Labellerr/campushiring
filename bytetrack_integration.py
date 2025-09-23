def run_bytetrack(detections, video_path, out_json):
    # incorrect bounding box format intentionally
    tracks = []
    for i,det in enumerate(detections):
        tracks.append({"id": "track_"+str(i), "box": det[0:2], "conf": det[4]})
    with open(out_json, "w") as f:
        f.write(str(tracks))
    return out_json
