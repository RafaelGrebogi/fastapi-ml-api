import json

def prepareResultJson2db(predictions, output_path, sampling_rate_hz=100):
    total_samples = len(predictions)
    normal_label = "Normal Walk"
    limping_label = "Limping"

    normal_count = predictions.count(normal_label)
    limping_count = predictions.count(limping_label)

    # Percentages
    normal_percent = round(100 * normal_count / total_samples, 2)
    limping_percent = round(100 * limping_count / total_samples, 2)

    # Find limping onset
    try:
        limping_onset_sample = predictions.index(limping_label)
        limping_onset_percent = round(100 * limping_onset_sample / total_samples, 2)
        limping_onset_time_sec = round(limping_onset_sample / sampling_rate_hz, 2)
    except ValueError:
        limping_onset_sample = None
        limping_onset_percent = None
        limping_onset_time_sec = None

    # Limping streak analysis
    streaks = []
    current_streak = 0
    for pred in predictions:
        if pred == limping_label:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    longest_limping_streak = max(streaks) if streaks else 0
    average_limping_streak = round(sum(streaks) / len(streaks), 2) if streaks else 0.0

    # Duration
    duration_sec = round(total_samples / sampling_rate_hz, 2)

    # Final JSON structure
    result_data = {
        "predictions": predictions,
        "stats": {
            "total_samples": total_samples,
            "sampling_rate_hz": sampling_rate_hz,
            "duration_seconds": duration_sec,
            "normal_walk_percent": normal_percent,
            "limping_percent": limping_percent,
            "limping_count": limping_count,
            "longest_limping_streak": longest_limping_streak,
            "average_limping_streak": average_limping_streak,
            "limping_onset_sample": limping_onset_sample,
            "limping_onset_percent": limping_onset_percent,
            "limping_onset_time_sec": limping_onset_time_sec
        }
    }

    # Write to file
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    return result_data  # In case you want to reuse the dict in memory
