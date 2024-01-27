import csv
import json


def parse_csv_to_json(csv_file_path):
    results = []

    # Ouvre le fichier CSV 
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)

        next(csv_reader, None)

        for row in csv_reader:
            try:
                # Extraction des valeurs de chaque colonne
                index = int(row[0])
                count = int(row[1])
                hate_speech = int(row[2])
                offensive_language = int(row[3])
                neither = int(row[4])
                class_label = int(row[5])
                tweet = row[6]

                # Détermine le type de problème en fonction des votes
                if hate_speech > offensive_language and hate_speech > neither:
                    problem_type = "hate speech"
                elif offensive_language > hate_speech and offensive_language > neither:
                    problem_type = "offensive language"
                else:
                    problem_type = "other"

                result = {"id": index, "message": tweet, "problem_type": problem_type}

                results.append(result)

            except (ValueError, IndexError):
                print(f"Error parsing row: {row}")

    return results


def save_json(output_json, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as jsonfile:
        json.dump(output_json, jsonfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    csv_file_path = "dataset_hate_speech.csv"
    output_file_path = "parsed_dataset.json"

    parsed_data = parse_csv_to_json(csv_file_path)

    if parsed_data:
        save_json(parsed_data, output_file_path)
        print(f"JSON output saved to {output_file_path}")
    else:
        print("No valid data to process.")
