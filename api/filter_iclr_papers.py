import json
import requests


def fetch_notes(limit=1000):
    all_notes = []
    base_url = "https://api2.openreview.net/notes"
    total_count = 11666  # Total number of notes to fetch

    for offset in range(0, total_count, limit):
        params = {
            "content.venueid": "ICLR.cc/2025/Conference/Submission",
            "details": "replyCount,presentation",
            "domain": "ICLR.cc/2025/Conference",
            "limit": limit,
            "offset": offset,
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data at offset {offset}")
            break

        data = response.json()
        notes = data.get("notes", [])
        if not notes:
            break

        for note in notes:
            content = note.get("content", {})
            id_ = note.get("id", "")
            title = content.get("title", {}).get("value", "")
            keywords = content.get("keywords", {}).get("value", [])
            abstract = content.get("abstract", {}).get("value", "")
            note_data = {
                "url": f"https://openreview.net/forum?id={id_}",
                "title": title,
                "keywords": keywords,
                "abstract": abstract,
            }
            all_notes.append(note_data)
        print(f"Fetched {len(notes)} notes at offset {offset}")

    return all_notes


def save_notes_to_json(notes, filename="./iclr_2025_submissions.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def main():
    print("Starting to fetch notes...")
    notes = fetch_notes()
    print(f"Total notes fetched: {len(notes)}")
    print("Saving notes to 'iclr_2025_submissions.json'...")
    save_notes_to_json(notes)
    print("Done.")


main()
