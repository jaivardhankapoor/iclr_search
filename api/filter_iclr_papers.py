# %%
import json


# def fetch_notes(limit=1000):
#     all_notes = []
#     base_url = "https://api2.openreview.net/notes"
#     total_count = 11666  # Total number of notes to fetch

#     for offset in range(0, total_count, limit):
#         params = {
#             "content.venueid": "ICLR.cc/2025/Conference/Submission",
#             "details": "replyCount,presentation",
#             "domain": "ICLR.cc/2025/Conference",
#             "limit": limit,
#             "offset": offset,
#         }
#         response = requests.get(base_url, params=params)
#         if response.status_code != 200:
#             print(f"Failed to fetch data at offset {offset}")
#             break

#         data = response.json()
#         notes = data.get("notes", [])
#         if not notes:
#             break

#         for note in notes:
#             content = note.get("content", {})
#             id_ = note.get("id", "")
#             title = content.get("title", {}).get("value", "")
#             keywords = content.get("keywords", {}).get("value", [])
#             abstract = content.get("abstract", {}).get("value", "")
#             note_data = {
#                 "url": f"https://openreview.net/forum?id={id_}",
#                 "title": title,
#                 "keywords": keywords,
#                 "abstract": abstract,
#             }
#             all_notes.append(note_data)
#         print(f"Fetched {len(notes)} notes at offset {offset}")

#     return all_notes


# def save_notes_to_json(notes, filename="./iclr_2025_submissions.json"):
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(notes, f, ensure_ascii=False, indent=2)


# def main():
#     print("Starting to fetch notes...")
#     notes = fetch_notes()
#     print(f"Total notes fetched: {len(notes)}")
#     print("Saving notes to 'iclr_2025_submissions.json'...")
#     save_notes_to_json(notes)
#     print("Done.")


# main()

# a = input(":dfshdfb")
# %%
# if __name__ == '__main__':
#     main()

# %%


# %%

import ell

# Initialize ell (logs and verbose)
ell.init(store="./logdir", autocommit=True, verbose=True)
# %%
# JSON list of papers (input sample)

with open("/mnt/iclr_2025_submissions.json", "r") as f:
    data = json.load(f)


papers = data
# print(papers)


# Define the ell function to analyze abstracts
@ell.simple(model="gpt-4o-mini")
def check_uncertainty_estimation(abstract: str) -> str:
    """
    You are an expert in machine learning and diffusion models.
    Your job is to determine if the following paper's abstract discusses uncertainty estimation in diffusion models.
    Answer 'Yes' if it does, otherwise answer 'No'.
    """
    return f"Does this paper's abstract discuss uncertainty estimation in diffusion models?\n\n{abstract}\n\nAnswer with 'Yes' or 'No'."


# Function to filter papers mentioning diffusion and check for uncertainty estimation
def filter_and_check_papers(papers):
    results = []
    for paper in papers:
        # print(paper['abstract'])
        if "diffusion" in paper["abstract"].lower():  # Check for 'diffusion' in abstract
            # print(paper["abstract"])
            response = check_uncertainty_estimation(paper["abstract"])  # Ask LLM to check for uncertainty estimation
            if response.strip().lower() == "yes":  # Check for explicit 'yes'
                results.append({"title": paper["title"], "abstract": paper["abstract"], "mentions_uncertainty": True})
    return results


# %%
# Execute the function
filtered_papers = filter_and_check_papers(papers)
