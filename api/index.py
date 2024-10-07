from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

with open("iclr_2025_submissions.json", "r") as f:
    papers_dict = json.load(f)

with open("embedding_array_fp16.npy", "rb") as f:
    embedding_array = np.load(f)

print(embedding_array.dtype)
client = OpenAI()


def get_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def get_similar_paper_indices(query_embedding, query, top_k=10):
    # compute the cosine similarity between the query embedding and the embedding array
    cosine_similarities = np.dot(embedding_array, query_embedding)
    # get the indices of the top_k*2 most similar papers
    top_indices = np.argsort(cosine_similarities)[::-1][: top_k * 2]

    # rerank using bm25
    corpus = [
        papers_dict[i]["title"] + " " + papers_dict[i]["abstract"] for i in top_indices
    ]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # combine cosine similarities and bm25 scores
    combined_scores = 0.5 * cosine_similarities[top_indices] + 0.5 * bm25_scores
    reranked_indices = np.argsort(combined_scores)[::-1][:top_k]

    return [top_indices[i] for i in reranked_indices]


def get_similar_papers(query_embedding, query, top_k=10):
    top_indices = get_similar_paper_indices(query_embedding, query, top_k)
    return [papers_dict[index] for index in top_indices]


app = Flask(__name__, template_folder="../templates", static_folder="../static")


@app.route("/search", methods=["POST"])
def search():
    # get the search term from the form
    search_term = request.form.get("search")
    query_embedding = get_embedding(search_term)
    similar_papers = get_similar_papers(query_embedding, search_term)
    return jsonify(similar_papers)


@app.route("/")
def home():
    # render index.html when the root route is accessed
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
