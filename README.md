# ICLR 2025 Paper Search

This Flask app allows users to search for ICLR 2025 paper submissions using semantic search and BM25 ranking.

## Prerequisites

- Python 3.7 or higher
- Vercel CLI
- OpenAI API key

## Local Development

1. clone this repository:

   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. set up your openai api key as an environment variable:

   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

4. run the flask app locally:

   ```
   python api/index.py
   ```

5. open your browser and go to `http://localhost:5000` to use the app.

## Deploying to Vercel

1. install the vercel cli:

   ```
   npm i -g vercel
   ```

2. login to vercel:

   ```
   vercel login
   ```

3. deploy the app:

   ```
   vercel
   ```

4. set the openai api key as an environment variable on vercel:

   ```
   vercel env add OPENAI_API_KEY
   ```

   enter your openai api key when prompted.

5. redeploy the app to use the new environment variable:

   ```
   vercel --prod
   ```

Your app should now be deployed and accessible via the Vercel URL provided.

## Files

- `api/index.py`: main flask application
- `vercel.json`: vercel configuration file
- `requirements.txt`: python dependencies
- `iclr_2025_submissions.json`: json file containing paper submissions (not included in this repository)
- `embedding_array_fp16.npy`: numpy array of paper embeddings (not included in this repository)

Make sure to include the `iclr_2025_submissions.json` and `embedding_array_fp16.npy` files in your repository or upload them to Vercel separately.

## Note

Ensure that your OpenAI API key has sufficient credits and permissions to create embeddings using the "text-embedding-3-small" model.