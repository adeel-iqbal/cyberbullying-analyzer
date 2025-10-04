import re

def clean_tweet(text):
    """
    Clean the tweet text: lowercase, remove mentions, links, hashtags, punctuation, extra spaces.
    """
    text = text.lower()
    text = re.sub(r'@\w+', '', text)                # remove mentions
    text = re.sub(r'http\S+|www\S+', '', text)     # remove URLs
    text = re.sub(r'#(\w+)', r'\1', text)          # remove '#' but keep word
    text = re.sub(r'[^a-z\s]', '', text)           # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces
    return text
