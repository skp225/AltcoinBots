import feedparser
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import time
from textblob import TextBlob
import os
import glob
import re

# Telegram configuration
TELEGRAM_BOT_TOKEN = ''
TELEGRAM_CHAT_ID = ''

# Prediction data directory
PREDICTION_DATA_DIR = r"C:\Users\user\Desktop\Curl Gecko\AltCoinResearch\prediction_data"

def get_latest_prediction_file():
    """Get the most recent prediction file from the prediction data directory"""
    try:
        # Get all prediction files
        prediction_files = glob.glob(os.path.join(PREDICTION_DATA_DIR, "crypto_predictions_batch*.json"))
        
        if not prediction_files:
            print("No prediction files found")
            return None
        
        # Sort by modification time (newest first)
        latest_file = max(prediction_files, key=os.path.getmtime)
        print(f"Using prediction file: {latest_file}")
        return latest_file
    except Exception as e:
        print(f"Error getting latest prediction file: {str(e)}")
        return None

def load_predicted_cryptocurrencies():
    """Load predicted cryptocurrencies from the latest prediction file"""
    predicted_cryptos = {}
    
    latest_file = get_latest_prediction_file()
    if not latest_file:
        return predicted_cryptos
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
            
        # Extract unique cryptocurrencies from the prediction data
        for entry in prediction_data.get("Brainlet", []):
            crypto = entry.get("cryptocurrency")
            if crypto and crypto not in predicted_cryptos:
                predicted_cryptos[crypto] = True
                
        print(f"Loaded {len(predicted_cryptos)} predicted cryptocurrencies")
        return predicted_cryptos
    except Exception as e:
        print(f"Error loading predicted cryptocurrencies: {str(e)}")
        return {}

# RSS Feed URLs
rss_urls = [
    "https://cryptopanic.com/news/all",
    "https://www.coindesk.com/arc/outboundfeeds/rss",
    "https://cointelegraph.com/rss/tag/altcoin",
    "https://altcoininvestor.com/latest/rss/",
    "https://dailycoin.com/altcoins/feed/",
    "https://www.investopedia.com/altcoins-5225935",
    "https://beincrypto.com/altcoin-news/feed/",
    "https://cryptocoin.news/category/news/altcoin/feed/",
    "https://www.altcoinbuzz.io/feed/",
    "https://cryptonews.com/news/altcoin-news/feed/",
    "https://www.reddit.com/r/CryptoCurrency.rss",
    "https://www.reddit.com/r/solana.rss"
]

# Known project names to look for
known_projects = {
    'Bitcoin': ['BTC', 'Bitcoin', 'BTCUSD'],
    'Ethereum': ['ETH', 'Ethereum', 'ETHUSD'],
    'Binance Coin': ['BNB', 'Binance Coin'],
    'Cardano': ['ADA', 'Cardano'],
    'Solana': ['SOL', 'Solana'],
    'Ripple': ['XRP', 'Ripple'],
    'Polkadot': ['DOT', 'Polkadot'],
    'Stellar': ['XLM', 'Stellar'],
    'Chainlink': ['LINK', 'Chainlink'],
    'Dogecoin': ['DOGE', 'Dogecoin'],
    'Shiba Inu': ['SHIB', 'Shiba Inu'],
    'Litecoin': ['LTC', 'Litecoin'],
    'Monero': ['XMR', 'Monero'],
    'EOS': ['EOS'],
    'TRON': ['TRX', 'TRON'],
    'NEO': ['NEO'],
    'IOTA': ['IOTA'],
    'DASH': ['DASH', 'Dash'],
    'Zcash': ['ZEC', 'Zcash'],
    'OmiseGo': ['OMG', 'OmiseGo'],
    'Basic Attention Token': ['BAT', 'Basic Attention Token'],
    'Ethereum Classic': ['ETC', 'Ethereum Classic']
}

def send_telegram_message(message):
    """Send a message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            print(f"Message sent to Telegram successfully")
            return True
        else:
            print(f"Failed to send Telegram message: {response.text}")
            return False
    except Exception as e:
        print(f"Error sending Telegram message: {str(e)}")
        return False

def extract_project_name(title, description, predicted_cryptos=None):
    """Try to identify the project name from title and description"""
    text = f"{title} {description}"
    text_lower = text.lower()
    
    # Check for predicted cryptocurrencies first (if available)
    if predicted_cryptos:
        for crypto in predicted_cryptos:
            if crypto.lower() in text_lower:
                return f"Predicted: {crypto}"
    
    # Check for exact matches in known projects
    for project, keywords in known_projects.items():
        for keyword in keywords:
            if keyword.lower() in text_lower or project.lower() in text_lower:
                return project
    
    # If no known project found, try to detect unknown projects
    
    # Pattern 1: Look for potential crypto symbols (3-5 uppercase letters)
    crypto_symbols = re.findall(r'\b[A-Z]{3,5}\b', text)
    
    # Pattern 2: Look for "X coin", "X token", "X crypto" patterns
    coin_patterns = re.findall(r'(\w+)(?:\s+)(coin|token|crypto|blockchain|protocol)', text_lower)
    
    # Pattern 3: Look for capitalized project names
    potential_names = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
    
    # Combine all potential project names
    potential_projects = []
    
    # Add crypto symbols
    for symbol in crypto_symbols:
        # Exclude common non-crypto acronyms
        if symbol not in ['USD', 'EUR', 'JPY', 'GBP', 'THE', 'AND', 'FOR', 'NFT', 'CEO', 'CTO', 'ICO']:
            potential_projects.append(symbol)
    
    # Add coin/token pattern matches
    for match in coin_patterns:
        if len(match[0]) > 2:  # Avoid very short names
            potential_projects.append(f"{match[0].capitalize()} {match[1]}")
    
    # Add capitalized names that might be projects
    for name in potential_names:
        if len(name) > 3 and name not in ['The', 'This', 'That', 'These', 'Those', 'There']:
            potential_projects.append(name)
    
    # If we found potential projects, return the first one
    if potential_projects:
        return f"*Unknown: {potential_projects[0]}*"
    
    # If no match found, return "General"
    return "*General*"

def fetch_rss(url):
    """Fetch and parse RSS feed"""
    try:
        feed = feedparser.parse(url)
        return feed.entries
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return []

def extract_article_data(entry, predicted_cryptos=None):
    """Extract relevant data from RSS entry"""
    title = entry.get('title', '')
    description = entry.get('description', '')
    link = entry.get('link', '')
    
    # Special handling for Reddit content
    source = entry.get('source', {}).get('title', '')
    if 'reddit.com' in link:
        # Set source as Reddit/subreddit
        source = "Reddit/r/CryptoCurrency"
        
        # Clean up Reddit HTML content if present
        if description and '<' in description:
            try:
                soup = BeautifulSoup(description, 'html.parser')
                description = soup.get_text(separator=' ', strip=True)
            except Exception as e:
                print(f"Error parsing Reddit HTML: {str(e)}")
    
    project_name = extract_project_name(title, description, predicted_cryptos)
    
    data = {
        'title': title,
        'link': link,
        'description': description,
        'published': entry.get('published', ''),
        'source': source,
        'project_name': project_name,
        'sentiment': analyze_sentiment(description + ' ' + title)
    }
    return data

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    analysis = TextBlob(text)
    # TextBlob provides polarity (between -1 and 1) and subjectivity (between 0 and 1)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity,
        'pos': 1 if analysis.sentiment.polarity > 0 else 0,
        'neu': 1 if analysis.sentiment.polarity == 0 else 0,
        'neg': 1 if analysis.sentiment.polarity < 0 else 0
    }

def save_to_json(data, filename):
    """Save data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def is_duplicate(article, existing_articles, threshold=0.8):
    """Check if an article is a duplicate of any existing article"""
    # Check for exact URL match
    if any(article['link'] == existing['link'] for existing in existing_articles):
        return True
    
    # Check for title similarity
    title = article['title'].lower()
    for existing in existing_articles:
        existing_title = existing['title'].lower()
        
        # Simple similarity check - shared words ratio
        if title and existing_title:
            title_words = set(title.split())
            existing_words = set(existing_title.split())
            
            if len(title_words) == 0 or len(existing_words) == 0:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(title_words.intersection(existing_words))
            union = len(title_words.union(existing_words))
            similarity = intersection / union
            
            if similarity > threshold:
                return True
    
    return False

def format_telegram_trending_message(trending_summary):
    """Format trending projects data for Telegram message"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = f"*📊 Crypto Trending Projects Update*\n"
    message += f"*Time:* {current_time}\n"
    message += f"*Total Articles:* {trending_summary['total_articles']}\n\n"
    
    # Separate predicted projects from regular projects
    predicted_projects = []
    regular_projects = []
    
    for project in trending_summary['trending_projects'][:20]:
        if project['name'].startswith('Predicted:'):
            predicted_projects.append(project)
        else:
            regular_projects.append(project)
    
    # Show top 10 regular projects
    message += "*Top 10 Trending Projects:*\n"
    for i, project in enumerate(regular_projects[:10], 1):
        # Add emoji based on position
        if i == 1:
            emoji = "🥇"
        elif i == 2:
            emoji = "🥈"
        elif i == 3:
            emoji = "🥉"
        else:
            emoji = "📌"
            
        message += f"{emoji} {i}. *{project['name']}*: {project['mentions']} mentions\n"
    
    # Show predicted projects if any
    if predicted_projects:
        message += "\n*🔮 Predicted Projects in News:*\n"
        for i, project in enumerate(predicted_projects[:5], 1):
            # Extract the actual name from "Predicted: X"
            name = project['name'].replace('Predicted: ', '')
            message += f"🔮 {i}. *{name}*: {project['mentions']} mentions\n"
    
    # Add sentiment analysis if available
    if 'sentiment_summary' in trending_summary:
        message += "\n*Overall Sentiment:*\n"
        sentiment = trending_summary['sentiment_summary']
        
        # Add sentiment emoji
        if sentiment['average_polarity'] > 0.2:
            sentiment_emoji = "🟢 Bullish"
        elif sentiment['average_polarity'] < -0.2:
            sentiment_emoji = "🔴 Bearish"
        else:
            sentiment_emoji = "⚪ Neutral"
            
        message += f"{sentiment_emoji} (Score: {sentiment['average_polarity']:.2f})\n"
    
    # Add a few notable articles if available
    if len(trending_summary.get('notable_articles', [])) > 0:
        message += "\n*Notable Articles:*\n"
        for i, article in enumerate(trending_summary['notable_articles'][:3], 1):
            message += f"{i}. [{article['title']}]({article['link']})\n"
    
    return message

def main():
    print("Starting RSS scraper...")
    
    # Load predicted cryptocurrencies from the latest prediction file
    print("Loading predicted cryptocurrencies...")
    predicted_cryptos = load_predicted_cryptocurrencies()
    predicted_crypto_list = list(predicted_cryptos.keys()) if predicted_cryptos else []
    if predicted_crypto_list:
        print(f"Loaded {len(predicted_crypto_list)} predicted cryptocurrencies: {', '.join(predicted_crypto_list[:5])}...")
    else:
        print("No predicted cryptocurrencies found or failed to load predictions")
    
    # Remove duplicate URLs
    unique_urls = list(dict.fromkeys(rss_urls))
    
    all_articles = []
    project_counter = {}  # Track frequency of projects
    duplicate_count = 0
    
    # Track sentiment for each project
    project_sentiment = {}
    
    for url in unique_urls:
        print(f"Processing: {url}")
        entries = fetch_rss(url)
        
        for entry in entries:
            article_data = extract_article_data(entry, predicted_crypto_list)
            
            # Check if this is a duplicate article
            if is_duplicate(article_data, all_articles):
                duplicate_count += 1
                continue
                
            all_articles.append(article_data)
            
            # Count project mentions
            project_name = article_data['project_name']
            if project_name in project_counter:
                project_counter[project_name] += 1
            else:
                project_counter[project_name] = 1
            
            # Track sentiment for each project
            if project_name not in project_sentiment:
                project_sentiment[project_name] = []
            project_sentiment[project_name].append(article_data['sentiment']['polarity'])
            
            # Random delay to be kind to RSS sources
            time.sleep(1)
    
    print(f"Filtered out {duplicate_count} duplicate articles")
    
    # Sort projects by frequency
    sorted_projects = sorted(project_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate average sentiment for each project
    avg_sentiment = {}
    for project, sentiments in project_sentiment.items():
        if sentiments:
            avg_sentiment[project] = sum(sentiments) / len(sentiments)
    
    # Calculate overall sentiment
    all_sentiments = [article['sentiment']['polarity'] for article in all_articles]
    average_polarity = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
    
    # Find notable articles (high sentiment or popular projects)
    notable_articles = []
    
    # Add articles about top projects
    top_projects = [p[0] for p in sorted_projects[:5]]
    for article in all_articles:
        if article['project_name'] in top_projects:
            # Add high sentiment articles first
            if abs(article['sentiment']['polarity']) > 0.5:
                notable_articles.append(article)
    
    # If we don't have enough notable articles, add some based on recency
    if len(notable_articles) < 3:
        # Sort by recency (if available) and add more
        recent_articles = sorted(
            [a for a in all_articles if a not in notable_articles],
            key=lambda x: x.get('published', ''),
            reverse=True
        )
        notable_articles.extend(recent_articles[:3-len(notable_articles)])
    
    # Separate predicted projects from regular projects
    predicted_projects = []
    regular_projects = []
    
    for name, count in sorted_projects:
        if name.startswith('Predicted:'):
            predicted_projects.append((name, count))
        else:
            regular_projects.append((name, count))
    
    # Create summary of trending projects
    trending_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_articles": len(all_articles),
        "trending_projects": [{"name": name, "mentions": count} for name, count in sorted_projects[:20]],
        "regular_projects": [{"name": name, "mentions": count} for name, count in regular_projects[:20]],
        "predicted_projects": [{"name": name, "mentions": count} for name, count in predicted_projects],
        "unknown_projects": [{"name": name, "mentions": count} for name, count in sorted_projects if name.startswith("*Unknown")],
        "sentiment_summary": {
            "average_polarity": average_polarity,
            "project_sentiment": avg_sentiment
        },
        "notable_articles": [
            {"title": a['title'], "link": a['link'], "project": a['project_name'], "sentiment": a['sentiment']['polarity']}
            for a in notable_articles[:5]
        ],
        "prediction_file": os.path.basename(get_latest_prediction_file()) if get_latest_prediction_file() else "None"
    }
    
    # Save articles to JSON
    articles_filename = f"rss_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_to_json(all_articles, articles_filename)
    print(f"Article data saved to {articles_filename}")
    
    # Save trending summary to JSON
    trends_filename = f"trending_projects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_to_json(trending_summary, trends_filename)
    print(f"Trending summary saved to {trends_filename}")
    
    # Print top trending projects
    print("\nTop 10 Trending Projects:")
    for i, (name, count) in enumerate(sorted_projects[:10], 1):
        print(f"{i}. {name}: {count} mentions")
    
    # Send Telegram update
    telegram_message = format_telegram_trending_message(trending_summary)
    send_result = send_telegram_message(telegram_message)
    
    if send_result:
        print("Telegram update sent successfully")
    else:
        print("Failed to send Telegram update")

if __name__ == "__main__":
    main()
