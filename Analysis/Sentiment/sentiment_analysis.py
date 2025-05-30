from datetime import datetime, date, timedelta
import requests, json, re, os

# Sentiment and websearch keys - you need to create yours
sentiment_key = os.getenv('sentiment_key')
websearch_key = os.getenv('websearch_key')

def get_project_keywords(project_names):
    """Convert comma-separated project names into a dictionary of project:keyword mappings"""
    if not project_names:
        return {}
    projects = [p.strip() for p in project_names.split(',')]
    project_pairs = {}
    for project in projects:
        project_pairs[project] = project  # Map project name to itself as keyword
    return project_pairs

# Get user input for project names
project_names = input("Enter comma-separated project names (e.g., Bitcoin,Ethereum): ")
crypto_key_pairs = get_project_keywords(project_names)

# Define from published date
date_since = date.today() - timedelta(days=1)

# Store inputs in different lists
cryptocurrencies = []
crypto_keywords = []

# Storing keys and values in separate lists
for i in range(len(crypto_key_pairs)):
    cryptocurrencies.append(list(crypto_key_pairs.keys())[i])
    crypto_keywords.append(list(crypto_key_pairs.values())[i])

# Search the web for news using the websearch API, send a request for each crypto in cryptocurrencies
def get_news_headlines():
    """Search the web for news headlines based the keywords in the global variable"""
    news_output = {}

    # Loop through keywords created odd looking dicts. Gotta loop through keys instead
    for crypto in crypto_keywords:
        # Create empty dicts in the news output
        news_output["{0}".format(crypto)] = {'description': [], 'title': []}

        # Configure the fetch request and select date range. Increase date range by adjusting timedelta(days=1)
        url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
        querystring = {
            "q": str(crypto),
            "pageNumber": "1",
            "pageSize": "30",
            "autoCorrect": "true",
            "fromPublishedDate": date_since,
            "toPublishedDate": "null"
        }
        headers = {
            'x-rapidapi-key': websearch_key,
            'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com"
        }

        # Get the raw response
        response = requests.request("GET", url, headers=headers, params=querystring)

        # Convert response to text format
        result = json.loads(response.text)

        # Store each headline and description in the dicts above
        for news in result['value']:
            news_output[crypto]["description"].append(news['description'])
            news_output[crypto]["title"].append(news['title'])

    return news_output

def analyze_headlines():
    """Analyse each headline pulled through the API for each crypto"""
    news_output = get_news_headlines()

    for crypto in crypto_keywords:
        # Empty list to store sentiment value
        news_output[crypto]['sentiment'] = {'pos': [], 'mid': [], 'neg': []}

        # Analyze the description sentiment for each crypto news gathered
        if len(news_output[crypto]['description']) > 0:
            for title in news_output[crypto]['title']:
                # Remove all non-alphanumeric characters from payload
                titles = re.sub('[^A-Za-z0-9]+', ' ', title)

                import http.client
                conn = http.client.HTTPSConnection('text-sentiment.p.rapidapi.com')

                # Format and send the request
                payload = 'text=' + titles
                headers = {
                    'content-type': 'application/x-www-form-urlencoded',
                    'x-rapidapi-key': sentiment_key,
                    'x-rapidapi-host': 'text-sentiment.p.rapidapi.com'
                }
                conn.request("POST", "/analyze", payload, headers)

                # Get the response and format it
                res = conn.getresponse()
                data = res.read()
                title_sentiment = json.loads(data)

                # Assign each positive, neutral and negative count to another list in the news output dict
                if not isinstance(title_sentiment, int):
                    if title_sentiment['pos'] == 1:
                        news_output[crypto]['sentiment']['pos'].append(title_sentiment['pos'])
                    elif title_sentiment['mid'] == 1:
                        news_output[crypto]['sentiment']['mid'].append(title_sentiment['mid'])
                    elif title_sentiment['neg'] == 1:
                        news_output[crypto]['sentiment']['neg'].append(title_sentiment['neg'])
                    else:
                        print(f'Sentiment not found for {crypto}')

    return news_output

def calc_sentiment():
    """Use the sentiment returned in the previous function to calculate percentages"""
    news_output = analyze_headlines()

    # Re-assigned the sentiment list value to a single % calc of all values in each of the 3 lists
    for crypto in crypto_keywords:
        # Length of title list can't be 0 otherwise we'd be dividing by 0 below
        if len(news_output[crypto]['title']) > 0:
            news_output[crypto]['sentiment']['pos'] = (len(news_output[crypto]['sentiment']['pos']) * 100) / len(news_output[crypto]['title'])
            news_output[crypto]['sentiment']['mid'] = (len(news_output[crypto]['sentiment']['mid']) * 100) / len(news_output[crypto]['title'])
            news_output[crypto]['sentiment']['neg'] = (len(news_output[crypto]['sentiment']['neg']) * 100) / len(news_output[crypto]['title'])

            # Print the output for each coin to verify the result
            print(crypto, news_output[crypto]['sentiment'])

    return news_output

# Call the function
calc_sentiment()
