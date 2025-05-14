from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
from textblob import TextBlob
import requests
from shapely.geometry import shape
import numpy as np
import plotly.utils as pu


app = Flask(__name__)

@app.route('/')
def dashboard():
    data = load_data()  # now returns list of dicts
    df = pd.DataFrame(data)

    products = sorted(df['PRODUCT'].dropna().unique().tolist())
    regions = sorted(df['ASM REGION STATE'].dropna().unique().tolist())
    feedbacks = sorted(df['FEEDBACK NOTATION'].dropna().unique().tolist())
    
    return render_template("dashboard.html", products=products, regions=regions, feedbacks=feedbacks)


@app.route('/filtered-data', methods=['POST'])
def filtered_data():
    filters = request.get_json()

    # Load your full loan data (replace with your actual data loading logic)
    all_loan_data = load_data()  # This should return a list of dictionaries

    # Apply filters to the dataset
    def matches_filters(item):
        # Product filter
        if filters.get('product') and item.get('PRODUCT') not in filters['product']:
            return False
        
        # Region filter
        if filters.get('region') and item.get('ASM REGION STATE') not in filters['region']:
            return False
        
        # Feedback filter
        if filters.get('feedback') and item.get('FEEDBACK NOTATION') not in filters['feedback']:
            return False
        
        # Sentiment filter (optional)
        if filters.get('sentiment') and item.get('SENTIMENT_CATEGORY') not in filters['sentiment']:
            return False
        
        # Loan amount filter
        loan_amt = float(item.get('LOAN AMOUNT', 0) or 0)
        if not (filters['min_loan'] <= loan_amt <= filters['max_loan']):
            return False
        
        return True

    filtered_data = list(filter(matches_filters, all_loan_data))
    

    # -------------------------------
    # Feedback Pie Chart
    feedback_counts = dict(Counter(
        item.get('FEEDBACK NOTATION', 'Unknown') for item in filtered_data
    ))
    feedback_pie = {
        'data': [
            go.Pie(
                labels=list(feedback_counts.keys()),
                values=list(feedback_counts.values()),
                hole=0.4
            ).to_plotly_json()
        ],
        'layout': {
            'title': 'Customer Feedback Distribution',
            'margin': {'t': 40, 'b': 40, 'l': 10, 'r': 10},
            'autosize': True
        }
    }

    # -------------------------------
    # Product Bar Chart


    product_counts = dict(Counter(
    item.get('PRODUCT', 'Unknown') for item in filtered_data
    ))

    # Create pie chart in dictionary format
    product_bar = {
        'data': [
            go.Pie(
                labels=list(product_counts.keys()),
                values=list(product_counts.values()),
                hole=0.4  # 0 for a standard pie chart (1 for donut-style)
            ).to_plotly_json()
        ],
        'layout': {
            'title': 'Product Distribution',
            'margin': {'t': 40, 'b': 40, 'l': 10, 'r': 10},
                      'autosize': True
        }
    }
    # -------------------------------
    # Loan Box Plot (distribution by product)

    # Step 1: Count feedback notations per region
    feedback_counts_by_region = defaultdict(Counter)

    for item in filtered_data:
        region = item.get('ASM REGION STATE', 'Unknown')
        feedback = item.get('FEEDBACK NOTATION', 'Unknown')
        feedback_counts_by_region[region][feedback] += 1

    # Step 2: Prepare data
    regions = sorted(feedback_counts_by_region.keys())
    feedback_types = sorted(set(fb for counts in feedback_counts_by_region.values() for fb in counts))

    # Step 3: Create bar traces
    traces = []

    for feedback in feedback_types:
        x_vals = []
        y_vals = []
        for region in regions:
            count = feedback_counts_by_region[region].get(feedback, 0)
            x_vals.append(region)
            y_vals.append(count)
        
        traces.append(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=feedback
            ).to_plotly_json()
        )

    # Step 4: Define chart layout
    loan_box =  {
    'data': traces,
    'layout': {
        'title': 'Feedback Notation Count by Region',
        'xaxis': {
            'tickangle': 45  # Rotate labels counter-clockwise by 45 degrees
        },
        'yaxis': {'title': 'Feedback Count'},
        'barmode': 'group',
        'margin': {'t': 30, 'b': 120},
                    'autosize': True  # Extra bottom margin for angled labels
    }
    }



    # -------------------------------
    # Region Bar Chart (total loan amount by region)


# Step 1: Compute average loan amount grouped by PRODUCT and FEEDBACK NOTATION
    grouped_loan = defaultdict(lambda: defaultdict(list))

    for item in filtered_data:
        product = item.get('PRODUCT', 'Unknown')
        feedback = item.get('FEEDBACK NOTATION', 'Unknown')
        try:
            loan_amt = float(item.get('LOAN AMOUNT', 0) or 0)
            grouped_loan[product][feedback].append(loan_amt)
        except:
            continue

    # Step 2: Compute average and convert to lakhs
    avg_loan_data = []
    products = sorted(grouped_loan.keys())
    feedback_types = set()

    for product in products:
        for feedback, amounts in grouped_loan[product].items():
            avg_amt_lakhs = sum(amounts) / len(amounts) / 1e5
            avg_loan_data.append((product, feedback, avg_amt_lakhs))
            feedback_types.add(feedback)

    # Step 3: Build grouped bar chart
    feedback_types = sorted(feedback_types)
    traces = []

    for feedback in feedback_types:
        x_vals = []
        y_vals = []
        for product in products:
            # Match (product, feedback) to get avg loan
            matched = [val for prod, fb, val in avg_loan_data if prod == product and fb == feedback]
            x_vals.append(product)
            y_vals.append(matched[0] if matched else 0)
        
        traces.append(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=feedback
            ).to_plotly_json()
        )

    region_bar= {
        'data': traces,
        'layout': {
            'title': 'Feedback Notation by Product',
            'yaxis': {'title': 'Average Loan Amount (in Lakhs)'},
            'barmode': 'group',
            'margin': {'t': 30,'b': 80},
            'autosize': True
        }
    }


    # Step 1: Compute total and count to calculate average loan amount
    grouped_loan_total = defaultdict(lambda: defaultdict(float))
    grouped_loan_count = defaultdict(lambda: defaultdict(int))

    for item in filtered_data:
        region = item.get('ASM REGION STATE', 'Unknown')
        feedback = item.get('FEEDBACK NOTATION', 'Unknown')
        try:
            loan_amt = float(item.get('LOAN AMOUNT', 0) or 0)
            grouped_loan_total[region][feedback] += loan_amt
            grouped_loan_count[region][feedback] += 1
        except:
            continue

    # Step 2: Prepare data for plotting
    regions = sorted(grouped_loan_total.keys())
    feedback_types = set()

    for feedbacks in grouped_loan_total.values():
        feedback_types.update(feedbacks.keys())

    feedback_types = sorted(feedback_types)

    # Step 3: Create bar traces with average loan values (in lakhs)
    traces = []

    for feedback in feedback_types:
        x_vals = []
        y_vals = []
        for region in regions:
            total = grouped_loan_total[region].get(feedback, 0)
            count = grouped_loan_count[region].get(feedback, 1)  # avoid division by zero
            avg_loan = total / count
            x_vals.append(region)
            y_vals.append(avg_loan / 100000)  # Convert to lakhs

        traces.append(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=feedback
            ).to_plotly_json()
        )

    # Step 4: Define the layout
    loan_amount_distribution_map = {
        'data': traces,
        'layout': {
            'title': 'Average Loan Amount by Region and Feedback',
            'xaxis': {
                'title': 'Region',
                'tickangle': 45
            },
            'yaxis': {
                'title': 'Average Loan Amount (in Lakhs)',
                'tickformat': '.2f'
            },
            'barmode': 'group',
            'margin': {'t': 40,'b': 160}
        }
    }

    # -------------------------------
    # -------------------------------

    # Sentiment Analysis
    # Filtered data
    filtered_data = list(filter(matches_filters, all_loan_data))

    # -------------------------------
    # Total Disbursed Amount by Feedback Category (in Lakhs)
    total_disbursed = defaultdict(float)
    for item in filtered_data:
        feedback = item.get('FEEDBACK NOTATION', 'Unknown')
        try:
            disbursed_amt = float(item.get('DISBURSED AMOUNT', 0) or 0)
            # Convert to lakhs
            disbursed_amt_lakhs = disbursed_amt / 100000
            total_disbursed[feedback] += disbursed_amt_lakhs
        except:
            continue

    total_disbursed_items = sorted(total_disbursed.items(), key=lambda x: x[1], reverse=False)
    x_vals = [item[0] for item in total_disbursed_items]
    y_vals = [item[1] for item in total_disbursed_items]

    total_disbursed_plot = {
        'data': [
            go.Bar(
                x=y_vals,
                y=x_vals,
                orientation='h',
                marker=dict(color='mediumseagreen')
            ).to_plotly_json()
        ],
        'layout': {
            'title': 'Total Disbursed Amount by Feedback Category (in Lakhs)',
            'xaxis': {'title': 'Total Amount (Lakhs)'},
            'template': 'plotly_white',
            'margin': {'t': 40, 'b': 50, 'l': 150, 'r': 50}
        }
    }

    # Sentiment Distribution
    sentiment_counts = defaultdict(int)
    for item in filtered_data:
        sentiment = item.get('SENTIMENT_CATEGORY', 'Unknown')
        sentiment_counts[sentiment] += 1

    sentiment_plot = {
        'data': [
            go.Histogram(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values()),
                histfunc='sum',
                name='Sentiment',
                marker=dict(color='royalblue')
            ).to_plotly_json()
        ],
        'layout': {
            'title': 'Sentiment Distribution from Customer Feedback',
            'xaxis': {'title': 'Sentiment'},
            'yaxis': {'title': 'Count'},
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    }

    # Average Sentiment Polarity by Region
    region_sentiment = defaultdict(list)
    for item in filtered_data:
        region = item.get('ASM REGION STATE', 'Unknown')
        try:
            sentiment_polarity = float(item.get('SENTIMENT_POLARITY', 0) or 0)
            region_sentiment[region].append(sentiment_polarity)
        except:
            continue

    # Compute average sentiment polarity by region
    avg_region_sentiment = {region: sum(values)/len(values) for region, values in region_sentiment.items()}
    avg_region_sentiment_sorted = sorted(avg_region_sentiment.items(), key=lambda x: x[1], reverse=False)

    x_vals = [item[0] for item in avg_region_sentiment_sorted]
    y_vals = [item[1] for item in avg_region_sentiment_sorted]

    region_sentiment_plot = {
        'data': [
            go.Bar(
                x=y_vals,
                y=x_vals,
                orientation='h',
                marker=dict(color='lightblue')
            ).to_plotly_json()
        ],
        'layout': {
            'title': 'Average Sentiment Polarity by Region',
            'xaxis': {'title': 'Avg Sentiment Polarity'},
            'margin': {'l': 210, 'r': 50, 't': 50, 'b': 50}
        }
    }

    # Average Loan Amount by Sentiment (in Lakhs)
    avg_loan_sentiment = defaultdict(lambda: defaultdict(float))
    avg_loan_count = defaultdict(lambda: defaultdict(int))

    for item in filtered_data:
        sentiment = item.get('SENTIMENT_CATEGORY', 'Unknown')
        try:
            loan_amount = float(item.get('LOAN AMOUNT', 0) or 0)
            # Convert to lakhs
            loan_amount_lakhs = loan_amount / 100000
            avg_loan_sentiment[sentiment]['total'] += loan_amount_lakhs
            avg_loan_count[sentiment]['count'] += 1
        except:
            continue

    # Calculate average loan amount by sentiment
    avg_loan_data = [
        {
            'SENTIMENT_CATEGORY': sentiment,
            'LOAN AMOUNT': avg_loan_sentiment[sentiment]['total'] / avg_loan_count[sentiment]['count']
        }
        for sentiment in avg_loan_sentiment
    ]

    sentiment_categories = [item['SENTIMENT_CATEGORY'] for item in avg_loan_data]
    loan_amounts = [item['LOAN AMOUNT'] for item in avg_loan_data]

    avg_loan_sentiment_plot = {
        'data': [
            go.Bar(
                x=sentiment_categories,
                y=loan_amounts,
                name='Loan Amount',
                marker=dict(color='lightcoral')
            ).to_plotly_json()
        ],
        'layout': {
            'title': 'Average Loan Amount by Sentiment (in Lakhs)',
            'xaxis': {'title': 'Sentiment Category'},
            'yaxis': {'title': 'Average Loan Amount (Lakhs)'},
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    }

    # Flow of Product → Region → Feedback
    # Flow of Product → Region → Feedback
    # fig_flow = px.parallel_categories(filtered_data,
    #                                 dimensions=['PRODUCT', 'ASM REGION STATE', 'FEEDBACK NOTATION'],
    #                                 color_continuous_scale=px.colors.sequential.Inferno,
    #                                 title='Flow of Product → Region → Feedback')

    # # Convert the entire figure to JSON
    # flow_plot = json.loads(pu.PlotlyJSONEncoder().encode(fig_flow))


#____________________________
#Code of geoplot


    # Step 1: Load GeoJSON
    # url = "https://raw.githubusercontent.com/harshal-96/geo_plot_json/main/Rajasthan.geojson"
    # response = requests.get(url)
    # geojson_data = json.loads(response.text)
    file_path = "Rajasthan.geojson"

    # Open the local GeoJSON file and load the data
    with open(file_path, 'r') as file:
        geojson_data = json.load(file)

    df=pd.read_excel('main_survey.xlsx')
    df['ASM REGION STATE'] = df['ASM REGION STATE'].replace("HEAD OFFICE, C - SCHEME, JAIPUR", 'Jaipur')


    # Mapping of places to their districts using the provided list
    place_to_district = {
        'Alwar': 'Alwar',
        'Bhilwara': 'Bhilwara',
        'Palanpur': 'Banaskantha District',  # This place is not in the provided list, so I leave it as is
        'Bikaner': 'Bikaner',
        'Neem Ka Thana': 'Sikar',
        'Sri Ganganagar': 'Ganganagar',
        'SRI GANGANAGAR': 'Ganganagar',
        'Behror Paykosh Center': 'Alwar',
        'Jaipur': 'Jaipur',
        'Nokha': 'Bikaner',  # 'Nokha' is part of Bikaner District
        'Jodhpur': 'Jodhpur',
        'Mehsana': 'Mehsana',  # Not in the provided list, keeping as is
        'Himatnagar': 'Sabarkantha',  # Not in the provided list, keeping as is
        'Kishangarh': 'Ajmer',
        'Newai': 'Tonk'
    }

    # Replacing place names with district names
    df['ASM REGION STATE'] = df['ASM REGION STATE'].replace(place_to_district)

    # Step 2: Your dataset
    district_stats = df.groupby(['ASM REGION STATE', 'FEEDBACK NOTATION']).size().reset_index(name='count')

    # Step 3: Match property name from GeoJSON
    geo_ids = [feature['properties'] for feature in geojson_data['features']]
    sample_props = geo_ids[0]
    matching_keys = [key for key in sample_props if 'dist_name' in key.lower() or 'region' in key.lower() or 'asm' in key.lower()]

    if not matching_keys:
        raise ValueError("No matching key found in GeoJSON.")
    matching_key = matching_keys[0]
    featureidkey = f"properties.{matching_key}"

    # Step 4: Merge all districts with stats (even if count is NaN)
    all_districts = pd.DataFrame([f['properties'][matching_key] for f in geojson_data['features']], columns=[matching_key])
    notation_type = 'POSITIVE'
    sub_stats = district_stats[district_stats['FEEDBACK NOTATION'] == notation_type]
    merged = all_districts.merge(sub_stats, left_on=matching_key, right_on='ASM REGION STATE', how='left')
    merged['count'] = merged['count'].fillna(0)

    # Step 5: Create bins for the discrete colors
    max_count = merged['count'].max()

    # Create 5 bins: 0, and then 4 equal ranges for values > 0
    bins = [0]

    # If we have non-zero values, create 4 ranges for them
    if max_count > 0:
        # Define bin edges based on max_count for positive values
        positive_bins = np.linspace(0, max_count, 9).tolist() # Creates 5 edges for positive range
        # Combine the zero bin with the positive bins, ensuring unique and sorted
        bins = sorted(list(set([0] + [int(np.ceil(b)) for b in positive_bins if b > 0])))
        # Ensure the last bin edge is at least the max_count if not already covered
        if bins[-1] < max_count:
            bins.append(int(np.ceil(max_count)))
        bins = sorted(list(set(bins))) # Ensure unique and sorted again

    # Generate labels based on the final bins list *after* it's constructed
    if len(bins) > 1:
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        # Special handling for the first bin if it starts at 0
        if bins[0] == 0:
            labels[0] = f"0-{int(bins[1])}"
    else:
        # Handle the case where all counts are 0 (bins will be [0])
        labels = ["0"]
    non_zero_values = merged.loc[merged['count'] > 0, 'count']

    # Step 1: Count unique non-zero values
    unique_val_count = len(non_zero_values.unique())

    # Step 2: Check if 10 is present in the unique values
    contains_ten = 10 in non_zero_values.unique()

    # Step 3: Dynamically calculate number of bins
    if unique_val_count <= 3:
        n_bins = 1
    elif unique_val_count <= 10 or not contains_ten:  # If count <= 10 or 10 not in unique values
        n_bins = int(unique_val_count / 2)
    else:
        n_bins = max(4, int(np.log2(unique_val_count)))  # Smooth scaling with data

    # Generate bin edges using non-zero values only
    bin_edges = pd.cut(non_zero_values, bins=n_bins, retbins=True, duplicates='drop')[1]

    # Insert 0 as a separate bin at the beginning
    bins = np.insert(bin_edges, 0, 0)  # e.g., [0, 5.0, 10.0, 15.0, 20.0]
    bins[0] = -0.1  # So 0 gets its own bin (since pd.cut is (a, b])
    bins = np.unique(bins)
    # Generate labels for the bins
    labels = []
    for i in range(len(bins) - 1):
        if bins[i] < 0.1 and bins[i+1] == 0:  # Special label for 0
            labels.append('0')
        else:
            labels.append(f"{int(bins[i]+1)}–{int(bins[i+1])}")
    labels = np.unique(labels)
    # Create the categorical bin column
    merged['count_category'] = pd.cut(
        merged['count'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    # Define colors: Gray for 0, others from colormap
    colors = ['rgb(211,211,211)'] + px.colors.sequential.YlOrBr[1:len(labels)]

    # Map each label to a color
    color_discrete_map = {label: colors[i] for i, label in enumerate(labels)}

    # Plot
    fig = px.choropleth(
        merged,
        geojson=geojson_data,
        featureidkey=featureidkey,
        locations=matching_key,
        color='count_category',
        color_discrete_map=color_discrete_map,
        title=f"Feedback Distribution: {notation_type}",
        labels={'count_category': 'Feedback Count'}
    )

    fig.update_traces(marker_line_width=1.2, marker_line_color='black')
    fig.update_geos(fitbounds="locations", visible=False)


    # Add district labels at centroids
    for feature in geojson_data['features']:
        props = feature['properties']
        name = props[matching_key]

        # Use shapely to get accurate centroid
        geom = shape(feature['geometry'])
        centroid = geom.centroid
        x, y = centroid.x, centroid.y

        fig.add_trace(go.Scattergeo(
            lon=[x],
            lat=[y],
            text=name,
            mode='text',
            showlegend=False,
            textfont=dict(color="black", size=9),
            hoverinfo='skip'  # optional: disables hover popups on labels
        ))

    # fig.update_layout(height=800,width=1000)
    fig.update_layout(autosize=True)
    positive_plot = json.loads(pu.PlotlyJSONEncoder().encode(fig))

    notation_type = 'Negative'
    sub_stats = district_stats[district_stats['FEEDBACK NOTATION'] == notation_type]
    merged = all_districts.merge(sub_stats, left_on=matching_key, right_on='ASM REGION STATE', how='left')
    merged['count'] = merged['count'].fillna(0)

    # Step 5: Create bins for the discrete colors
    max_count = merged['count'].max()

    # Create 5 bins: 0, and then 4 equal ranges for values > 0
    bins = [0]

    # If we have non-zero values, create 4 ranges for them
    if max_count > 0:
        # Define bin edges based on max_count for positive values
        positive_bins = np.linspace(0, max_count, 9).tolist() # Creates 5 edges for positive range
        # Combine the zero bin with the positive bins, ensuring unique and sorted
        bins = sorted(list(set([0] + [int(np.ceil(b)) for b in positive_bins if b > 0])))
        # Ensure the last bin edge is at least the max_count if not already covered
        if bins[-1] < max_count:
            bins.append(int(np.ceil(max_count)))
        bins = sorted(list(set(bins))) # Ensure unique and sorted again

    # Generate labels based on the final bins list *after* it's constructed
    if len(bins) > 1:
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        # Special handling for the first bin if it starts at 0
        if bins[0] == 0:
            labels[0] = f"0-{int(bins[1])}"
    else:
        # Handle the case where all counts are 0 (bins will be [0])
        labels = ["0"]

    # Separate zero and non-zero counts
    non_zero_values = merged.loc[merged['count'] > 0, 'count']

    # Define number of desired bins (optional: use Sturges' rule or fixed value)
    n_bins = int(len(non_zero_values.unique())/2)  # or use something like int(np.sqrt(len(non_zero_values))) for dynamic bin count

    # Generate bin edges using non-zero values only
    bin_edges = pd.cut(non_zero_values, bins=n_bins, retbins=True, duplicates='drop')[1]

    # Insert 0 as a separate bin at the beginning
    bins = np.insert(bin_edges, 0, 0)  # e.g., [0, 5.0, 10.0, 15.0, 20.0]
    bins[0] = -0.1  # So 0 gets its own bin (since pd.cut is (a, b])
    bins = np.unique(bins)
    # Generate labels for the bins
    labels = []
    for i in range(len(bins) - 1):
        if bins[i] < 0.1 and bins[i+1] == 0:  # Special label for 0
            labels.append('0')
        else:
            labels.append(f"{int(bins[i]+1)}–{int(bins[i+1])}")
    labels = np.unique(labels)
    # Create the categorical bin column
    merged['count_category'] = pd.cut(
        merged['count'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    # Define colors: Gray for 0, others from colormap
    colors = ['rgb(211,211,211)'] + px.colors.sequential.YlOrBr[1:len(labels)]

    # Map each label to a color
    color_discrete_map = {label: colors[i] for i, label in enumerate(labels)}

    # Plot
    fig = px.choropleth(
        merged,
        geojson=geojson_data,
        featureidkey=featureidkey,
        locations=matching_key,
        color='count_category',
        color_discrete_map=color_discrete_map,
        title=f"Feedback Distribution: {notation_type}",
        labels={'count_category': 'Feedback Count'}
    )

    fig.update_traces(marker_line_width=1.2, marker_line_color='black')
    fig.update_geos(fitbounds="locations", visible=False)

    # Add district labels at centroids
    for feature in geojson_data['features']:
        props = feature['properties']
        name = props[matching_key]

        # Use shapely to get accurate centroid
        geom = shape(feature['geometry'])
        centroid = geom.centroid
        x, y = centroid.x, centroid.y

        fig.add_trace(go.Scattergeo(
            lon=[x],
            lat=[y],
            text=name,
            mode='text',
            showlegend=False,
            textfont=dict(color="black", size=9),
            hoverinfo='skip'  # optional: disables hover popups on labels
        ))

    # fig.update_layout(height=800,width=1000)
    fig.update_layout(autosize=True)
    negative_plot = json.loads(pu.PlotlyJSONEncoder().encode(fig))

    # -------------------------------
    return jsonify({
        'data': filtered_data,
        'graphs': {
            'feedback_pie': feedback_pie,
            'product_bar': product_bar,
            'loan_box': loan_box,
            'region_bar': region_bar,
            'loan_amount_distribution_map': loan_amount_distribution_map,
            'total_disbursed_plot': total_disbursed_plot,
            'sentiment_plot': sentiment_plot,
            'region_sentiment_plot': region_sentiment_plot,
            'avg_loan_sentiment_plot': avg_loan_sentiment_plot,
            'positive_plot': positive_plot,
            'negative_plot': negative_plot
        }
    })



def load_data():
    df = pd.read_excel('main_survey.xlsx')

    # Ensure text is not null
    df['CUSTOMER FEEDBACK'] = df['CUSTOMER FEEDBACK'].astype(str).fillna("")

    # Apply TextBlob sentiment polarity analysis
    df['SENTIMENT_POLARITY'] = df['CUSTOMER FEEDBACK'].apply(lambda text: TextBlob(text).sentiment.polarity)

    # Classify Sentiment Category
    def get_sentiment_label(score):
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    df['SENTIMENT_CATEGORY'] = df['SENTIMENT_POLARITY'].apply(get_sentiment_label)
    
    return df.to_dict('records')
if __name__ == '__main__':
    app.run(debug=True)
