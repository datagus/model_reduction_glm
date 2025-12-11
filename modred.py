# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==6.0.0",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
#     "statsmodels==0.14.6",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Model_Reduction")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import numpy as np

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    return alt, mo, pd, sm, train_test_split


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore', message='You passed a.*to `is_pandas_dataframe`')
    return


@app.cell(hide_code=True)
def _():
    #mo.md(text="## Data Fetching")
    return


@app.cell
def _(mo):
    mo.md("""
    # Model Reduction
    """)
    return


@app.cell
def _(mo):
    mo.md(text="""
        The objective of this notebook application is to introduce students with model reduction using generalized linear models

        Data source: https://archive.ics.uci.edu/dataset/327/phishing+websites


        Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X.
        """).callout(kind="info")
    return


@app.cell
def _(mo, pd):
    @mo.cache()
    def fetch_pish_data():
        url = "https://raw.githubusercontent.com/datagus/ASDA2025/main/datasets/exercse_week9/pishing.csv"
        pish = pd.read_csv(url, encoding='latin-1')
        return pish

    pish_df = fetch_pish_data()
    return (pish_df,)


@app.cell(hide_code=True)
def _(mo):
    # 1. creating the long description of the predictors
    feature_long = {

        "having_ip_address": mo.md(""" 

    ### **IP Address in URL (Phishing Indicator)**
    If an IP address is used instead of a domain name in the URL (e.g.`http://125.98.3.123/fake.html`), users can be confident that someone is likely trying to steal personal information. Sometimes, the IP address is written in hexadecimal form, for example:  
    `http://0x58.0xCC.0xCA.0x62/2/paypal.ca/index.html`

    ### **Rule**<br> 
    **IF** domain contains IP address → **Phishing**  
    **Otherwise** → **Legitimate**

    """),

        "url_length": mo.md("""

    ## **Long URL to Hide the Suspicious Part**
    Phishers can use long URL to hide the doubtful part in the address bar. For example:

    `http://federmacedoadv.com.br/3f/aze/ab51e2e319e51502f416dbe46b773a5e/?cmd=_home&amp;dispatch=11004d58f5b74f8dc1e7c2e8dd4105e811004d58f5b74f8dc1e7c2e8dd4105e8@phishing.website.html`

    To ensure accuracy of our study, we calculated the length of URLs in the dataset and produced an average URL length. The results showed that if the length of the URL is greater than or equal 54 characters then the URL classified as phishing. By reviewing our dataset we were able to find 1220 URLs lengths equals to 54 or more which constitute 48.8% of the total dataset size.

    ### **Rule**:
    **IF** URL length < 54 → **Legitimate** <br>
    **ELSEIF** URL length ≥ 54 and ≤ 75 → **Suspicious**<br>
    **Otherwise** → **Phishing**
    """),

        "shortining_service": mo.md("""
    ## **Using URL Shortening Services “TinyURL”**
    URL shortening is a method on the “World Wide Web” in which a URL may be made considerably smaller in length and still lead to the required webpage. This is accomplished by means of an “HTTP Redirect” on a domain name that is short, which links to the webpage that has a long URL. For example, the URL `http://portal.hud.ac.uk/` can be shortened to `bit.ly/19DXSk4`.

    ### Rule: 
    **IF** TinyURL → **Phishing** <br>
    **ELSE** → Legitimate
    """),

        "having_at_symbol": mo.md("""
    ## **URL’s having “@” Symbol**
    Using “@” symbol in the URL leads the browser to ignore everything preceding the “@” symbol and the real address often follows the “@” symbol.

    ### **Rule**
    **IF** Url Having @ Symbol → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "double_slash_redirecting": mo.md("""
    ## **Redirecting using “//”**
    The existence of “//” within the URL path means that the user will be redirected to another website. An example of such URL’s is: `http://www.legitimate.com//http://www.phishing.com`. We examin the location where the “//” appears. We find that if the URL starts with “HTTP”, that means the “//” should appear in the sixth position. However, if the URL employs “HTTPS” then the “//” should appear in seventh position.

    ### **Rule** 
    **IF** The Position of the Last Occurrence of "//" in the URL > 7 → **Phishing** <br> 
    **Otherwise** → **Legitimate**
    """),

        "prefix_suffix": mo.md("""
    ## **Adding Prefix or Suffix Separated by (-) to the Domain**
    The dash symbol is rarely used in legitimate URLs. Phishers tend to add prefixes or suffixes separated by (-) to the domain name so that users feel that they are dealing with a legitimate webpage. For example `http://www.Confirme-paypal.com/`

    ### **Rule**
    **IF** Domain Name Part Includes (-) Symbol → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "having_sub_domain": mo.md("""
    ## **Sub Domain and Multi Sub Domains**  
    Let us assume we have the following link: `http://www.hud.ac.uk/students/`. A domain name might include the country-code top-level domains (ccTLD), which in our example is “uk”. The “ac” part is shorthand for “academic”, the combined “ac.uk” is called a second-level domain (SLD) and “hud” is the actual name of the domain.

    To produce a rule for extracting this feature, we firstly have to omit the (www.) from the URL which is in fact a sub domain in itself. Then, we have to remove the (ccTLD) if it exists. Finally, we count the remaining dots. If the number of dots is greater than one, then the URL is classified as “Suspicious” since it has one sub domain. However, if the dots are greater than two, it is classified as “Phishing” since it will have multiple sub domains. Otherwise, if the URL has no sub domains, we will assign “Legitimate” to the feature.

    ### **Rule**
    **IF** Dots In Domain Part = 1 → **Legitimate** <br>
    **ELSEIF** Dots In Domain Part = 2 → **Suspicious** <br>
    **Otherwise** → **Phishing**
    """),

        "sslfinal_state": mo.md("""
    ## **HTTPS (Hyper Text Transfer Protocol with Secure Sockets Layer)**  
    Let us assume we have the following link: `http://www.hud.ac.uk/students/`. A domain name might include the country-code top-level The existence of HTTPS is very important in giving the impression of website legitimacy, but this is clearly not enough. The authors in (Mohammad, Thabtah and McCluskey 2012) (Mohammad, Thabtah and McCluskey 2013) suggest checking the certificate assigned with HTTPS including the extent of the trust certificate issuer, and the certificate age. Certificate Authorities that are consistently listed among the top trustworthy names include: “GeoTrust, GoDaddy, Network Solutions, Thawte, Comodo, Doster and VeriSign”. Furthermore, by testing out our datasets, we find that the minimum age of a reputable certificate is two years.

    ### **Rule**
    **IF** Use https and Issuer Is Trusted &and Age of Certificate≥ 1 Years → **Legitimate** <br>
    **ELSEIF** Using https and Issuer Is Not Trusted → **Suspicious** <br>
    **Otherwise** → **Phishing**
    """),

        "domain_registration_length": mo.md("""
    ## **Domain Registration Length**  
    Based on the fact that a phishing website lives for a short period of time, we believe that trustworthy domains are regularly paid for several years in advance. In our dataset, we find that the longest fraudulent domains have been used for one year only. 

    ### **Rule**
    **IF** Domains Expires on≤ 1 years  → **Pishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "favicon": mo.md("""
    ## **Favicon**  
    A favicon is a graphic image (icon) associated with a specific webpage. Many existing user agents such as graphical browsers and newsreaders show favicon as a visual reminder of the website identity in the address bar. If the favicon is loaded from a domain other than that shown in the address bar, then the webpage is likely to be considered a Phishing attempt. 

    ### **Rule**
    **IF** Favicon Loaded From External Domain  → **Pishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "port": mo.md("""
    ## **Using Non-Standard Port**  
    This feature is useful in validating if a particular service (e.g. HTTP) is up or down on a specific server. In the aim of controlling intrusions, it is much better to merely open ports that you need. Several firewalls, Proxy and Network Address Translation (NAT) servers will, by default, block all or most of the ports and only open the ones selected. If all ports are open, phishers can run almost any service they want and as a result, user information is threatened. The most important ports and their preferred status are shown in Table 2.


    | Port | Service         | Meaning                                                             | Preferred Status |
    |------|------------------|---------------------------------------------------------------------|------------------|
    | 21   | FTP              | Transfer files from one host to another                             | Close            |
    | 22   | SSH              | Secure File Transfer Protocol                                       | Close            |
    | 23   | Telnet           | Provide a bidirectional interactive text-oriented communication     | Close            |
    | 80   | HTTP             | Hypertext Transfer Protocol                                         | Open             |
    | 443  | HTTPS            | Hypertext Transfer Protocol Secured                                 | Open             |
    | 445  | SMB              | Providing shared access to files, printers, serial ports            | Close            |
    | 1433 | MSSQL            | Store and retrieve data as requested by other software applications | Close            |
    | 1521 | ORACLE           | Access Oracle database from web                                     | Close            |
    | 3306 | MySQL            | Access MySQL database from web                                      | Close            |
    | 3389 | Remote Desktop   | Allow remote access and remote collaboration                        | Close            |

    ### **Rule**
    **IF** "Port # is of the " Preffered Status  → **Pishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "https_token": mo.md("""
    ## **The Existence of “HTTPS” Token in the Domain Part of the URL**  
    The phishers may add the “HTTPS” token to the domain part of a URL in order to trick users. For example,
    `http://https-www-paypal-it-webapps-mpp-home.soft-hair.com/`

    ### **Rule**
    **IF** "Using " HTTP Token in Domain Part of The URL  → **Pishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "request_url": mo.md("""
    ## **Request URL**  
    Request URL examines whether the external objects contained within a webpage such as images, videos and sounds are loaded from another domain. In legitimate webpages, the webpage address and most of objects embedded within the webpage are sharing the same domain. 

    ### **Rule**
    **IF** % of Request URL <22% → **Legitimate** <br>
    **ELSEIF** %of Request URL≥22% and 61% → **Suspicious** <br>
    **Otherwise** → feature =**Phishing**
    """),

        "url_of_anchor": mo.md("""
    ## **URL of Anchor**  
    An anchor is an element defined by the `<a>` tag. This feature is treated exactly as “Request URL”. However, for this feature we examine:  
    - If the `<a>` tags and the website have different domain names.  
    - If the anchor does not link to any webpage, e.g.:  
      `<a href="#">`, `<a href="#content">`, `<a href="#skip">`, `<a href="JavaScript::void(0)">`

    ### **Rule**
    **IF** % of URL of Anchor < 31% → **Legitimate** <br>
    **ELSEIF** % of URL of Anchor ≥ 31% and ≤ 67% → **Suspicious** <br>
    **Otherwise** → **Phishing**
    """),

        "links_in_tags": mo.md("""
    ## **Links in `<Meta>`, `<Script>` and `<Link>` tags**  
    Given that legitimate webpages commonly use `<Meta>` tags for metadata, `<Script>` tags for client-side scripts, and `<Link>` tags for loading other resources, these links are expected to share the same domain as the webpage.

    ### **Rule**
    **IF** % of these links < 17% → **Legitimate** <br>
    **ELSEIF** % of these links ≥ 17% and ≤ 81% → **Suspicious** <br>
    **Otherwise** → **Phishing**
    """),

        "sfh": mo.md("""
    ## **Server Form Handler (SFH)**  
    SFHs that contain an empty string or “about:blank” are considered doubtful because an action should be taken upon the submitted information.  
    If the domain name in the SFH differs from the domain name of the webpage, this is considered suspicious because submitted information is rarely handled by external domains.

    ### **Rule**
    **IF** SFH is empty or "about:blank" → **Phishing** <br>
    **ELSEIF** SFH refers to a different domain → **Suspicious** <br>
    **Otherwise** → **Legitimate**
    """), 

        "submitting_to_email": mo.md("""
    ## **Submitting Information to Email**  
    A phisher may redirect the user’s personal information to an email account using server-side scripts such as `mail()` in PHP or client-side functions like `mailto:`.

    ### **Rule**
    **IF** the form uses `mail()` or `mailto:` → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "abnormal_url": mo.md("""
    ## **Abnormal URL**  
    This feature is extracted from the WHOIS database.  
    For legitimate websites, the identity of the domain is usually part of the URL.

    ### **Rule**
    **IF** the host name is not included in the URL → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "redirect": mo.md("""
    ## **Website Forwarding**  
    Phishing websites tend to redirect multiple times, whereas legitimate ones rarely redirect more than once.

    ### **Rule**
    **IF** number of redirects ≤ 1 → **Legitimate** <br>
    **ELSEIF** number of redirects ≥ 2 and < 4 → **Suspicious** <br>
    **Otherwise** → **Phishing**
    """),

        "on_mouseover": mo.md("""
    ## **Status Bar Customization**  
    Phishers may use JavaScript (such as `onMouseOver`) to display a fake URL in the status bar.

    ### **Rule**
    **IF** onMouseOver changes the status bar → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "rightclick": mo.md("""
    ## **Disabling Right Click**  
    Phishers sometimes disable right-click using JavaScript (`event.button == 2`) to prevent users from viewing page source.

    ### **Rule**
    **IF** right click is disabled → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "popupwindow": mo.md("""
    ## **Using Pop-up Window**  
    Legitimate websites rarely ask for personal information through pop-up windows.

    ### **Rule**
    **IF** pop-up window contains text fields → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "iframe": mo.md("""
    ## **IFrame Redirection**  
    Phishers may embed invisible iframes (`frameBorder=0`) to load malicious content.

    ### **Rule**
    **IF** an iframe is used → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "age_of_domain": mo.md("""
    ## **Age of Domain**  
    Phishing websites tend to be very new, while legitimate domains are usually at least 6 months old.

    ### **Rule**
    **IF** age of domain ≥ 6 months → **Legitimate** <br>
    **Otherwise** → **Phishing**
    """),

        "dnsrecord": mo.md("""
    ## **DNS Record**  
    If the DNS record is empty or missing, the site is likely phishing.

    ### **Rule**
    **IF** no DNS record exists → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "web_traffic": mo.md("""
    ## **Website Traffic**  
    This feature measures the popularity of the website based on visitor traffic (e.g., Alexa ranking).

    ### **Rule**
    **IF** website rank < 100,000 → **Legitimate** <br>
    **ELSEIF** website rank > 100,000 → **Suspicious** <br>
    **Otherwise** → **Phishing**
    """),

        "page_rank": mo.md("""
    ## **PageRank**  
    Phishing webpages usually have a PageRank close to zero.

    ### **Rule**
    **IF** PageRank < 0.2 → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """),

        "google_index": mo.md("""
    ## **Google Index**  
    This feature checks whether the webpage is indexed by Google.

    ### **Rule**
    **IF** webpage is indexed by Google → **Legitimate** <br>
    **Otherwise** → **Phishing**
    """),

        "links_pointing_to_page": mo.md("""
    ## **Number of Links Pointing to Page**  
    The number of external links pointing to a webpage is an indicator of legitimacy.

    ### **Rule**
    **IF** number of links = 0 → **Phishing** <br>
    **ELSEIF** number of links > 0 and ≤ 2 → **Suspicious** <br>
    **Otherwise** → **Legitimate**
    """),

        "statistical_report": mo.md("""
    ## **Statistical Reports Based Feature**  
    Statistical reports (e.g., PhishTank, StopBadware) list top phishing domains and IP addresses.

    ### **Rule**
    **IF** host belongs to top phishing IPs or domains → **Phishing** <br>
    **Otherwise** → **Legitimate**
    """)
    }
    return (feature_long,)


@app.cell(hide_code=True)
def add_description(mo, pish_df):
    # 3. Creating dropdown with columns names
    columns=pish_df.columns.tolist()
    column_selector = mo.ui.dropdown(
        options=columns[:-1],
        value=columns[0],
        label = "Select the predictor",
        full_width=True
    )
    return (column_selector,)


@app.cell
def _(column_selector, feature_long, mo):
    # 4. Creating ui to check the column descriptions
    des_long = feature_long[column_selector.value]

    accordi = mo.accordion({
        "Click to expand": des_long
    })

    column_description = mo.vstack([column_selector, accordi], gap=0.05)
    return (column_description,)


@app.cell(hide_code=True)
def _(alt, column_selector, pish_df):
    # 5. Creating chart for feature distribution

    _aggregated = (
        pish_df
        .groupby(column_selector.value)
        .size()
        .reset_index(name='count')
    )


    charti = (
        alt.Chart(_aggregated)  # ✅ Correct - pass DataFrame directly
        .mark_bar()
        .encode(
            x=alt.X(field=column_selector.value, type="quantitative", bin=True, title=column_selector.value),
            y=alt.Y(aggregate="count", type="quantitative", title="Number of records"),
            tooltip=[
                alt.Tooltip(
                    column_selector.value,
                    type="quantitative",
                    bin=True,
                    title=column_selector.value,
                ),
                alt.Tooltip(
                    "count()",
                    type="quantitative",
                    format=",.0f",
                    title="Number of records",
                ),
            ],
        ).properties(width="container").configure_view(stroke=None)
    )
    return


@app.cell(hide_code=True)
def _():
    #mo.md(text="## Splitting the data")
    return


@app.cell(hide_code=True)
def _(pish_df, train_test_split):
    # 6. Splitting the data

    #creating backup
    df = pish_df.copy()

    # Prepare features and target
    X = pish_df.drop(columns=['result'])
    y = pish_df['result']

    # Convert target to binary (0 and 1) - assuming -1 is legitimate, 1 is phishing
    y_binary = (y == 1).astype(int)

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Phishing rate in train: {y_train.mean():.2%}")
    print(f"Phishing rate in test: {y_test.mean():.2%}")
    return X, X_test, X_train, y_test, y_train


@app.cell
def _(X_train, alt):
    # Calculate correlation matrix
    _correlation_matrix = X_train.corr()

    # Convert to long format for Altair
    _corr_long = _correlation_matrix.reset_index().melt(
        id_vars='index',
        var_name='variable2',
        value_name='correlation'
    )
    _corr_long.columns = ['variable1', 'variable2', 'correlation']

    # Create Altair heatmap
    correlation_heatmap = (
        alt.Chart(_corr_long)
        .mark_rect()
        .encode(
            x=alt.X('variable1:N', title='Feature', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('variable2:N', title='Feature'),
            color=alt.Color(
                'correlation:Q',
                scale=alt.Scale(scheme='redblue', domain=[-1, 1], reverse=True),
                title='Correlation'
            ),
            tooltip=[
                alt.Tooltip('variable1:N', title='Feature 1'),
                alt.Tooltip('variable2:N', title='Feature 2'),
                alt.Tooltip('correlation:Q', format='.3f', title='Correlation')
            ]
        )
        .properties(
            width=600,
            height=600,
            title='Correlation Matrix of Phishing Website Features'
        )
        .configure_view(stroke=None)
        .configure_axis(labelFontSize=10)
    ).interactive()
    return (correlation_heatmap,)


@app.cell
def _(correlation_heatmap, mo):
    heatmap_accordion = mo.accordion({"Expand to see the correlation heatmap": correlation_heatmap})
    return (heatmap_accordion,)


@app.cell
def _(X_train, pd):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Calculate the Variance Inflation Factor for each feature
    _vif_data = pd.DataFrame()
    _vif_data["Feature"] = X_train.columns
    _vif_data["VIF"] = [
        variance_inflation_factor(X_train.values, i) 
        for i in range(X_train.shape[1])
    ]

    # Sort by VIF value (descending)
    _vif_data = _vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    # Add interpretation column
    _vif_data['Multicollinearity'] = _vif_data['VIF'].apply(
        lambda x: 'High (>10)' if x > 10 else ('Moderate (5-10)' if x > 5 else 'Low (<5)')
    )
    return (variance_inflation_factor,)


@app.cell
def _(X_train, alt, pd, y_train):
    from sklearn.feature_selection import mutual_info_classif

    # Calculate mutual information with target
    _mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    _mi_df = pd.DataFrame({
        'Feature': X_train.columns,
        'MI_Score': _mi_scores
    }).sort_values('MI_Score', ascending=False)

    # Create bar chart
    mi_chart = (
        alt.Chart(_mi_df)
        .mark_bar()
        .encode(
            x=alt.X('MI_Score:Q', title='Mutual Information Score'),
            y=alt.Y('Feature:N', sort='-x', title='Feature'),
            color=alt.Color('MI_Score:Q', scale=alt.Scale(scheme='viridis'), legend=None),
            tooltip=[
                alt.Tooltip('Feature:N'),
                alt.Tooltip('MI_Score:Q', format='.3f', title='MI Score')
            ]
        )
        .properties(
            width='container',
            height=600,
            title='Mutual Information: Feature Importance with Target'
        )
        .configure_view(stroke=None)
    )
    return (mi_chart,)


@app.cell
def _(mi_chart, mo):
    mutual_accordion= mo.accordion({"Expand to see the Mutual Information Score": mi_chart})
    return (mutual_accordion,)


@app.cell
def _(X_train, alt, pd, variance_inflation_factor):
    # Plotting the Variance Inflation Factor
    _vif_data_chart = pd.DataFrame()
    _vif_data_chart["Feature"] = X_train.columns
    _vif_data_chart["VIF"] = [
        variance_inflation_factor(X_train.values, i) 
        for i in range(X_train.shape[1])
    ]
    _vif_data_chart = _vif_data_chart.sort_values('VIF', ascending=False)

    # Create color coding based on VIF thresholds
    _vif_data_chart['Severity'] = _vif_data_chart['VIF'].apply(
        lambda x: 'High (>10)' if x > 10 else ('Moderate (5-10)' if x > 5 else 'Low (<5)')
    )

    # Create bar chart
    vif_chart = (
        alt.Chart(_vif_data_chart)
        .mark_bar()
        .encode(
            x=alt.X('VIF:Q', title='Variance Inflation Factor'),
            y=alt.Y('Feature:N', sort='-x', title='Feature'),
            color=alt.Color(
                'Severity:N',
                scale=alt.Scale(
                    domain=['Low (<5)', 'Moderate (5-10)', 'High (>10)'],
                    range=['#2ecc71', '#f39c12', '#e74c3c']
                ),
                title='Multicollinearity Level'
            ),
            tooltip=[
                alt.Tooltip('Feature:N', title='Feature'),
                alt.Tooltip('VIF:Q', format='.2f', title='VIF'),
                alt.Tooltip('Severity:N', title='Level')
            ]
        )
        .properties(
            width='container',
            height=600,
            title='Variance Inflation Factor (VIF) by Feature'
        )
        .configure_view(stroke=None)
    )
    return (vif_chart,)


@app.cell
def _(mo, vif_chart):
    vif_accordion = mo.accordion({"Expand to see the Variance Inflation Factor Barplot": vif_chart})
    return (vif_accordion,)


@app.cell
def _():
    ## mo.md("""## Running model""")
    return


@app.cell
def _(X, mo):
    # 7. Create checkboxes for variable selection
    predictor_names = X.columns.tolist()
    predictor_selector = mo.ui.array([
        mo.ui.checkbox(label=var, value=True) for var in predictor_names
    ])

    predictors = mo.vstack([
        mo.md("### Select Variables for Model"),
        mo.hstack(
            [predictor_selector[i] for i in range(0, len(predictor_selector))],
            widths="equal",
            wrap=True
        )
    ])
    return predictor_names, predictor_selector, predictors


@app.cell
def _(X_train, predictor_names, predictor_selector, sm, y_train):
    # 8. Creating function to fit model
    coli = [predictor_names[j] for j in range(0,len(predictor_selector)) if predictor_selector[j].value]

    def glm_model2(coli):
        X_train2 = X_train[coli]
        X_train_const = sm.add_constant(X_train2)

        model = sm.GLM(
        y_train, 
        X_train_const, 
        family=sm.families.Binomial()
        )
        result = model.fit()
        return result

    result2 = glm_model2(coli)
    return coli, result2


@app.cell
def _(pd, result2):
    # 9. Creating model statistics
    model_stats = pd.DataFrame({
                'Metric': ['Deviance', 'Null Deviance', 'Deviance Explained', 'Log-Likelihood', 'AIC'],
                'Value': [
                    round(result2.deviance,2),
                    round(result2.null_deviance,2),
                    1 - (result2.deviance / result2.null_deviance),
                    round(result2.llf,2),
                    round(result2.aic,2)
                ]
            })
    return (model_stats,)


@app.cell
def _():
    #mo.md("""## Running Predictions""")
    return


@app.cell
def _(X_test, coli, pd, result2, sm, y_test):
    # 10. Making predictions on test set
    X_test_const = sm.add_constant(X_test[coli])

    # Get predicted probabilities
    y_pred_proba = result2.predict(X_test_const)

    # Convert to binary predictions (threshold = 0.5)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    # Create comparison dataframe
    glm_predictions = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted_Probability': y_pred_proba,
        'Predicted_Class': y_pred_binary,
        'Correct': (y_test.values == y_pred_binary)
    })
    return glm_predictions, y_pred_binary, y_pred_proba


@app.cell
def _(pd, y_pred_binary, y_pred_proba, y_test):
    # 11. Calculating model performance metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    glm_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Value': [
            accuracy_score(y_test, y_pred_binary),
            precision_score(y_test, y_pred_binary),
            recall_score(y_test, y_pred_binary),
            f1_score(y_test, y_pred_binary),
            roc_auc_score(y_test, y_pred_proba)
        ]
    })
    return confusion_matrix, glm_metrics


@app.cell
def _(confusion_matrix, pd, y_pred_binary, y_test):
    # 12. Creating confusion matrix visualization
    conf_matrix = confusion_matrix(y_test, y_pred_binary)

    confusion_df = pd.DataFrame({
        'Predicted_Legitimate': [conf_matrix[0, 0], conf_matrix[1, 0]],
        'Predicted_Phishing': [conf_matrix[0, 1], conf_matrix[1, 1]]
    }, index=['Actual_Legitimate', 'Actual_Phishing'])
    return (confusion_df,)


@app.cell
def _(mo):
    mo.md("""
    ## **Feature Selection**
    """)
    return


@app.cell
def _(heatmap_accordion, mo):
    mo.vstack([mo.md("### Correlation Heatmap"),heatmap_accordion])
    return


@app.cell
def _(mo, vif_accordion):
    mo.vstack([mo.md("### Variance Inflation Factor"),vif_accordion])
    return


@app.cell
def _(mo, mutual_accordion):
    mo.vstack([mo.md("### Mutual Information Classification"),mutual_accordion])
    return


@app.cell
def _(mo):
    mo.md("""
    ## **Dashboard**
    """)
    return


@app.cell
def _(confusion_df, glm_metrics, mo, model_stats):
    metrics = mo.hstack([model_stats,glm_metrics,confusion_df], align="center", wrap=True, widths=[2,2,5])
    return (metrics,)


@app.cell
def _(alt, glm_predictions):
    # Create logistic curve visualization
    # Sort predictions by probability for smooth curve
    _sorted_predictions = glm_predictions.sort_values('Predicted_Probability').reset_index(drop=True)
    _sorted_predictions['observation'] = range(len(_sorted_predictions))

    # Create the logistic curve chart
    logistic_chart = (
        alt.Chart(_sorted_predictions)
        .mark_line(color='red', strokeWidth=2)
        .encode(
            x=alt.X('observation:Q', title='Observation (sorted by probability)'),
            y=alt.Y('Predicted_Probability:Q', title='Probability', scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip('observation:Q', title='Observation'),
                alt.Tooltip('Predicted_Probability:Q', format='.3f', title='Predicted Probability')
            ]
        )
    )

    # Overlay actual values as points
    actual_points = (
        alt.Chart(_sorted_predictions)
        .mark_circle(size=20, opacity=0.5)
        .encode(
            x=alt.X('observation:Q'),
            y=alt.Y('Actual:Q', title='Actual Value'),
            color=alt.Color('Actual:N',
                           scale=alt.Scale(domain=[0, 1], range=['#1f77b4', '#ff7f0e']),
                           legend=alt.Legend(title='Actual Class',
                                            labelExpr="datum.value == 0 ? 'Pishing (0)' : 'Legitimate (1)'")),
            tooltip=[
                alt.Tooltip('observation:Q', title='Observation'),
                alt.Tooltip('Actual:N', title='Actual Class'),
                alt.Tooltip('Predicted_Probability:Q', format='.3f', title='Predicted Probability')
            ]
        )
    )

    # Combine both layers
    logis = (logistic_chart + actual_points).properties(
        width='container',
        height=400,
        title='Logistic Regression: Predicted Probabilities (Red Curve) vs Actual Values (Points)'
    ).configure_view(stroke=None)
    return (logis,)


@app.cell
def _(X_test, alt, column_selector, y_pred_proba, y_test):
    # Alternative: Plot against a specific predictor variable
    # Choose a predictor to visualize (e.g., 'url_length')
    predictor_var = column_selector.value

    # Combine test data with predictions
    _test_with_pred = X_test[[predictor_var]].copy()
    _test_with_pred['Actual'] = y_test.values
    _test_with_pred['Predicted_Probability'] = y_pred_proba

    # Aggregate to show clearer pattern
    _aggregated_by_predictor = (
        _test_with_pred
        .groupby(predictor_var)
        .agg({'Predicted_Probability': 'mean', 'Actual': 'mean'})
        .reset_index()
    )

    # Create logistic curve by predictor
    predictor_logistic = (
        alt.Chart(_aggregated_by_predictor)
        .mark_line(color='red', strokeWidth=3)
        .encode(
            x=alt.X(f'{predictor_var}:Q', title=predictor_var),
            y=alt.Y('Predicted_Probability:Q', title='Mean Predicted Probability', scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip(f'{predictor_var}:Q'),
                alt.Tooltip('Predicted_Probability:Q', format='.3f', title='Mean Pred. Probability')
            ]
        )
    )

    # Overlay actual mean values
    actual_line = (
        alt.Chart(_aggregated_by_predictor)
        .mark_line(color='blue', strokeWidth=2, strokeDash=[5, 5])
        .encode(
            x=alt.X(f'{predictor_var}:Q'),
            y=alt.Y('Actual:Q', title='Mean Actual Value'),
            tooltip=[
                alt.Tooltip(f'{predictor_var}:Q'),
                alt.Tooltip('Actual:Q', format='.3f', title='Mean Actual')
            ]
        )
    )

    pred_log = (predictor_logistic + actual_line).properties(
        width='container',
        height=400,
        title=f'Logistic Function: Predicted (Red) vs Actual (Blue Dashed) by {predictor_var}'
    ).configure_view(stroke=None)
    return (pred_log,)


@app.cell
def _(column_description, logis, metrics, mo, pred_log, predictors):
    mo.hstack([
        mo.vstack([predictors,metrics]),
        mo.vstack([logis, column_description,pred_log])
    ])
    return


if __name__ == "__main__":
    app.run()
